"""AWQ INT4 quantization — memory-safe version.

Reads weights directly from safetensors files on disk, one tensor at a time,
instead of loading the full model into GPU memory. Process is:

1. Load one weight tensor from disk as a numpy array
2. Load its corresponding AWQ scale from the scales dict
3. Quantize on CPU (no GPU needed for this step)
4. Save quantized result
5. Free memory → next tensor

This keeps peak memory under ~200 MB for the quantizer itself.
"""

import gc
import json
import os
from typing import Any

import torch

from utils.memory import log_memory, get_device
from utils.errors import QuantizationError


def find_safetensors_files(model_path: str) -> list[str]:
    """Find all .safetensors files in a model directory, ordered by shard index."""
    import glob

    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    return files





def iter_weights(model_path: str):
    """Yield (tensor_name, safetensors_file_path) for all weight tensors.

    Only yields tensors belonging to nn.Linear modules (excludes
    embeddings, norms, and other non-linear parameters).
    """
    import safetensors

    safetensors_files = find_safetensors_files(model_path)

    for sf_path in safetensors_files:
        with safetensors.safe_open(sf_path, framework="pt") as f:
            tensor_names = f.keys()

        for name in tensor_names:
            if not name.endswith(".weight"):
                continue
            if "embed" in name:
                continue
            if "norm" in name or "layernorm" in name or "rmsnorm" in name:
                continue
            yield name, sf_path


def load_weight_from_safetensors(sf_path: str, tensor_name: str) -> torch.Tensor:
    """Load a single weight tensor from a safetensors file."""
    import safetensors

    with safetensors.safe_open(sf_path, framework="pt") as f:
        tensor = f.get_tensor(tensor_name)
    return tensor


def normalize_safetensors_name(name: str) -> str:
    """Convert safetensors weight name to calibration stat key.

    Example: model.language_model.layers.0.mlp.gate_proj.weight
          → model.layers.0.mlp.gate_proj
    """
    import re
    key = name[:-len(".weight")] if name.endswith(".weight") else name
    # Collapse model.language_model / model.model / model prefixes to model.
    key = re.sub(r'^(?:model\.language_model\.|model\.model\.|model\.)', 'model.', key)
    return key


def quantize_layer_cpu(
    weight: torch.Tensor,
    scale_factors: torch.Tensor,
    group_size: int = 32,
) -> dict:
    """Quantize a single linear layer on CPU.

    Args:
        weight: Weight matrix, shape [d_out, d_in].
        scale_factors: AWQ per-channel scale factors, shape [d_in].
        group_size: INT4 group size.

    Returns:
        Quantized layer dict with packed weights, group scales, etc.
    """
    d_out, d_in = weight.shape

    # AWQ scaling: scale salient weight channels UP before quantization
    # (W' = W * s) so they occupy more of the INT4 range; the inverse (1/s) is
    # applied at dequantization time. Activations would be divided by s at
    # inference, but here that is folded into the stored weight, so the
    # dequantized weight is Q(W * s) / s ≈ W.
    s = scale_factors.to(dtype=weight.dtype)
    w_scaled = weight * s.unsqueeze(0)

    # Group-wise INT4 quantization
    packed_groups: list[torch.Tensor] = []
    group_scales: list[torch.Tensor] = []

    for g_start in range(0, d_in, group_size):
        g_end = min(g_start + group_size, d_in)
        w_group = w_scaled[:, g_start:g_end]

        if w_group.size(1) < group_size:
            pad_size = group_size - w_group.size(1)
            w_group = torch.nn.functional.pad(w_group, (0, pad_size))

        # INT4 quantize — per-row-per-group scales (matches _sym_quantize in export.py)
        group_max = w_group.abs().amax(dim=1, keepdim=True)  # [d_out, 1]
        group_max = group_max.clamp(min=1e-10)
        qscale = group_max / 7.0  # [d_out, 1]

        w_int4 = (w_group / qscale).round().clamp(-7, 7).to(torch.int8)

        # Pack two INT4 into one INT8
        w_paired = w_int4.contiguous().view(d_out, group_size // 2, 2)
        packed = (w_paired[:, :, 0] & 0x0F) | ((w_paired[:, :, 1] & 0x0F) << 4)

        packed_groups.append(packed.to(torch.uint8))
        group_scales.append(qscale.to(dtype=torch.float16))  # [d_out, 1] per group

    return {
        "packed_weights": packed_groups,
        "group_scales": group_scales,
        "scale_factors": s.to(dtype=torch.float16),
        "shape": (d_out, d_in),
        "group_size": group_size,
    }


def quantize_all_layers(
    model_path: str,
    scales: dict[str, torch.Tensor],
    group_size: int = 32,
    output_dir: str | None = None,
    verbose: bool = True,
    device: str = "cpu",
) -> dict[str, dict]:
    """Quantize all linear layers reading weights directly from safetensors.

    NEVER loads the full model. Processes one weight tensor at a time.

    Args:
        model_path: Path to the FP16 model directory.
        scales: {layer_name: scale_factors} from compute_all_scales().
        group_size: INT4 group size.
        output_dir: Where to save quantized state. If None, not saved to disk.
        verbose: Print progress.
        device: Device for dequantization ("cpu" or "cuda"). For quantization
            itself, CPU is always used for memory safety.

    Returns:
        Quantized state dict: {normalized_layer_name: quantized_dict}.
    """
    quantized_state: dict[str, dict] = {}
    total_fp16_bytes = 0
    total_quantized_bytes = 0

    if verbose:
        print(f"Quantizing model at {model_path}")
        print(f"  Using {len(scales)} layer scales")

    for tensor_name, sf_path in iter_weights(model_path):
        calib_key = normalize_safetensors_name(tensor_name)

        if calib_key not in scales:
            if verbose:
                print(f"  [SKIP] {tensor_name} → no scale found")
            continue

        scale = scales[calib_key]

        if verbose:
            print(f"  Loading {tensor_name}...", end=" ", flush=True)

        # Load one weight tensor from disk
        weight = load_weight_from_safetensors(sf_path, tensor_name)

        if verbose:
            print(f"shape={list(weight.shape)} → quantizing...", end=" ", flush=True)

        # Quantize on CPU
        q = quantize_layer_cpu(weight, scale, group_size=group_size)
        quantized_state[calib_key] = q

        # Stats — count the FULL INT4 footprint: packed weights (uint8/byte)
        # plus per-group FP16 scales plus per-channel AWQ scale factors (FP16).
        # Ignoring the scale metadata overstates the compression ratio.
        d_out, d_in = q["shape"]
        fp16_bytes = d_out * d_in * 2
        quantized_bytes = (
            sum(p.numel() * p.element_size() for p in q["packed_weights"])       # packed int4
            + sum(s.numel() * s.element_size() for s in q["group_scales"])        # per-group fp16 scales
            + q["scale_factors"].numel() * q["scale_factors"].element_size()       # per-channel AWQ scales
        )
        total_fp16_bytes += fp16_bytes
        total_quantized_bytes += quantized_bytes

        ratio = fp16_bytes / max(quantized_bytes, 1)
        if verbose:
            print(f"✓ {fp16_bytes/1e6:.1f}MB → {quantized_bytes/1e6:.1f}MB ({ratio:.1f}×)")

        # Free memory
        del weight, q
        gc.collect()

    total_ratio = total_fp16_bytes / max(total_quantized_bytes, 1)
    if verbose:
        print(f"\n  ── Compression Summary ──")
        print(f"  Layers quantized: {len(quantized_state)}")
        print(f"  FP16 total:  {total_fp16_bytes / 1e6:.1f} MB")
        print(f"  INT4 total:  {total_quantized_bytes / 1e6:.1f} MB")
        print(f"  Compression: {total_ratio:.1f}× ({100 / total_ratio:.1f}% of original)")

    # Save
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        meta = {
            "model": os.path.basename(model_path.rstrip("/")),
            "method": "AWQ + INT4 group-wise",
            "group_size": group_size,
            "alpha": "grid_search (default)" if True else 0.5,  # TODO: plumb actual no_grid_search flag from CLI
            "layers_quantized": len(quantized_state),
            "fp16_mb": round(total_fp16_bytes / 1e6, 1),
            "int4_mb": round(total_quantized_bytes / 1e6, 1),
            "compression_ratio": round(total_ratio, 1),
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        torch.save(quantized_state, os.path.join(output_dir, "quantized_state.pt"))
        if verbose:
            print(f"\nSaved quantized model → {os.path.join(output_dir, 'quantized_state.pt')}")

    return quantized_state


def dequantize_layer(q: dict) -> torch.Tensor:
    """Dequantize a full quantized layer back to FP16 weights (verify-only).

    Inverts ``quantize_layer_cpu``: the stored INT4 encodes ``W * s`` (AWQ
    scales weights UP before quantization), so the group-dequantized tensor is
    divided by ``s`` to recover ``W ≈ Q(W * s) / s``.

    Used by ``verify_reconstruction`` to measure per-layer reconstruction MSE
    against the original FP16 weights on disk. This is NOT an inference path:
    running a model from the quantized artifact is done via ``awq export`` to a
    real INT4 runtime (AutoAWQ / vLLM), which folds ``s`` into the preceding
    norm instead of dividing weights by ``s`` (see ``awq/export.py``).

    Args:
        q: Quantized layer dict with keys 'packed_weights', 'group_scales',
            'scale_factors', 'shape', 'group_size'.

    Returns:
        FP16 weight matrix of shape [d_out, d_in].
    """
    d_out, d_in = q["shape"]
    group_size = q["group_size"]

    parts = []
    for packed, qscale in zip(q["packed_weights"], q["group_scales"]):
        parts.append(_dequantize_group(packed, qscale, d_out, group_size))

    w = torch.cat(parts, dim=1)[:, :d_in]

    # Invert the AWQ scale: stored = Q(W * s), so W ≈ dequant / s
    s = q["scale_factors"].to(dtype=torch.float16)
    w = w / s.unsqueeze(0)

    return w


def verify_reconstruction(
    quantized_state: dict[str, dict],
    model_path: str,
    num_layers: int = 3,
    verbose: bool = True,
) -> dict[str, float]:
    """Verify quantization by comparing a few dequantized layers against disk.

    Processes one layer at a time — never loads the full model.
    """
    errors: dict[str, float] = {}
    checked = 0

    for tensor_name, sf_path in iter_weights(model_path):
        calib_key = normalize_safetensors_name(tensor_name)

        if calib_key not in quantized_state:
            continue
        if checked >= num_layers:
            break

        q = quantized_state[calib_key]

        # Load original weight
        fp16_weight = load_weight_from_safetensors(sf_path, tensor_name).float()

        # Dequantize via the canonical path (includes AWQ 1/s) so the MSE
        # reflects quantization error, not scale magnitude.
        deq_weight = dequantize_layer(q).float()

        mse = (fp16_weight - deq_weight).pow(2).mean().item()
        errors[calib_key] = mse

        if verbose:
            print(f"  {calib_key:<55} MSE={mse:.8f}")

        checked += 1

    return errors


def _dequantize_group(
    packed: torch.Tensor,
    group_scale: torch.Tensor,
    d_out: int,
    group_size: int,
) -> torch.Tensor:
    """Dequantize a packed INT4 group to FP16."""
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)

    w_deq = torch.stack([low, high], dim=-1).reshape(d_out, group_size)
    w_deq = w_deq.to(dtype=torch.float16) * group_scale.to(dtype=torch.float16)
    return w_deq
