"""AWQ INT4 quantization — memory-safe version (Issue #8).

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

import numpy as np
import torch

from utils.memory import log_memory

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def find_safetensors_file(model_path: str) -> list[str]:
    """Find all .safetensors files in a model directory, ordered by shard index."""
    import glob

    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    return files


def load_index_file(model_path: str) -> dict:
    """Load the safetensors index JSON to map tensor names to filenames."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index_data = json.load(f)
        return index_data.get("weight_map", {})
    return {}


def iter_weights(model_path: str):
    """Yield (tensor_name, numpy_array) for all linear weight tensors.

    Uses safetensors to read one tensor at a time from disk.
    Only yields tensors belonging to nn.Linear modules (excludes
    embeddings, norms, and other non-linear parameters).
    """
    import safetensors

    safetensors_files = find_safetensors_file(model_path)
    weight_map = load_index_file(model_path)

    # Determine which tensor names to look for — only linear layer weights
    for sf_path in safetensors_files:
        # Use safetensors to get the list of tensors in this file
        with safetensors.safe_open(sf_path, framework="pt") as f:
            tensor_names = f.keys()

        for name in tensor_names:
            # Skip non-weight tensors (embeddings, norms, biases, etc.)
            # We only want Linear layer weights (have "weight" suffix and are in a linear-like path)
            if not name.endswith(".weight"):
                continue
            # Skip embedding layers (model.embed_tokens)
            if "embed" in name:
                continue
            # Skip norm layers (they don't have weights)
            if "norm" in name or "layernorm" in name or "rmsnorm" in name:
                continue
            # Skip lm_head (we keep it as-is in our AWQ wrapper)
            # Actually include it for now
            if name == "lm_head.weight":
                # Include lm_head — it's a linear layer
                pass

            yield name, sf_path


def load_weight_from_safetensors(sf_path: str, tensor_name: str) -> torch.Tensor:
    """Load a single weight tensor from a safetensors file."""
    import safetensors

    with safetensors.safe_open(sf_path, framework="pt") as f:
        tensor = f.get_tensor(tensor_name)
    return tensor


def quantize_layer_cpu(
    weight: torch.Tensor,
    scale_factors: torch.Tensor,
    group_size: int = 32,
) -> dict:
    """Quantize a single linear layer on CPU.

    Same logic as quantize.py but designed to work with CPU tensors.
    """
    d_out, d_in = weight.shape

    # Apply AWQ scaling: W' = W / s
    s = scale_factors.to(dtype=weight.dtype)
    w_scaled = weight / s.unsqueeze(0)

    # Group-wise INT4 quantization
    packed_groups: list[torch.Tensor] = []
    group_scales: list[torch.Tensor] = []

    for g_start in range(0, d_in, group_size):
        g_end = min(g_start + group_size, d_in)
        w_group = w_scaled[:, g_start:g_end]

        if w_group.size(1) < group_size:
            pad_size = group_size - w_group.size(1)
            w_group = torch.nn.functional.pad(w_group, (0, pad_size))

        # INT4 quantize
        group_max = w_group.abs().max()
        if group_max < 1e-10:
            group_max = 1e-10
        qscale = group_max / 7.0

        w_int4 = (w_group / qscale).round().clamp(-7, 7).to(torch.int8)

        # Pack two INT4 into one INT8
        w_paired = w_int4.view(d_out, group_size // 2, 2)
        packed = (w_paired[:, :, 0] & 0x0F) | ((w_paired[:, :, 1] & 0x0F) << 4)

        packed_groups.append(packed.to(torch.uint8))
        group_scales.append(qscale.to(dtype=torch.float16))

    return {
        "packed_weights": packed_groups,
        "group_scales": group_scales,
        "scale_factors": s.to(dtype=torch.float16),
        "shape": (d_out, d_in),
        "group_size": group_size,
    }


def quantize_all_layers_memory_safe(
    model_path: str,
    scales: dict[str, torch.Tensor],
    group_size: int = 32,
    output_dir: str | None = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """Quantize all linear layers reading weights directly from safetensors.

    NEVER loads the full model. Processes one weight tensor at a time.

    Args:
        model_path: Path to the FP16 model directory.
        scales: {layer_name: scale_factors} dict.
        group_size: INT4 group size.
        output_dir: Where to save quantized state.
        verbose: Print progress.

    Returns:
        Quantized state dict.
    """
    quantized_state: dict[str, dict] = {}
    total_fp16_bytes = 0
    total_quantized_bytes = 0

    if verbose:
        print(f"Quantizing model at {model_path}")
        print(f"  Using {len(scales)} layer scales")

    # Build a name map: safetensors name → calibration name
    # Calibration stats use "model.layers.X.xxx" but safetensors use
    # "model.language_model.layers.X.xxx.weight"
    def normalize_safetensors_name(name: str) -> str:
        """Convert safetensors weight name to calibration stat key."""
        # Remove .weight suffix
        key = name.replace(".weight", "")
        # Remove "language_model." prefix if present
        key = key.replace("model.language_model.", "model.")
        return key

    # Explicitly skip layers: lm_head, tiny projections, and even-numbered layers
    SKIP_LAYERS = {"lm_head", "model.lm_head", "model.model.lm_head"}
    # Tiny projections (d_out=16)
    for i in range(0, 24, 2):  # only even layers that also have linear_attn
        for name in ("in_proj_a", "in_proj_b"):
            SKIP_LAYERS.add(f"model.layers.{i}.linear_attn.{name}")
    # Even-numbered layers → keep in FP16 to prevent error compounding
    for i in range(0, 24, 2):  # 0, 2, 4, ..., 22
        for key in list(scales.keys()):
            if f"model.layers.{i}." in key:
                SKIP_LAYERS.add(key)

    for tensor_name, sf_path in iter_weights(model_path):
        calib_key = normalize_safetensors_name(tensor_name)
        if calib_key in SKIP_LAYERS:
            if verbose:
                print(f"  [SKIP] {tensor_name}")
            continue
        if calib_key not in scales:
            if verbose:
                print(f"  [SKIP] {tensor_name} → {calib_key} — no scale found")
            continue

        scale = scales[calib_key]

        if verbose:
            print(f"  Loading {tensor_name}...", end=" ", flush=True)

        # Load one weight tensor from disk
        weight = load_weight_from_safetensors(sf_path, tensor_name)

        if verbose:
            print(f"shape={list(weight.shape)} → quantizing...", end=" ", flush=True)

        # Quantize
        q = quantize_layer_cpu(weight, scale, group_size=group_size)
        quantized_state[calib_key] = q

        # Stats
        d_out, d_in = q["shape"]
        fp16_bytes = d_out * d_in * 2
        quantized_bytes = sum(p.numel() for p in q["packed_weights"])
        total_fp16_bytes += fp16_bytes
        total_quantized_bytes += quantized_bytes

        ratio = fp16_bytes / max(quantized_bytes, 1)
        if verbose:
            print(f"✓ {fp16_bytes/1e6:.1f}MB → {quantized_bytes/1e6:.1f}MB ({ratio:.1f}×)")

        # Free memory
        del weight, q
        gc.collect()

    if verbose:
        total_ratio = total_fp16_bytes / max(total_quantized_bytes, 1)
        print(f"\n  ── Compression Summary ──")
        print(f"  Layers quantized: {len(quantized_state)}")
        print(f"  FP16 total:  {total_fp16_bytes / 1e6:.1f} MB")
        print(f"  INT4 total:  {total_quantized_bytes / 1e6:.1f} MB")
        print(f"  Compression: {total_ratio:.1f}× ({100 / total_ratio:.1f}% of original)")

    # Save
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "qwen_awq_int4")
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata separately (not in the same dict to keep loading fast)
    meta = {
        "model": model_path.split("/")[-1],
        "method": "AWQ + INT4 group-wise",
        "group_size": group_size,
        "alpha": 0.5,
        "layers_quantized": len(quantized_state),
        "fp16_mb": round(total_fp16_bytes / 1e6, 1),
        "int4_mb": round(total_quantized_bytes / 1e6, 1),
        "compression_ratio": round(total_ratio, 1),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save quantized state
    torch.save(quantized_state, os.path.join(output_dir, "quantized_state.pt"))
    if verbose:
        print(f"\nSaved quantized model → {os.path.join(output_dir, 'quantized_state.pt')}")
        log_memory("post_quantize")

    return quantized_state


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
        if tensor_name not in quantized_state:
            continue
        if checked >= num_layers:
            break

        q = quantized_state[tensor_name]
        d_out, d_in = q["shape"]
        group_size = q["group_size"]

        # Load original weight
        fp16_weight = load_weight_from_safetensors(sf_path, tensor_name).float()

        # Dequantize
        deq_parts: list[torch.Tensor] = []
        for g_idx, (packed, qscale) in enumerate(zip(q["packed_weights"], q["group_scales"])):
            w_deq = _dequantize_group(packed, qscale, d_out, group_size)
            deq_parts.append(w_deq)

        deq_weight = torch.cat(deq_parts, dim=1)[:, :d_in]

        mse = (fp16_weight - deq_weight).pow(2).mean().item()
        max_err = (fp16_weight - deq_weight).abs().max().item()
        errors[tensor_name] = mse

        if verbose:
            print(f"  {tensor_name:<55} MSE={mse:.8f}  max_abs_err={max_err:.4f}")

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


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    MODEL_PATH = "models/Qwen3.5-2B-FP16"
    SCALES_PATH = os.path.join(RESULTS_DIR, "awq_scales.pt")

    print(f"Loading scales from {SCALES_PATH}...")
    scales = torch.load(SCALES_PATH, map_location="cpu", weights_only=True)
    print(f"  {len(scales)} layer scales loaded")

    # Quantize — never loads the full model
    log_memory("before_quantize")
    quantized = quantize_all_layers_memory_safe(
        MODEL_PATH,
        scales,
        group_size=32,
        verbose=True,
    )

    # Quick verification on a few layers
    print("\nVerifying reconstruction quality (3 layers)...")
    errors = verify_reconstruction(quantized, MODEL_PATH, num_layers=3)
    avg_mse = sum(errors.values()) / max(len(errors), 1)
    print(f"\n  Average MSE: {avg_mse:.8f}")

    print(f"\n✅ Quantization complete. {len(quantized)} layers quantized.")
    log_memory("final")
