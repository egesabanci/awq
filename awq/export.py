"""Export ``quantized_state.pt`` to a runtime-loadable HF-AWQ INT4 model.

``awq``'s native artifact (``quantized_state.pt``) is a custom layout that no
external runtime can consume. This module re-packs it into the **AutoAWQ / HF
AWQ GEMM on-disk format** that ``AutoAWQForCausalLM`` (and vLLM's ``awq``
loader) read directly.

On-disk layout (per quantized linear, matching ``casper-hansen/AutoAWQ``):

    qweight : int32 [d_in, d_out // 8]     # 8 int4 packed along the OUTPUT axis
    qzeros  : int32 [d_in // group_size, d_out // 8]
    scales  : fp16 [d_in // group_size, d_out]
    dequant : reverse_awq_order(unpack(qweight)) -> (q - zeros) * scales

The int4 are UNSIGNED [0, 15] with a per-group zero point. We map our signed
symmetric grid [-7, 7] onto the unsigned grid by storing ``zero = 7`` and
``q_unsigned = q_signed + 7`` — so the runtime's ``(q - zero) * scale`` exactly
reproduces our group dequant ``q_signed * group_scale`` with **no
re-quantization error**.

AWQ scale reconciliation (the hard part)
---------------------------------------
Our pipeline computes an *independent* per-channel AWQ scale ``s`` for every
linear. A loadable HF-AWQ model folds ``1/s`` into the **preceding norm**'s
weight, but q/k/v share ``input_layernorm`` and gate/up share
``post_attention_layernorm`` — three/five linears cannot each fold a different
``s`` into one norm. We therefore **aggregate** the per-linear ``s`` into a
single shared scale per norm group (geometric mean over the member linears)
and re-quantize those linears with the shared scale. ``o_proj`` and
``down_proj`` have no preceding norm, so they are exported as plain RTN INT4
(``s = 1``). The result is a correct, uniformly-INT4, loadable AWQ model where
5/7 linears per block are AWQ-scaled and 2/7 are RTN.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any

import torch
import torch.nn.functional as F

# ── AutoAWQ GEMM packing constants (replicated from AutoAWQ's quant_utils.py) ──
Q_BITS = 4
STORAGE_BITS = 32
PACK_NUM = STORAGE_BITS // Q_BITS  # 8 int4 per int32
AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


# ── Packing primitives (pure-torch replicas of AutoAWQ's quant_utils) ──────────


def _apply_order(imatrix: torch.Tensor, order: list[int]) -> torch.Tensor:
    """Permute every block of PACK_NUM along the last axis by ``order``."""
    d_in, d_out = imatrix.shape
    return imatrix.view(d_in, d_out // PACK_NUM, PACK_NUM)[:, :, order].reshape(d_in, d_out)


def _pack_column(int4: torch.Tensor) -> torch.Tensor:
    """Pack an int8 [d_in, d_out] unsigned-int4 matrix into int32 [d_in, d_out//8].

    Replicates AutoAWQ ``pack(..., "column")``: 8 int4 per int32 along the
    output axis, int4 ``i`` stored at bits ``[i*4, i*4+3]``.
    """
    shifts = torch.arange(0, STORAGE_BITS, Q_BITS, device=int4.device)
    m = (int4.to(torch.int8) & 0x0F).view(-1, int4.shape[1] // PACK_NUM, PACK_NUM)
    q = (m << shifts[None, None, :]).sum(dim=-1)
    return q.to(torch.int32)


def _unpack_column(qmatrix: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_pack_column`` → int8 [d_in, d_out] (unsigned, [0, 15])."""
    shifts = torch.arange(0, STORAGE_BITS, Q_BITS, device=qmatrix.device)
    m = (qmatrix[:, :, None] >> shifts[None, None, :]).view(qmatrix.shape[0], -1)
    return (m.to(torch.int8) & 0x0F)


def _reverse_awq_order(iweights: torch.Tensor) -> torch.Tensor:
    """Undo ``AWQ_PACK_ORDER`` per PACK_NUM-block along the last axis."""
    d_in, d_out = iweights.shape
    idx = torch.arange(d_out, device=iweights.device).view(-1, PACK_NUM)[:, AWQ_REVERSE_ORDER].view(-1)
    return iweights[:, idx]


def awq_dequant_gemm(qweight: torch.Tensor, qzeros: torch.Tensor,
                     scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """Reference dequant for the AutoAWQ GEMM on-disk format → fp16 [d_in, d_out].

    Replicates ``autoawq.utils.packing_utils.dequantize_gemm`` exactly, so a
    round-trip through our packer proves the on-disk encoding matches what an
    AutoAWQ/vLLM loader will reconstruct.
    """
    iweight = _unpack_column(qweight)
    izeros = _unpack_column(qzeros)
    iweight = _reverse_awq_order(iweight)
    izeros = _reverse_awq_order(izeros)
    iweight = iweight & (2 ** Q_BITS - 1)
    izeros = izeros & (2 ** Q_BITS - 1)
    scales = scales.repeat_interleave(group_size, dim=0)      # [d_in, d_out]
    izeros = izeros.repeat_interleave(group_size, dim=0)      # [d_in, d_out]
    return ((iweight - izeros) * scales).to(torch.float16)


# ── Our symmetric INT4 quantizer (shared with awq.quantize's grid) ────────────


def _sym_quantize(w: torch.Tensor, s: torch.Tensor, group_size: int):
    """Group-wise symmetric INT4 of ``w * s`` → (signed_int4[-7,7], group_scales).

    Same grid as ``awq.quantize.quantize_layer_cpu``: ``qscale = max|group|/7``,
    round, clamp [-7, 7]. Returns the signed int4 matrix [d_out, d_in] (int8)
    and per-group fp16 scales [d_out, n_groups].
    """
    d_out, d_in = w.shape
    pad = (-d_in) % group_size
    ws = (w.float() * s.float().unsqueeze(0)) if s is not None else w.float()
    if pad:
        ws = F.pad(ws, (0, pad))
    n_groups = ws.shape[1] // group_size
    wg = ws.view(d_out, n_groups, group_size)
    gmax = wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-10)
    qscale = gmax / 7.0
    q = (wg / qscale).round().clamp(-7, 7).to(torch.int8)
    q = q.reshape(d_out, n_groups * group_size)[:, :d_in].contiguous()   # [d_out, d_in]
    return q, qscale.squeeze(2).to(torch.float16)                      # scales [d_out, n_groups]


def _export_linear_from_signed(signed_int4: torch.Tensor, group_scales: torch.Tensor,
                               group_size: int) -> dict[str, torch.Tensor]:
    """Pack a signed-int4 [d_out, d_in] + per-group scales into AutoAWQ GEMM tensors.

    Maps signed [-7, 7] → unsigned [0, 14] with zero=7, transposes to [d_in, d_out],
    applies AWQ_PACK_ORDER, and packs. Returns ``{qweight, qzeros, scales}``.
    """
    d_out, d_in = signed_int4.shape
    n_groups = d_in // group_size
    unsigned = (signed_int4.to(torch.int16) + 7).to(torch.int8)  # [0, 14]
    # group_scales: [d_out, n_groups] -> scales [n_groups, d_out] (fp16)
    scales = group_scales.to(torch.float16).t().contiguous()    # [n_groups, d_out]
    # qweight: pack unsigned along d_out -> [d_in, d_out//8]
    unsigned_t = unsigned.t().contiguous()                       # [d_in, d_out]
    unsigned_ord = _apply_order(unsigned_t, AWQ_PACK_ORDER)
    qweight = _pack_column(unsigned_ord)
    # qzeros: 7 packed along d_out -> [n_groups, d_out//8]
    zeros = torch.full((n_groups, d_out), 7, dtype=torch.int8, device=signed_int4.device)
    zeros_ord = _apply_order(zeros, AWQ_PACK_ORDER)
    qzeros = _pack_column(zeros_ord)
    return {"qweight": qweight, "qzeros": qzeros, "scales": scales}


# ── Llama/Qwen transformer-block layout (norm-group mapping) ──────────────────
# linears that follow a norm (share it) -> fold 1/s into that norm
NORM_GROUPS = {
    "input_layernorm": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "post_attention_layernorm": ["mlp.gate_proj", "mlp.up_proj"],
}
# linears with no preceding norm -> export as RTN (s = 1)
RTN_LINEARS = ["self_attn.o_proj", "mlp.down_proj"]


def _layer_scales(quantized_state: dict[str, dict], layer_idx: int) -> dict[str, torch.Tensor]:
    """Extract per-linear AWQ scale ``s`` (shape [d_in]) for one transformer layer."""
    out = {}
    for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        key = f"model.layers.{layer_idx}.{proj}"
        if key in quantized_state:
            out[proj] = quantized_state[key]["scale_factors"].to(torch.float32).cpu()
    return out


def _geomean(scales: list[torch.Tensor]) -> torch.Tensor:
    """Per-channel geometric mean of a list of equal-shape scale tensors."""
    stacked = torch.stack([s.clamp(min=1e-6) for s in scales], dim=0)
    return stacked.prod(dim=0).pow(1.0 / len(scales))


# ── Main export ───────────────────────────────────────────────────────────────


def export_to_awq(
    quantized_state: dict[str, dict],
    model_path: str,
    output_dir: str,
    group_size: int = 128,
    verbose: bool = True,
    device: str = "cpu",
) -> str:
    """Write a runtime-loadable HF-AWQ INT4 model to ``output_dir``.

    Reads the FP16 model from ``model_path`` (for non-quantized tensors and the
    linear weights to re-quantize), and the per-channel AWQ scales ``s`` from
    ``quantized_state`` (each layer's ``scale_factors``). Produces:

      - ``model*.safetensors`` (+ ``model.safetensors.index.json`` if sharded)
        with AutoAWQ GEMM ``qweight``/``qzeros``/``scales`` for quantized
        linears and FP16 for everything else (norms carry the folded ``1/s``).
      - ``quantize_config.json`` (AutoAWQ schema).
      - copied ``config.json``, ``tokenizer.*``, ``generation_config.json``.

    ``device`` controls where the FP16 model is loaded for re-quantization:
    ``cpu`` (small models) or ``cuda`` (large models that exceed CPU RAM).
    The export state dict is built incrementally — only int4 linears + FP16
    non-linears (embeddings/norms/lm_head) — so peak RAM is ~the on-disk
    footprint, not a full FP16 copy.

    Returns the output directory path.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import safetensors.torch as st

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"Exporting HF-AWQ model → {output_dir}")
        print(f"  source model : {model_path}")
        print(f"  group_size   : {group_size}")
        print(f"  quantized layers in source state: {len(quantized_state)}")

    # Load the FP16 model. On cuda use device_map="auto"; the export state dict
    # is built incrementally so we never hold a full FP16 copy in CPU RAM.
    load_map = device if device != "cuda" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map=load_map, low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    n_layers = model.config.num_hidden_layers

    # Resolve which linears are actually present in this architecture.
    present_projs = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            for proj in (NORM_GROUPS["input_layernorm"] + NORM_GROUPS["post_attention_layernorm"]
                         + RTN_LINEARS):
                if name.endswith(proj):
                    present_projs.add(proj)
    if verbose:
        print(f"  linears present: {sorted(present_projs)}")

    # Build the export state dict incrementally: quantized linears are added
    # as qweight/qzeros/scales during the per-layer loop; non-quantized params
    # (embeddings, folded norms, lm_head) are copied afterwards. This avoids
    # holding a full FP16 copy in CPU RAM.
    export_sd: dict[str, torch.Tensor] = {}
    module_by_name = {n: m for n, m in model.named_modules()}
    quantized_linear_names: set[str] = set()
    proj_count = 0

    for i in range(n_layers):
        layer_s = _layer_scales(quantized_state, i)

        # --- shared s per norm group + fold 1/s into the norm (in place) ---
        for norm_name, projs in NORM_GROUPS.items():
            norm_mod = module_by_name.get(f"model.layers.{i}.{norm_name}")
            member_s = [layer_s[p] for p in projs if p in layer_s]
            if not member_s or norm_mod is None or norm_mod.weight is None:
                continue
            s_shared = _geomean(member_s).to(torch.float16).to(norm_mod.weight.device)
            with torch.no_grad():
                norm_mod.weight.data = (norm_mod.weight.data.to(torch.float32) / s_shared).to(torch.float16)

            for proj in projs:
                if proj not in present_projs:
                    continue
                linear_name = f"model.layers.{i}.{proj}"
                mod = module_by_name.get(linear_name)
                if mod is None or not isinstance(mod, torch.nn.Linear):
                    continue
                w = mod.weight.data.to(torch.float32).cpu()
                signed, gscales = _sym_quantize(w, s_shared.cpu(), group_size)
                packed = _export_linear_from_signed(signed, gscales, group_size)
                _inject_linear(export_sd, linear_name, packed, mod)
                quantized_linear_names.add(linear_name)
                proj_count += 1
                if verbose:
                    print(f"  [AWQ ] {linear_name:<45} s_shared[{s_shared.min():.2f},{s_shared.max():.2f}]")

        # --- RTN linears (no preceding norm) ---
        for proj in RTN_LINEARS:
            if proj not in present_projs or proj not in layer_s:
                continue
            linear_name = f"model.layers.{i}.{proj}"
            mod = module_by_name.get(linear_name)
            if mod is None or not isinstance(mod, torch.nn.Linear):
                continue
            w = mod.weight.data.to(torch.float32).cpu()
            signed, gscales = _sym_quantize(w, torch.ones(w.shape[1]), group_size)
            packed = _export_linear_from_signed(signed, gscales, group_size)
            _inject_linear(export_sd, linear_name, packed, mod)
            quantized_linear_names.add(linear_name)
            proj_count += 1
            if verbose:
                print(f"  [RTN ] {linear_name:<45}")

    if verbose:
        print(f"\n  Exported {proj_count} quantized linears across {n_layers} layers.")

    # Copy every NON-quantized parameter (embeddings, folded norms, lm_head, ...)
    # as FP16. Quantized linears were already added as qweight/qzeros/scales.
    for name, mod in model.named_modules():
        if name in quantized_linear_names:
            continue
        for pname, param in mod.named_parameters(recurse=False):
            full = f"{name}.{pname}" if name else pname
            export_sd[full] = param.detach().cpu().clone()
    cfg_dict = model.config.to_dict()
    del model

    # 2. Write sharded safetensors + index.
    shard_bytes = 4 * 1024 ** 3  # 4 GB shards
    _save_sharded(export_sd, output_dir, shard_bytes, verbose)

    # 3. quantize_config.json (AutoAWQ schema) + patched config.json.
    qcfg = {
        "bits": 4,
        "group_size": group_size,
        "zero_point": True,
        "sym": False,
        "w_sym": False,
        "quant_method": "awq",
        "version": "gemm",
        "modules_to_not_convert": [],
    }
    with open(os.path.join(output_dir, "quantize_config.json"), "w") as f:
        json.dump(qcfg, f, indent=2)

    cfg = cfg_dict
    cfg["quantization_config"] = qcfg
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # 4. Copy tokenizer + generation config.
    for fn in os.listdir(model_path):
        if fn.startswith("tokenizer") or fn in ("generation_config.json", "vocab.json",
                                                "merges.txt", "special_tokens_map.json"):
            src = os.path.join(model_path, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(output_dir, fn))

    if verbose:
        print(f"\n✅ Export complete → {output_dir}")
    return output_dir


def _inject_linear(sd: dict[str, torch.Tensor], linear_name: str,
                   packed: dict[str, torch.Tensor], mod: torch.nn.Linear) -> None:
    """Replace a linear's ``.weight`` (and bias) with AutoAWQ packed tensors."""
    sd.pop(f"{linear_name}.weight", None)
    sd[f"{linear_name}.qweight"] = packed["qweight"].contiguous()
    sd[f"{linear_name}.qzeros"] = packed["qzeros"].contiguous()
    sd[f"{linear_name}.scales"] = packed["scales"].contiguous()
    if mod.bias is not None:
        sd[f"{linear_name}.bias"] = mod.bias.data.cpu().clone()


def _save_sharded(sd: dict[str, torch.Tensor], out_dir: str,
                  shard_bytes: int, verbose: bool) -> None:
    """Save a state dict to sharded safetensors + a model.safetensors.index.json."""
    import safetensors.torch as st

    # estimate per-tensor byte size
    sizes = {k: (v.numel() * v.element_size()) for k, v in sd.items()}
    total = sum(sizes.values())

    shards: list[dict[str, torch.Tensor]] = []
    cur: dict[str, torch.Tensor] = {}
    cur_bytes = 0
    weight_map: dict[str, str] = {}
    for k, v in sd.items():
        sz = sizes[k]
        if cur and cur_bytes + sz > shard_bytes:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[k] = v
        cur_bytes += sz
    if cur:
        shards.append(cur)

    if len(shards) == 1:
        st.save_file(shards[0], os.path.join(out_dir, "model.safetensors"))
        weight_map = {k: "model.safetensors" for k in shards[0]}
    else:
        for idx, shard in enumerate(shards, 1):
            fn = f"model-{idx:05d}-of-{len(shards):05d}.safetensors"
            st.save_file(shard, os.path.join(out_dir, fn))
            for k in shard:
                weight_map[k] = fn

    index = {
        "metadata": {"total_size": total},
        "weight_map": weight_map,
    }
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    if verbose:
        print(f"  Saved {len(shards)} shard(s), {total / 1e9:.2f} GB total")


def load_quantized_state(path: str) -> dict[str, dict]:
    """Load a ``quantized_state.pt`` produced by ``awq quantize``."""
    return torch.load(path, map_location="cpu", weights_only=True)