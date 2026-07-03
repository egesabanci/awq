"""AWQ scale computation.

Computes per-channel scaling factors for each linear layer
using the AWQ formula: s = (channel_importance^alpha) / (mean^alpha).

Key insight: salient channels (high activation magnitude) get more
precision by scaling them up before quantization, then compensating
by scaling activations down — net output unchanged.
"""

import os
import re
from typing import Any

import torch
import torch.nn.functional as F

from utils.errors import ScaleError


def compute_awq_scale(
    weight: torch.Tensor,
    channel_importance: torch.Tensor,
    alpha: float = 0.5,
    clamp_min: float = 0.1,
    clamp_max: float = 10.0,
    group_size: int = 32,
    grid_search: bool = True,
    n_grid: int = 20,
) -> torch.Tensor:
    """Compute per-channel AWQ scaling factor for one linear layer.

    AWQ (Lin et al. 2023) protects salient weight channels — those aligned with
    large-magnitude activations — by scaling them UP before quantization and
    dividing activations by the same factor at inference. In this pipeline the
    activation scaling is folded into the stored weight, so the dequantized
    weight is ``Q(W * s) / s`` ≈ ``W``.

    Scale candidates are parameterised by a single exponent α applied to the
    activation magnitude (the per-channel importance), matching the candidate
    form used by the reference implementation (``mit-han-lab/llm-awq``
    ``_search_module_scale``: ``s = x_max^α``):

        s = (channel_importance ** α) / (channel_importance.mean() ** α)

    The ``weight`` matrix enters through a grid search over α that minimises the
    activation-weighted reconstruction error of ``Q(W·s)/s`` — i.e. the weight is
    actually quantized with each candidate scale and the result is scored, as in
    the reference search. When ``grid_search=False`` a fixed ``alpha`` is used.

    Args:
        weight: Weight matrix W, shape [d_out, d_in].
        channel_importance: Activation magnitude per channel, shape [d_in].
            From calibration pass: |X|.mean(dim=0) across all samples.
        alpha: Fixed exponent, used only when ``grid_search=False``.
        clamp_min: Minimum scale value for numerical stability.
        clamp_max: Maximum scale value for numerical stability.
        group_size: INT4 group size. The grid search quantizes with the same
            group size the quantizer will use, so the search reflects the real
            quantization grid.
        grid_search: If True (default), search α over ``[0, 1)`` and pick the
            best; if False, use the fixed ``alpha``.
        n_grid: Number of α candidates (default 20, per AWQ paper).

    Returns:
        Scale factors s of shape [d_in], where s[c] > 1.0 means the c-th input
        channel is scaled UP before quantization (more INT4 precision there).
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight, got shape {weight.shape}")
    if channel_importance.dim() != 1:
        raise ValueError(f"Expected 1D channel_importance, got shape {channel_importance.shape}")
    if weight.size(1) != channel_importance.size(0):
        raise ValueError(
            f"Channel mismatch: weight d_in={weight.size(1)} vs "
            f"channel_importance={channel_importance.size(0)}"
        )

    weight = weight.float()
    s_x = channel_importance.float().to(weight.device)
    s_x_mean = s_x.mean()

    def _candidate(a: float) -> torch.Tensor:
        # Activation-based candidate, mean-normalised (absolute scale is
        # invariant under the group quantizer, so normalisation is harmless).
        if s_x_mean > 0:
            s = (s_x ** a) / (s_x_mean ** a)
        else:
            s = torch.ones_like(s_x)
        return s.clamp(clamp_min, clamp_max)

    if not grid_search:
        return _candidate(alpha).cpu()

    # Grid search over α, minimising the activation-weighted reconstruction
    # error  ||(W - Q(W·s)/s) · diag(s_x)||²  which approximates the output
    # error ||(W - W_hat) · x||² (since |x| ~ s_x). α=0 is plain RTN (s=1), so
    # the search can only do as well as RTN, never worse.
    best_s = _candidate(0.0)
    best_err = _recon_error(weight, best_s, s_x, group_size)
    for i in range(1, n_grid):
        a = i / n_grid
        s = _candidate(a)
        err = _recon_error(weight, s, s_x, group_size)
        if err < best_err:
            best_err = err
            best_s = s
    return best_s.cpu()


def _pseudo_quantize_int4(w: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Group-wise INT4 pseudo-quantization round-trip (FP → FP).

    Mirrors ``quantize_layer_cpu``'s quantization grid exactly
    (``qscale = max|group| / 7``, round, clamp to [-7, 7]) without packing, so
    the AWQ grid search scores the *actual* quantizer. Used by scale search.
    """
    d_out, d_in = w.shape
    pad = (-d_in) % group_size
    if pad:
        w = F.pad(w, (0, pad))
    n_groups = w.shape[1] // group_size
    wg = w.view(d_out, n_groups, group_size)
    gmax = wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-10)
    qscale = gmax / 7.0
    q = (wg / qscale).round().clamp(-7, 7)
    deq = q * qscale
    return deq.view(d_out, -1)[:, :d_in].contiguous()


def _recon_error(
    weight: torch.Tensor,
    s: torch.Tensor,
    s_x: torch.Tensor,
    group_size: int,
) -> float:
    """Activation-weighted reconstruction error for a candidate AWQ scale.

    Uses the correct AWQ direction: weight scaled UP by s before quantization,
    divided by s after. Returns ``||((W - Q(W·s)/s) · diag(s_x))||²``.
    """
    s_col = s.unsqueeze(0)
    w_hat = _pseudo_quantize_int4(weight * s_col, group_size) / s_col
    return float((((weight - w_hat) ** 2) * s_x.unsqueeze(0)).sum().item())


def build_skip_set(calibration_stats: dict[str, torch.Tensor],
                   quantize_strategy: str = "alternating",
                   skip_lm_head: bool = True,
                   skip_tiny_projections: bool = True) -> set[str]:
    """Build a set of layer names to skip (keep in FP16).

    Args:
        calibration_stats: Dict of {layer_name: channel_importance}.
        quantize_strategy:
            "all" — quantize everything except explicit skips.
            "alternating" — keep even-numbered layers in FP16.
            "last_only" — only quantize the last half of the layers.
            "first_only" — only quantize the first half of the layers.
        skip_lm_head: Skip lm_head projection.
        skip_tiny_projections: Skip Qwen3.5-style hybrid-attention tiny
            projections (``linear_attn.in_proj_{a,b,z}``) if present.

    Returns:
        Set of layer name strings to skip.
    """
    skips: set[str] = set()

    if skip_lm_head:
        skips.add("lm_head")
        skips.add("model.lm_head")
        skips.add("model.model.lm_head")

    # Derive the actual transformer layer indices from the calibration stats
    # instead of hardcoding a fixed layer count (the old assumption only fit a
    # single 24-layer model).
    layer_indices: set[int] = set()
    for key in calibration_stats:
        m = re.search(r"model\.layers\.(\d+)\.", key)
        if m:
            layer_indices.add(int(m.group(1)))
    n_layers = (max(layer_indices) + 1) if layer_indices else 0

    if skip_tiny_projections:
        # Skip Qwen3.5 hybrid-attention tiny projections by name. These names
        # only exist in Qwen3.5-style models; on standard attention models they
        # simply match nothing, so this is harmless. Layer count is derived
        # from stats (falls back to 24 when stats are empty).
        upper = max(24, n_layers)
        for i in range(upper):
            for proj in ("in_proj_a", "in_proj_b", "in_proj_z"):
                skips.add(f"model.layers.{i}.linear_attn.{proj}")

    if quantize_strategy == "alternating":
        for key in calibration_stats:
            m = re.search(r"model\.layers\.(\d+)\.", key)
            if m and int(m.group(1)) % 2 == 0:
                skips.add(key)
    elif quantize_strategy == "last_only":
        half = n_layers // 2
        for key in calibration_stats:
            m = re.search(r"model\.layers\.(\d+)\.", key)
            if m and int(m.group(1)) < half:
                skips.add(key)
    elif quantize_strategy == "first_only":
        half = n_layers // 2
        for key in calibration_stats:
            m = re.search(r"model\.layers\.(\d+)\.", key)
            if m and int(m.group(1)) >= half:
                skips.add(key)

    return skips


def compute_all_scales(
    model: torch.nn.Module,
    calibration_stats: dict[str, torch.Tensor],
    alpha: float = 0.5,
    output_path: str | None = None,
    verbose: bool = True,
    skip_set: set[str] | None = None,
    group_size: int = 32,
    grid_search: bool = True,
    n_grid: int = 20,
) -> dict[str, torch.Tensor]:
    """Compute AWQ scale factors for ALL linear layers in the model.

    Args:
        model: FP16 model (used to extract weight matrices).
        calibration_stats: {layer_name: channel_importance} from calibration.
        alpha: AWQ scaling strength.
        output_path: Where to save scales .pt file. If None, not saved to disk.
        verbose: Print per-layer diagnostics.
        skip_set: Set of layer names to skip (keep in FP16).
            If None, uses build_skip_set() defaults.
        group_size: INT4 group size; passed to compute_awq_scale so the grid
            search matches the quantizer's grid.
        grid_search: If True, search α per layer (proper AWQ); if False, use
            the fixed ``alpha``.
        n_grid: Number of α candidates when ``grid_search=True`` (default 20,
            per AWQ paper).

    Returns:
        dict of {layer_name: scale_factors} — [d_in] per layer.
    """
    if skip_set is None:
        skip_set = build_skip_set(calibration_stats)

    # Build name → weight mapping
    named_weights: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            named_weights[name] = module.weight.data

    scales: dict[str, torch.Tensor] = {}
    failed: list[tuple[str, str]] = []  # (layer_name, error msg) — surfaced even when quiet
    stats = {
        "total_layers": 0,
        "skipped_layers": 0,
        "salient_channels_total": 0,
        "total_channels": 0,
    }

    for layer_name in sorted(calibration_stats.keys()):
        if layer_name in skip_set:
            stats["skipped_layers"] += 1
            if verbose:
                reason = "excluded from INT4"
                if "lm_head" in layer_name:
                    reason = "lm_head too sensitive"
                elif "in_proj_a" in layer_name or "in_proj_b" in layer_name or "in_proj_z" in layer_name:
                    reason = "tiny projection"
                else:
                    reason = "quantize_strategy skip"
                print(f"  [SKIP] {layer_name} — {reason}")
            continue

        if layer_name not in named_weights:
            if verbose:
                print(f"  [SKIP] {layer_name} — no weight found")
            continue

        weight = named_weights[layer_name]
        channel_importance = calibration_stats[layer_name]

        try:
            s = compute_awq_scale(
                weight, channel_importance,
                alpha=alpha,
                group_size=group_size,
                grid_search=grid_search,
                n_grid=n_grid,
            )
            scales[layer_name] = s

            stats["total_layers"] += 1

            n_salient = (s > 1.05).sum().item()
            stats["salient_channels_total"] += n_salient
            stats["total_channels"] += s.size(0)

            if verbose:
                d_in = s.size(0)
                s_min = s.min().item()
                s_max = s.max().item()
                s_mean = s.mean().item()
                pct_salient = 100.0 * n_salient / d_in
                print(f"  {layer_name:<55} d_in={d_in:<5} "
                      f"s=[{s_min:.3f}, {s_max:.3f}] mean={s_mean:.3f} "
                      f"salient={pct_salient:.1f}%")

        except Exception as e:
            # Always record failures (not just when verbose) so a silent
            # "0 layers" result can be diagnosed instead of masquerading as success.
            failed.append((layer_name, str(e)))
            if verbose:
                print(f"  [ERROR] {layer_name}: {e}")
            continue

    # Summary
    if verbose:
        print(f"\n  ── Scale Stats ──")
        print(f"  Layers computed: {stats['total_layers']}")
        print(f"  Layers skipped:  {stats['skipped_layers']}")
        if failed:
            print(f"  Layers failed:   {len(failed)}")
        if stats["total_channels"] > 0:
            pct = 100.0 * stats["salient_channels_total"] / stats["total_channels"]
            print(f"  Salient channels: {stats['salient_channels_total']:,} / "
                  f"{stats['total_channels']:,} ({pct:.1f}%)")
        if scales:
            s_all = torch.cat([s.flatten() for s in scales.values()])
            print(f"  Scale range: [{s_all.min():.3f}, {s_all.max():.3f}]")
            print(f"  Scale mean: {s_all.mean():.3f}")

    # Surface failures loudly even in quiet mode, and refuse to write an empty
    # scales file (downstream steps would silently produce nothing).
    if failed:
        preview = "; ".join(f"{n}: {msg}" for n, msg in failed[:3])
        print(f"[WARN] {len(failed)} layer(s) failed during scale computation. "
              f"First: {preview}")
    if len(scales) == 0:
        raise ScaleError(
            "compute_all_scales produced 0 scales (all layers were skipped, "
            "missing from the model, or failed). Check that the model path and "
            "calibration stats match the same architecture."
            + (f" {len(failed)} layer(s) raised errors." if failed else "")
        )

    # Save
    if output_path is not None and scales:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(scales, output_path)
        if verbose:
            print(f"\nSaved AWQ scales → {output_path}")
            print(f"  Layers: {len(scales)}")

    return scales
