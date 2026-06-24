"""AWQ scale computation.

Computes per-channel scaling factors for each linear layer
using the AWQ formula: s = (channel_importance^alpha) / (mean^alpha).

Key insight: salient channels (high activation magnitude) get more
precision by scaling them up before quantization, then compensating
by scaling activations down — net output unchanged.
"""

import os
from typing import Any

import torch


def compute_awq_scale(
    weight: torch.Tensor,
    channel_importance: torch.Tensor,
    alpha: float = 0.5,
    clamp_min: float = 0.1,
    clamp_max: float = 10.0,
) -> torch.Tensor:
    """Compute per-channel AWQ scaling factor for one linear layer.

    Args:
        weight: Weight matrix W, shape [d_out, d_in].
        channel_importance: Activation magnitude per channel, shape [d_in].
            From calibration pass: |X|.mean(dim=0) across all samples.
        alpha: AWQ scaling strength (default 0.5 per paper).
            0.0 = no scaling (identity), 1.0 = full scaling.
        clamp_min: Minimum scale value for numerical stability.
        clamp_max: Maximum scale value for numerical stability.

    Returns:
        Scale factors s of shape [d_in], where s[c] > 1.0 means the
        c-th input channel is scaled UP before quantization.
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
    channel_importance = channel_importance.float()

    # AWQ scale formula: s = (channel_importance^alpha) / (channel_importance.mean()^alpha)
    # Channels with higher activation magnitudes get scaled UP (s > 1)
    # to preserve their precision during quantization.
    if channel_importance.mean() > 0:
        s = (channel_importance ** alpha) / (channel_importance.mean() ** alpha)
    else:
        s = torch.ones_like(channel_importance)

    # Clamp for numerical stability
    s = s.clamp(clamp_min, clamp_max)

    return s


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
            "last_only" — only quantize the last 12 layers.
            "first_only" — only quantize the first 12 layers.
        skip_lm_head: Skip lm_head projection.
        skip_tiny_projections: Skip attention projections with d_out < 64.

    Returns:
        Set of layer name strings to skip.
    """
    skips: set[str] = set()

    if skip_lm_head:
        skips.add("lm_head")
        skips.add("model.lm_head")
        skips.add("model.model.lm_head")

    if skip_tiny_projections:
        for i in range(24):
            for name in ("in_proj_a", "in_proj_b", "in_proj_z"):
                skips.add(f"model.layers.{i}.linear_attn.{name}")

    if quantize_strategy == "alternating":
        even_layers = set(range(0, 24, 2))
        for key in calibration_stats:
            for i in even_layers:
                if f"model.layers.{i}." in key:
                    skips.add(key)
    elif quantize_strategy == "last_only":
        first_12 = set(range(0, 12))
        for key in calibration_stats:
            for i in first_12:
                if f"model.layers.{i}." in key:
                    skips.add(key)
    elif quantize_strategy == "first_only":
        last_12 = set(range(12, 24))
        for key in calibration_stats:
            for i in last_12:
                if f"model.layers.{i}." in key:
                    skips.add(key)

    return skips


def compute_all_scales(
    model: torch.nn.Module,
    calibration_stats: dict[str, torch.Tensor],
    alpha: float = 0.5,
    output_path: str | None = None,
    verbose: bool = True,
    skip_set: set[str] | None = None,
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
            s = compute_awq_scale(weight, channel_importance, alpha=alpha)
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
            if verbose:
                print(f"  [ERROR] {layer_name}: {e}")
            continue

    # Summary
    if verbose:
        print(f"\n  ── Scale Stats ──")
        print(f"  Layers computed: {stats['total_layers']}")
        print(f"  Layers skipped:  {stats['skipped_layers']}")
        if stats["total_channels"] > 0:
            pct = 100.0 * stats["salient_channels_total"] / stats["total_channels"]
            print(f"  Salient channels: {stats['salient_channels_total']:,} / "
                  f"{stats['total_channels']:,} ({pct:.1f}%)")
        if scales:
            s_all = torch.cat([s.flatten() for s in scales.values()])
            print(f"  Scale range: [{s_all.min():.3f}, {s_all.max():.3f}]")
            print(f"  Scale mean: {s_all.mean():.3f}")

    # Save
    if output_path is not None and scales:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(scales, output_path)
        if verbose:
            print(f"\nSaved AWQ scales → {output_path}")
            print(f"  Layers: {len(scales)}")

    return scales
