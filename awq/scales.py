"""AWQ scale computation — Issue #7.

Computes per-channel scaling factors for each linear layer
using the AWQ formula: s = (mean(|W|)^alpha) / (global_mean^alpha).

Key insight: salient channels (high activation magnitude) get more
precision by scaling them up before quantization, then compensating
by scaling activations down — net output unchanged.
"""

import os
from typing import Any

import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def compute_awq_scale(
    weight: torch.Tensor,
    channel_importance: torch.Tensor,
    alpha: float = 0.5,
    salient_fraction: float = 0.01,
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
        salient_fraction: Fraction of channels considered salient (top-k%).
            Used only for diagnostics, not in the formula itself.
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
    # The channel_importance is |X|.mean(0) from the calibration pass,
    # representing activation magnitude per input channel.
    # Channels with higher activation magnitudes get scaled UP (s > 1)
    # to preserve their precision during quantization.
    # This is the core insight of the AWQ paper: activation magnitudes
    # reveal which weight channels are most important.
    if channel_importance.mean() > 0:
        s = (channel_importance ** alpha) / (channel_importance.mean() ** alpha)
    else:
        s = torch.ones_like(channel_importance)

    # Clamp for numerical stability
    s = s.clamp(clamp_min, clamp_max)

    return s


def compute_all_scales(
    model: torch.nn.Module,
    calibration_stats: dict[str, torch.Tensor],
    alpha: float = 0.5,
    salient_fraction: float = 0.01,
    output_path: str | None = None,
    verbose: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute AWQ scale factors for ALL linear layers in the model.

    Matches each calibration stat to its corresponding weight matrix
    by layer name, then computes per-channel scale.

    Args:
        model: FP16 model (used to extract weight matrices).
        calibration_stats: {layer_name: channel_importance} from calibration.
        alpha: AWQ scaling strength.
        salient_fraction: Fraction of salient channels (diagnostic only).
        output_path: Where to save scales .pt file.
        verbose: Print per-layer diagnostics.

    Returns:
        dict of {layer_name: scale_factors} — [d_in] per layer.
    """
    # Build name → weight mapping
    named_weights: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            named_weights[name] = module.weight.data

    # Skip these layers — too sensitive for INT4 quantization
    SKIP_LAYERS = {
        "lm_head", "model.lm_head", "model.model.lm_head",
        # Tiny projection layers (d_out < 64) — INT4 has no benefit
        "model.layers.0.linear_attn.in_proj_a", "model.layers.0.linear_attn.in_proj_b",
        "model.layers.1.linear_attn.in_proj_a", "model.layers.1.linear_attn.in_proj_b",
        "model.layers.2.linear_attn.in_proj_a", "model.layers.2.linear_attn.in_proj_b",
        "model.layers.4.linear_attn.in_proj_a", "model.layers.4.linear_attn.in_proj_b",
        "model.layers.5.linear_attn.in_proj_a", "model.layers.5.linear_attn.in_proj_b",
        "model.layers.6.linear_attn.in_proj_a", "model.layers.6.linear_attn.in_proj_b",
        "model.layers.8.linear_attn.in_proj_a", "model.layers.8.linear_attn.in_proj_b",
        "model.layers.9.linear_attn.in_proj_a", "model.layers.9.linear_attn.in_proj_b",
        "model.layers.10.linear_attn.in_proj_a", "model.layers.10.linear_attn.in_proj_b",
        "model.layers.12.linear_attn.in_proj_a", "model.layers.12.linear_attn.in_proj_b",
        "model.layers.13.linear_attn.in_proj_a", "model.layers.13.linear_attn.in_proj_b",
        "model.layers.14.linear_attn.in_proj_a", "model.layers.14.linear_attn.in_proj_b",
        "model.layers.16.linear_attn.in_proj_a", "model.layers.16.linear_attn.in_proj_b",
        "model.layers.17.linear_attn.in_proj_a", "model.layers.17.linear_attn.in_proj_b",
        "model.layers.18.linear_attn.in_proj_a", "model.layers.18.linear_attn.in_proj_b",
        "model.layers.20.linear_attn.in_proj_a", "model.layers.20.linear_attn.in_proj_b",
        "model.layers.21.linear_attn.in_proj_a", "model.layers.21.linear_attn.in_proj_b",
        "model.layers.22.linear_attn.in_proj_a", "model.layers.22.linear_attn.in_proj_b",
    }

    # Keep every OTHER layer in FP16 to prevent error compounding
    # Quantize only: layers 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23 (odd layers)
    QUANTIZE_LAYERS = set(range(1, 24, 2))  # odd-numbered layers
    TOTAL_LAYERS = 24

    for i in range(TOTAL_LAYERS):
        if i not in QUANTIZE_LAYERS:
            # Keep even-numbered layers in FP16
            for name in list(calibration_stats.keys()):
                if f"model.layers.{i}." in name:
                    SKIP_LAYERS.add(name)

    scales: dict[str, torch.Tensor] = {}
    stats = {
        "total_layers": 0,
        "salient_channels_total": 0,
        "total_channels": 0,
    }

    for layer_name in sorted(calibration_stats.keys()):
        if layer_name in SKIP_LAYERS:
            reason = "excluded from INT4"
            if "lm_head" in layer_name:
                reason = "lm_head too sensitive"
            elif "in_proj_a" in layer_name or "in_proj_b" in layer_name:
                reason = "tiny projection (16×2048)"
            elif any(f"model.layers.{i}." in layer_name for i in range(24) if i not in QUANTIZE_LAYERS):
                reason = f"kept in FP16 (even layer)"
            if verbose:
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
                weight,
                channel_importance,
                alpha=alpha,
                salient_fraction=salient_fraction,
            )
            scales[layer_name] = s

            stats["total_layers"] += 1

            # Diagnostic: count channels where s != 1 (salient)
            n_salient = (s > 1.05).sum().item()  # channels scaled >5%
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

    # Stats summary
    if verbose:
        print(f"\n  ── Scale Stats ──")
        print(f"  Layers computed: {stats['total_layers']}")
        if stats["total_channels"] > 0:
            pct = 100.0 * stats["salient_channels_total"] / stats["total_channels"]
            print(f"  Salient channels: {stats['salient_channels_total']:,} / "
                  f"{stats['total_channels']:,} ({pct:.1f}%)")
        s_all = torch.cat([s.flatten() for s in scales.values()])
        print(f"  Scale range: [{s_all.min():.3f}, {s_all.max():.3f}]")
        print(f"  Scale mean: {s_all.mean():.3f}")
        print(f"  Channels with s>1.05: {(s_all > 1.05).sum().item():,} "
              f"({100.0 * (s_all > 1.05).sum().item() / s_all.numel():.1f}%)")

    # Save
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "awq_scales.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(scales, output_path)
    if verbose:
        print(f"\nSaved AWQ scales → {output_path}")
        print(f"  Layers: {len(scales)}")

    return scales


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    # Load model
    from eval.runner import load_fp16_model

    print("Loading model for weight extraction...")
    model, _ = load_fp16_model("models/Qwen3.5-2B-FP16", device="mps")

    # Load calibration stats
    calib_path = os.path.join(RESULTS_DIR, "calibration_stats_v2.pt")
    calibration_stats = torch.load(calib_path)
    print(f"Loaded calibration stats: {len(calibration_stats)} layers\n")

    # Compute scales
    scales = compute_all_scales(
        model,
        calibration_stats,
        alpha=0.5,
        verbose=True,
    )

    print(f"\n✅ Scale computation complete. {len(scales)} layers.")
