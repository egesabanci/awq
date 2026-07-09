"""AWQ CLI — command-line interface for the quantization pipeline.

Usage:
    awq --help
    awq calibrate --model /path/to/model --dataset wikitext ...
    awq run --model /path/to/model --dataset wikitext --output-dir ./results
"""

import argparse
import os
import sys
from typing import Any

# Lazy imports — heavy libs (torch, transformers) are only imported
# when a subcommand actually needs them.


def _warn_if_cuda_mismatch(requested_device: str | None) -> str | None:
    """Check if the requested device is available; warn if not.

    Returns the resolved device string (may fall back to auto-detect).
    """
    if requested_device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[WARN] --device cuda requested but CUDA is not available.")
                print("       Install PyTorch with CUDA support:")
                print("       pip install torch --index-url https://download.pytorch.org/whl/cu124")
                print("       Falling back to cpu.")
                return "cpu"
        except ImportError:
            print("[WARN] torch not importable. Falling back to cpu.")
            return "cpu"
    return requested_device


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="awq",
        description="Model-agnostic AWQ (Activation-aware Weight Quantization) CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  awq calibrate --model Qwen/Qwen3-0.6B --dataset wikitext --output results/stats.pt
  awq scales     --model Qwen/Qwen3-0.6B --calibration-stats results/stats.pt
  awq quantize   --model Qwen/Qwen3-0.6B --scales results/awq_scales.pt
  awq export     --model Qwen/Qwen3-0.6B --from results/awq_quantized/quantized_state.pt --to results/awq_hf
  awq run        --model Qwen/Qwen3-0.6B --dataset wikitext --output-dir results/
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── calibrate ────────────────────────────────────────────────────
    calib = subparsers.add_parser("calibrate", help="Run calibration pass")
    calib.add_argument("--model", required=True, help="Model path or HF repo ID")
    calib.add_argument("--dataset", default="wikitext",
                       help="Calibration dataset (wikitext; bundled WikiText-2 samples)")
    calib.add_argument("--output", default="results/calibration_stats.pt",
                       help="Output path for calibration stats")
    calib.add_argument("--samples", type=int, default=128,
                       help="Number of calibration samples (default: 128)")
    calib.add_argument("--batch-size", type=int, default=5,
                       help="Batch size (default: 5)")
    calib.add_argument("--max-length", type=int, default=2048,
                       help="Max sequence length (default: 2048)")
    calib.add_argument("--device", default=None,
                       help="Device (auto-detected if omitted)")
    calib.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")

    # ── scales ───────────────────────────────────────────────────────
    scales = subparsers.add_parser("scales", help="Compute AWQ scales")
    scales.add_argument("--model", required=True, help="Model path or HF repo ID")
    scales.add_argument("--calibration-stats", required=True,
                        help="Path to calibration_stats.pt")
    scales.add_argument("--output", default="results/awq_scales.pt",
                        help="Output path for scale factors")
    scales.add_argument("--alpha", type=float, default=0.5,
                       help="AWQ scaling strength (used only with --no-grid-search)")
    scales.add_argument("--group-size", type=int, default=32,
                       help="INT4 group size; must match `awq quantize` (default: 32)")
    scales.add_argument("--no-grid-search", action="store_true",
                       help="Disable per-layer alpha grid search; use --alpha instead")
    scales.add_argument("--n-grid", type=int, default=20,
                       help="Number of alpha candidates for grid search (default: 20, per AWQ paper)")
    scales.add_argument("--quantize-strategy", default="alternating",
                        choices=["all", "alternating", "last_only", "first_only"],
                        help="Which layers to quantize (default: alternating)")
    scales.add_argument("--no-skip-lm-head", action="store_true",
                        help="Include lm_head in quantization")
    scales.add_argument("--device", default=None,
                        help="Device (auto-detected if omitted)")
    scales.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    # ── quantize ─────────────────────────────────────────────────────
    quant = subparsers.add_parser("quantize", help="Quantize model weights")
    quant.add_argument("--model", required=True, help="Model path or HF repo ID")
    quant.add_argument("--scales", required=True, help="Path to awq_scales.pt")
    quant.add_argument("--output-dir", default="results/awq_quantized",
                       help="Output directory for quantized model")
    quant.add_argument("--group-size", type=int, default=32,
                       help="INT4 group size (default: 32)")
    quant.add_argument("--verify-layers", type=int, default=3,
                       help="Number of layers to verify (default: 3, 0 to skip)")
    quant.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")

    # ── export ───────────────────────────────────────────────────────
    exp = subparsers.add_parser("export", help="Export quantized_state.pt to a runtime-loadable HF-AWQ INT4 model")
    exp.add_argument("--model", required=True, help="FP16 model path or HF repo ID (for config + weights to re-quantize)")
    exp.add_argument("--from", dest="from_path", required=True,
                     help="Path to quantized_state.pt (from `awq quantize`)")
    exp.add_argument("--to", dest="to_dir", default="results/awq_hf",
                     help="Output directory for the HF-AWQ model")
    exp.add_argument("--group-size", type=int, default=32,
                     help="INT4 group size (must match the quantized artifact; default 32)")
    exp.add_argument("--device", default="cpu",
                     help="Device to load the FP16 model on for re-quantization (cpu or cuda; use cuda for large models)")
    exp.add_argument("--quiet", action="store_true", help="Suppress progress output")

    # ── run ──────────────────────────────────────────────────────────
    run = subparsers.add_parser("run", help="Run full pipeline end-to-end")
    run.add_argument("--model", required=True, help="Model path or HF repo ID")
    run.add_argument("--dataset", default="wikitext",
                     help="Calibration dataset (wikitext; bundled WikiText-2 samples)")
    run.add_argument("--output-dir", default="results",
                     help="Output directory for all artifacts")
    run.add_argument("--samples", type=int, default=128,
                     help="Number of calibration samples")
    run.add_argument("--batch-size", type=int, default=5,
                     help="Calibration batch size")
    run.add_argument("--max-length", type=int, default=2048,
                     help="Max sequence length")
    run.add_argument("--group-size", type=int, default=32,
                     help="INT4 group size")
    run.add_argument("--alpha", type=float, default=0.5,
                     help="AWQ scaling strength (used only with --no-grid-search)")
    run.add_argument("--no-grid-search", action="store_true",
                     help="Disable per-layer alpha grid search; use --alpha instead")
    run.add_argument("--n-grid", type=int, default=20,
                     help="Number of alpha candidates for grid search (default: 20)")
    run.add_argument("--quantize-strategy", default="alternating",
                     choices=["all", "alternating", "last_only", "first_only"])
    run.add_argument("--no-skip-lm-head", action="store_true",
                     help="Include lm_head in quantization (default: skip)")
    run.add_argument("--verify-layers", type=int, default=3,
                     help="Layers to dequantize and verify post-quantization (default: 3, 0 to skip)")
    run.add_argument("--device", default=None,
                     help="Device (auto-detected if omitted)")
    run.add_argument("--quiet", action="store_true",
                     help="Suppress progress output")

    return parser


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Execute the calibrate subcommand."""
    verbose = not args.quiet
    device = args.device

    # Load calibration dataset
    print(f"Loading calibration data from {args.dataset}...")
    prompts = _load_calibration_prompts(args.dataset, args.samples)

    # Load model
    from awq.models import load_model
    from utils.memory import get_device

    if device is None:
        device = get_device()
    model, tokenizer = load_model(args.model, device)
    from utils.memory import log_memory
    log_memory("after_model_load", device)

    # Run calibration
    from awq.calibrate import run_calibration

    stats = run_calibration(
        model, tokenizer, prompts,
        output_path=args.output,
        max_length=args.max_length,
        device=device,
        verbose=verbose,
        batch_size=args.batch_size,
    )

    print(f"\n✅ Calibration complete. {len(stats)} layers captured → {args.output}")
    return 0


def cmd_scales(args: argparse.Namespace) -> int:
    """Execute the scales subcommand."""
    verbose = not args.quiet

    # Validate input
    from utils.errors import require_calibration_stats

    calibration_stats = require_calibration_stats(args.calibration_stats)

    # Load model (for weight d_in dimensions)
    from awq.models import load_model
    from utils.memory import get_device

    device = args.device or get_device()
    model, _ = load_model(args.model, device)

    # Compute scales
    from awq.scales import compute_all_scales, build_skip_set

    skip_set = build_skip_set(
        calibration_stats,
        quantize_strategy=args.quantize_strategy,
        skip_lm_head=not args.no_skip_lm_head,
        skip_tiny_projections=True,
    )

    scales = compute_all_scales(
        model, calibration_stats,
        alpha=args.alpha,
        output_path=args.output,
        verbose=verbose,
        skip_set=skip_set,
        group_size=args.group_size,
        grid_search=not args.no_grid_search,
        n_grid=args.n_grid,
    )

    print(f"\n✅ Scale computation complete. {len(scales)} layers → {args.output}")
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    """Execute the quantize subcommand."""
    verbose = not args.quiet

    # Validate inputs
    from utils.errors import require_scales

    scales = require_scales(args.scales)

    # Quantize
    from awq.quantize import quantize_all_layers, verify_reconstruction

    quantized = quantize_all_layers(
        args.model, scales,
        group_size=args.group_size,
        output_dir=args.output_dir,
        verbose=verbose,
    )

    # Verify
    if args.verify_layers > 0:
        print(f"\nVerifying reconstruction ({args.verify_layers} layers)...")
        errors = verify_reconstruction(quantized, args.model, num_layers=args.verify_layers)
        if errors:
            avg_mse = sum(errors.values()) / len(errors)
            print(f"  Average MSE: {avg_mse:.8f}")

    print(f"\n✅ Quantization complete. {len(quantized)} layers → {args.output_dir}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Execute the export subcommand: quantized_state.pt -> HF-AWQ INT4 dir."""
    verbose = not args.quiet
    from awq.export import export_to_awq, load_quantized_state

    # Validate the quantized state path (must point at quantized_state.pt)
    if not os.path.exists(args.from_path):
        print(f"[ERROR] quantized state not found: {args.from_path}", file=sys.stderr)
        return 1
    quantized_state = load_quantized_state(args.from_path)
    if not isinstance(quantized_state, dict) or len(quantized_state) == 0:
        print("[ERROR] quantized state is empty or malformed", file=sys.stderr)
        return 1

    export_to_awq(quantized_state, args.model, args.to_dir,
                  group_size=args.group_size, verbose=verbose, device=args.device)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the full pipeline.

    Steps:
        1. Prepare calibration data
        2. Run calibration pass
        3. Compute AWQ scales
        4. Quantize model (with reconstruction verification)
    """
    verbose = not args.quiet
    device = args.device

    # Set up paths
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    calib_stats_path = os.path.join(output_dir, "calibration_stats.pt")
    scales_path = os.path.join(output_dir, "awq_scales.pt")
    quantized_dir = os.path.join(output_dir, "awq_quantized")

    print("=" * 60)
    print("AWQ FULL PIPELINE")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Output:     {output_dir}")
    print(f"  Samples:    {args.samples}")
    print(f"  Device:     {device or 'auto'}")
    print()

    # ── Step 1: Prepare calibration data ─────────────────────────────
    print("─" * 60)
    print("STEP 1/4 — Prepare calibration data")
    print("─" * 60)
    prompts = _load_calibration_prompts(args.dataset, args.samples)
    print(f"  Loaded {len(prompts)} calibration samples\n")

    # ── Step 2: Calibrate ────────────────────────────────────────────
    print("─" * 60)
    print("STEP 2/4 — Calibration pass")
    print("─" * 60)

    from awq.models import load_model
    from utils.memory import get_device, log_memory

    if device is None:
        device = get_device()
    model, tokenizer = load_model(args.model, device)
    log_memory("after_model_load", device)

    from awq.calibrate import run_calibration

    stats = run_calibration(
        model, tokenizer, prompts,
        output_path=calib_stats_path,
        max_length=args.max_length,
        device=device,
        verbose=verbose,
        batch_size=args.batch_size,
    )
    del model, tokenizer, prompts
    from utils.memory import empty_cache
    empty_cache()

    print(f"\n  ✓ Calibration stats → {calib_stats_path}\n")

    # ── Step 3: Compute scales ──────────────────────────────────────
    print("─" * 60)
    print("STEP 3/4 — Compute AWQ scales")
    print("─" * 60)

    from utils.errors import require_calibration_stats

    calib = require_calibration_stats(calib_stats_path)

    # Re-load model for weight dimensions
    model, _ = load_model(args.model, device)

    from awq.scales import compute_all_scales, build_skip_set

    skip_set = build_skip_set(
        calib,
        quantize_strategy=args.quantize_strategy,
        skip_lm_head=not getattr(args, "no_skip_lm_head", False),
    )

    scales = compute_all_scales(
        model, calib,
        alpha=args.alpha,
        output_path=scales_path,
        verbose=verbose,
        skip_set=skip_set,
        group_size=args.group_size,
        grid_search=not args.no_grid_search,
        n_grid=args.n_grid,
    )
    del model
    empty_cache()

    print(f"\n  ✓ AWQ scales → {scales_path}\n")

    # ── Step 4: Quantize ────────────────────────────────────────────
    print("─" * 60)
    print("STEP 4/4 — Quantize model weights")
    print("─" * 60)

    from utils.errors import require_scales
    from awq.quantize import quantize_all_layers, verify_reconstruction

    _ = require_scales(scales_path)

    quantized = quantize_all_layers(
        args.model, scales,
        group_size=args.group_size,
        output_dir=quantized_dir,
        verbose=verbose,
    )

    # Verify
    if args.verify_layers > 0:
        errors = verify_reconstruction(quantized, args.model, num_layers=args.verify_layers, verbose=verbose)
        if errors:
            avg_mse = sum(errors.values()) / len(errors)
            print(f"  Average MSE: {avg_mse:.8f}")
    else:
        print("  (verification skipped: --verify-layers 0)")

    print(f"\n  ✓ Quantized model → {quantized_dir}\n")

    print()
    print("=" * 60)
    print("✅ PIPELINE COMPLETE")
    print(f"   Artifacts in: {output_dir}")
    print("=" * 60)
    return 0


# ── Data helpers ─────────────────────────────────────────────────────


def _load_calibration_prompts(dataset: str, n_samples: int) -> list[str]:
    """Load and format calibration prompts from the specified dataset."""
    if dataset == "wikitext":
        import json
        # Load pre-saved WikiText-2 samples
        calib_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "natural_calibration.json"
        )
        if os.path.exists(calib_path):
            with open(calib_path) as f:
                texts = json.load(f)
            return texts[:n_samples]

        # Fall back to loading from HF
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        texts = [s["text"] for s in ds if s["text"].strip()]
        return texts[:n_samples]

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Only 'wikitext' is supported.")


# ── Entry point ─────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the AWQ CLI.

    Returns exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Warn if user asked for CUDA but it's not available
    if hasattr(args, 'device') and args.device:
        args.device = _warn_if_cuda_mismatch(args.device)

    try:
        if args.command == "calibrate":
            return cmd_calibrate(args)
        elif args.command == "scales":
            return cmd_scales(args)
        elif args.command == "quantize":
            return cmd_quantize(args)
        elif args.command == "export":
            return cmd_export(args)
        elif args.command == "run":
            return cmd_run(args)
        else:
            parser.print_help()
            return 0
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

        # Memory diagnostics on OOM
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            from utils.errors import diagnose_oom
            from utils.memory import get_device as _get_device
            diagnose_oom(args.device or _get_device())

        return 1


if __name__ == "__main__":
    sys.exit(main())
