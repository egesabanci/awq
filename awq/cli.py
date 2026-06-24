"""AWQ CLI — command-line interface for the quantization pipeline.

Usage:
    python -m awq --help
    python -m awq calibrate --model /path/to/model --dataset c4 ...
    python -m awq run --model /path/to/model --dataset c4 --output-dir ./results
"""

import argparse
import os
import sys
from typing import Any

# Lazy imports — heavy libs (torch, transformers) are only imported
# when a subcommand actually needs them.


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m awq",
        description="AWQ quantization experimentation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m awq calibrate --model Qwen/Qwen3.5-2B --dataset c4 --output-dir results
  python -m awq scales --model Qwen/Qwen3.5-2B --calibration-stats results/stats.pt
  python -m awq quantize --model Qwen/Qwen3.5-2B --scales results/scales.pt
  python -m awq benchmark --model Qwen/Qwen3.5-2B --awq-dir results/awq/
  python -m awq run --model Qwen/Qwen3.5-2B --dataset c4 --output-dir results/
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── calibrate ────────────────────────────────────────────────────
    calib = subparsers.add_parser("calibrate", help="Run calibration pass")
    calib.add_argument("--model", required=True, help="Model path or HF repo ID")
    calib.add_argument("--dataset", default="wikitext",
                       choices=["wikitext", "c4", "toolace"],
                       help="Calibration dataset")
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
                        help="AWQ scaling strength (default: 0.5)")
    scales.add_argument("--quantize-strategy", default="alternating",
                        choices=["all", "alternating", "last_only", "first_only"],
                        help="Which layers to quantize (default: alternating)")
    scales.add_argument("--no-skip-lm-head", action="store_true",
                        help="Include lm_head in quantization")
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

    # ── benchmark ───────────────────────────────────────────────────
    bench = subparsers.add_parser("benchmark",
                                   help="Run FP16 + AWQ comparison")
    bench.add_argument("--model", required=True, help="Model path or HF repo ID")
    bench.add_argument("--awq-dir", default="results/awq_quantized",
                       help="Directory with quantized model")
    bench.add_argument("--output", default="results/comparison.json",
                       help="Output path for comparison results")
    bench.add_argument("--dataset", default="toolace",
                       choices=["toolace", "wikitext"],
                       help="Evaluation dataset")
    bench.add_argument("--eval-samples", type=int, default=10,
                       help="Number of evaluation samples")
    bench.add_argument("--max-new-tokens", type=int, default=256,
                       help="Max new tokens per generation")
    bench.add_argument("--device", default=None,
                       help="Device (auto-detected if omitted)")
    bench.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")

    # ── run ──────────────────────────────────────────────────────────
    run = subparsers.add_parser("run", help="Run full pipeline end-to-end")
    run.add_argument("--model", required=True, help="Model path or HF repo ID")
    run.add_argument("--dataset", default="wikitext",
                     choices=["wikitext", "c4", "toolace"],
                     help="Calibration dataset")
    run.add_argument("--output-dir", default="results",
                     help="Output directory for all artifacts")
    run.add_argument("--samples", type=int, default=128,
                     help="Number of calibration samples")
    run.add_argument("--eval-samples", type=int, default=10,
                     help="Number of evaluation samples")
    run.add_argument("--batch-size", type=int, default=5,
                     help="Calibration batch size")
    run.add_argument("--max-length", type=int, default=2048,
                     help="Max sequence length")
    run.add_argument("--max-new-tokens", type=int, default=256,
                     help="Max new tokens per generation")
    run.add_argument("--group-size", type=int, default=32,
                     help="INT4 group size")
    run.add_argument("--alpha", type=float, default=0.5,
                     help="AWQ scaling strength")
    run.add_argument("--quantize-strategy", default="alternating",
                     choices=["all", "alternating", "last_only", "first_only"])
    run.add_argument("--device", default=None,
                     help="Device (auto-detected if omitted)")
    run.add_argument("--skip-benchmark", action="store_true",
                     help="Skip the benchmark step after quantization")
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
    from eval.runner import load_fp16_model
    from utils.memory import limit_memory, device as dev_util

    if device is None:
        device = dev_util.get_device()
    if device == "mps":
        limit_memory(0.7, device)

    model, tokenizer = load_fp16_model(args.model, device)
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
    from eval.runner import load_fp16_model
    from utils.memory import device as dev_util

    device = args.device or dev_util.get_device()
    model, _ = load_fp16_model(args.model, device)

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


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Execute the benchmark subcommand."""
    verbose = not args.quiet
    device = args.device

    # Load evaluation data
    print(f"Loading evaluation data ({args.eval_samples} samples from {args.dataset})...")
    prompts, expected = _load_eval_data(args.dataset, args.eval_samples)

    # Validate AWQ directory
    from utils.errors import require_quantized_dir

    awq_state_path = require_quantized_dir(args.awq_dir)

    # Run comparison
    from eval.benchmark import run_awq_comparison

    comparison = run_awq_comparison(
        fp16_model_path=args.model,
        awq_quantized_path=awq_state_path,
        prompts=prompts,
        expected=expected,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    print(f"\n✅ Benchmark complete → {args.output}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the full pipeline.

    Steps:
        1. Prepare calibration data
        2. Run calibration pass
        3. Compute AWQ scales
        4. Quantize model
        5. (Optional) Run benchmark comparison
    """
    verbose = not args.quiet
    device = args.device

    # Set up paths
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    calib_stats_path = os.path.join(output_dir, "calibration_stats.pt")
    scales_path = os.path.join(output_dir, "awq_scales.pt")
    quantized_dir = os.path.join(output_dir, "awq_quantized")
    comparison_path = os.path.join(output_dir, "comparison.json")

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

    from eval.runner import load_fp16_model
    from utils.memory import limit_memory, device as dev_util, log_memory

    if device is None:
        device = dev_util.get_device()
    if device == "mps":
        limit_memory(0.7, device)

    model, tokenizer = load_fp16_model(args.model, device)
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
    model, _ = load_fp16_model(args.model, device)

    from awq.scales import compute_all_scales, build_skip_set

    skip_set = build_skip_set(
        calib,
        quantize_strategy=args.quantize_strategy,
    )

    scales = compute_all_scales(
        model, calib,
        alpha=args.alpha,
        output_path=scales_path,
        verbose=verbose,
        skip_set=skip_set,
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
    errors = verify_reconstruction(quantized, args.model, num_layers=3, verbose=verbose)
    if errors:
        avg_mse = sum(errors.values()) / len(errors)
        print(f"  Average MSE: {avg_mse:.8f}")

    print(f"\n  ✓ Quantized model → {quantized_dir}\n")

    # ── Step 5: Benchmark (optional) ─────────────────────────────────
    if not args.skip_benchmark:
        print("─" * 60)
        print("STEP 5/5 — Benchmark comparison")
        print("─" * 60)

        eval_prompts, eval_expected = _load_eval_data(args.dataset, args.eval_samples)

        from utils.errors import require_quantized_dir

        awq_state_path = require_quantized_dir(quantized_dir)

        from eval.benchmark import run_awq_comparison

        comparison = run_awq_comparison(
            fp16_model_path=args.model,
            awq_quantized_path=awq_state_path,
            prompts=eval_prompts,
            expected=eval_expected,
            output_path=comparison_path,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        print(f"\n  ✓ Comparison results → {comparison_path}")
    else:
        print("  (benchmark skipped via --skip-benchmark)")

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

    elif dataset == "c4":
        raise NotImplementedError(
            "C4 requires streaming from HF which may be slow. "
            "Use 'wikitext' instead."
        )

    elif dataset == "toolace":
        from data.loader import load_toolace_splits, format_prompt

        calib_samples, _ = load_toolace_splits(
            calibration_size=n_samples, eval_size=0, seed=42
        )
        return [format_prompt(s) for s in calib_samples]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _load_eval_data(dataset: str, n_samples: int) -> tuple[list[str], list[dict]]:
    """Load and format evaluation prompts with expected outputs."""
    if dataset == "wikitext":
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        texts = [s["text"] for s in ds if s["text"].strip()]
        prompts = texts[:n_samples]
        # No expected tool calls for generic text
        expected = [{"name": "", "arguments": {}} for _ in prompts]
        return prompts, expected

    elif dataset == "toolace":
        from data.loader import load_toolace_splits, format_prompt, extract_expected_tool_calls

        _, eval_samples = load_toolace_splits(
            calibration_size=0, eval_size=n_samples, seed=42
        )
        prompts = [format_prompt(s) for s in eval_samples]
        expected = [extract_expected_tool_calls(s) for s in eval_samples]
        # Flatten expected (take first tool call per sample)
        flat_expected = []
        for exp_list in expected:
            if exp_list and len(exp_list) > 0:
                flat_expected.append(exp_list[0])
            else:
                flat_expected.append({"name": "", "arguments": {}})
        return prompts, flat_expected

    raise ValueError(f"Unknown dataset: {dataset}")


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

    try:
        if args.command == "calibrate":
            return cmd_calibrate(args)
        elif args.command == "scales":
            return cmd_scales(args)
        elif args.command == "quantize":
            return cmd_quantize(args)
        elif args.command == "benchmark":
            return cmd_benchmark(args)
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
        if "out of memory" in str(e).lower() or "mps" in str(e).lower():
            from utils.errors import diagnose_oom
            diagnose_oom(args.device or "mps")

        return 1


if __name__ == "__main__":
    sys.exit(main())
