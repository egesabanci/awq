#!/usr/bin/env python3
"""Main benchmark: load FP16 baseline, evaluate, report results.

Usage:
    python eval/benchmark.py                          # full run
    python eval/benchmark.py --mode baseline          # just baseline
    python eval/benchmark.py --mode report            # generate report from saved outputs
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from data.loader import load_splits, format_prompt, extract_expected_tool_calls

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def run_baseline(args: argparse.Namespace) -> dict:
    """Run the FP16 baseline end-to-end."""
    from eval.runner import run_fp16_baseline

    print("=" * 60)
    print("PHASE 1 — FP16 Baseline")
    print("=" * 60)

    # Load evaluation samples
    print(f"\nLoading evaluation split from {args.data_dir}...")
    _, eval_samples = load_splits(data_dir=args.data_dir)
    print(f"  {len(eval_samples)} evaluation samples loaded")

    # Format prompts
    prompts = [format_prompt(s) for s in eval_samples]
    expected = [extract_expected_tool_calls(s) for s in eval_samples]

    # Run inference
    results = run_fp16_baseline(
        model_path=args.model_path,
        prompts=prompts,
        expected=expected,
        output_path=os.path.join(RESULTS_DIR, "fp16_baseline.json"),
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        progress_callback=lambda i, n: print(f"  [{i}/{n}]", end="\r" if i < n else "\n"),
    )

    # Evaluate
    print("\nEvaluating baseline outputs...")
    from eval.metrics import run_full_evaluation

    eval_results = run_full_evaluation(
        fp16_outputs=results["outputs"],
        awq_outputs=results["outputs"],  # self-comparison for baseline sanity
        expected=results.get("expected", []),
    )

    # Merge results
    results["evaluation"] = eval_results

    # Save merged
    merged_path = os.path.join(RESULTS_DIR, "baseline_report.json")
    with open(merged_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved baseline report → {merged_path}")

    # Print summary table
    _print_summary(results)

    return results


def generate_report(args: argparse.Namespace) -> str:
    """Generate human-readable BASELINE.md from saved results."""
    report_path = os.path.join(RESULTS_DIR, "BASELINE.md")

    # Load results if they exist
    baseline_path = os.path.join(RESULTS_DIR, "baseline_report.json")
    if not os.path.exists(baseline_path):
        print(f"No baseline report found at {baseline_path}. Run --mode baseline first.")
        return ""

    with open(baseline_path) as f:
        results = json.load(f)

    timing = results.get("timing", {})
    memory = results.get("memory", {})
    eval_data = results.get("evaluation", {})
    metrics = eval_data.get("per_metric", {})

    lines = []
    lines.append("# FP16 Baseline Report")
    lines.append("")
    lines.append(f"- **Model:** {results.get('model', 'N/A')}")
    lines.append(f"- **Device:** {results.get('device', 'N/A')}")
    lines.append(f"- **Dtype:** {results.get('dtype', 'N/A')}")
    lines.append(f"- **Samples:** {results.get('num_prompts', 'N/A')}")
    lines.append("")

    # Timing
    lines.append("## Timing")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total time | {timing.get('total_seconds', 'N/A')} s |")
    lines.append(f"| Mean throughput | {timing.get('mean_tokens_per_sec', 'N/A')} tok/s |")
    lines.append(f"| Median throughput | {timing.get('median_tokens_per_sec', 'N/A')} tok/s |")
    lines.append(f"| Total tokens generated | {timing.get('total_tokens_generated', 'N/A')} |")
    lines.append("")

    # Memory
    lines.append("## Memory")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Peak allocated | {memory.get('peak_allocated_gb', 'N/A')} GB |")
    lines.append(f"| Current allocated | {memory.get('current_allocated_gb', 'N/A')} GB |")
    lines.append("")

    # Quality metrics (self-consistency check)
    lines.append("## Tool-Calling Quality (Self-Consistency Check)")
    lines.append("")
    lines.append("| Metric | Mean | Std |")
    lines.append("|--------|------|-----|")
    for metric_name, scores in metrics.items():
        display = metric_name.replace("_", " ").title()
        lines.append(f"| {display} | {scores.get('fp16_mean', 0)*100:.1f}% | {scores.get('fp16_std', 0)*100:.1f}% |")
    lines.append("")

    # Semantic similarity (self → self should be 1.0)
    sem = eval_data.get("semantic_similarity", {})
    lines.append("## Semantic Similarity (Baseline Self-Check)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Mean cosine | {sem.get('mean', 'N/A')} |")
    lines.append(f"| Std | {sem.get('std', 'N/A')} |")
    lines.append(f"| Min | {sem.get('min', 'N/A')} |")
    lines.append("")

    # Sample outputs
    lines.append("## Sample Outputs")
    lines.append("")
    outputs = results.get("outputs", [])
    for i, out in enumerate(outputs[:5]):
        lines.append(f"### Sample {i+1}")
        lines.append("")
        lines.append("```")
        lines.append(out[:300])
        if len(out) > 300:
            lines.append("... (truncated)")
        lines.append("```")
        lines.append("")

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved baseline report → {report_path}")
    return report_path


def _print_summary(results: dict) -> None:
    """Print a formatted summary table to stdout."""
    eval_data = results.get("evaluation", {})
    metrics = eval_data.get("per_metric", {})
    timing = results.get("timing", {})
    memory = results.get("memory", {})

    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)

    print(f"\n📊 Quality Metrics:")
    print(f"  {'Metric':<35} {'Score':<10}")
    print(f"  {'─'*35} {'─'*10}")
    for metric_name, scores in metrics.items():
        display = metric_name.replace("_", " ").title()
        print(f"  {display:<35} {scores.get('fp16_mean', 0)*100:>5.1f}%")

    print(f"\n⚡ Performance:")
    print(f"  Throughput:    {timing.get('mean_tokens_per_sec', '?'):>6.1f} tok/s")
    print(f"  Total tokens:  {timing.get('total_tokens_generated', '?'):>6}")
    print(f"  Total time:    {timing.get('total_seconds', '?'):>6.1f} s")

    print(f"\n💾 Memory:")
    print(f"  Peak:          {memory.get('peak_allocated_gb', '?'):>5.2f} GB")
    print(f"  Current:       {memory.get('current_allocated_gb', '?'):>5.2f} GB")

    print(f"\n  ✅ Baseline complete — ready for AWQ comparison.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWQ ToolCall Benchmark")
    parser.add_argument("--mode", choices=["baseline", "report", "all"], default="all")
    parser.add_argument("--model-path", default="models/Qwen3.5-2B-FP16")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode in ("baseline", "all"):
        run_baseline(args)

    if args.mode in ("report", "all"):
        generate_report(args)

    print("\nDone.")
