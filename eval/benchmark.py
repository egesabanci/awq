#!/usr/bin/env python3
"""Main benchmark: run FP16 baseline or AWQ comparison and report results."""

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch

from data.loader import save_splits, format_prompt, extract_expected_tool_calls, load_splits


def run_baseline(
    model_path: str,
    prompts: list[str],
    expected: list[dict] | None = None,
    output_path: str | None = None,
    max_new_tokens: int = 512,
    device: str | None = None,
) -> dict:
    """Run FP16 baseline end-to-end."""
    from eval.runner import run_fp16_baseline

    results = run_fp16_baseline(
        model_path=model_path,
        prompts=prompts,
        expected=expected,
        output_path=output_path,
        max_new_tokens=max_new_tokens,
        device=device,
        progress_callback=lambda i, n: print(f"  [{i}/{n}]", end="\r" if i < n else "\n"),
    )

    # Evaluate baseline outputs (self-comparison)
    from eval.metrics import run_full_evaluation

    eval_results = run_full_evaluation(
        fp16_outputs=results["outputs"],
        awq_outputs=results["outputs"],  # self-comparison for baseline
        expected=results.get("expected", []),
    )
    results["evaluation"] = eval_results
    _print_summary(results)

    return results


def run_awq_comparison(
    fp16_model_path: str,
    awq_quantized_path: str,
    prompts: list[str],
    expected: list[dict] | None = None,
    output_path: str | None = None,
    max_new_tokens: int = 512,
    device: str | None = None,
) -> dict:
    """Compare FP16 vs AWQ-quantized outputs."""
    from eval.runner import run_fp16_baseline
    from awq.inference import load_awq_model as load_awq_model_func

    if device is None:
        from utils.memory import get_device
        device = get_device()

    # Run FP16 baseline
    print("=" * 60)
    print("FP16 BASELINE")
    print("=" * 60)
    fp16_results = run_fp16_baseline(
        model_path=fp16_model_path,
        prompts=prompts,
        expected=expected,
        output_path=None,
        max_new_tokens=max_new_tokens,
        device=device,
        progress_callback=lambda i, n: print(f"  [{i}/{n}]", end="\r" if i < n else "\n"),
    )

    # Load and run AWQ model
    print("\n" + "=" * 60)
    print("AWQ QUANTIZED")
    print("=" * 60)
    from utils.memory import memory_tracker

    with memory_tracker("load_awq", device):
        model, tokenizer, q_state = load_awq_model_func(awq_quantized_path, fp16_model_path, device=device)

    from eval.runner import generate_text

    awq_outputs = []
    tokens_per_sec_list = []
    print(f"\nGenerating {len(prompts)} prompts with AWQ model...")
    for i, prompt in enumerate(prompts):
        out, tps = generate_text(model, tokenizer, prompt, max_new_tokens, device=device)
        awq_outputs.append(out)
        tokens_per_sec_list.append(tps)
        print(f"  [{i+1}/{len(prompts)}]", end="\r")
    print()

    # Compare
    from eval.metrics import run_full_evaluation

    eval_results = run_full_evaluation(
        fp16_outputs=fp16_results["outputs"],
        awq_outputs=awq_outputs,
        expected=fp16_results.get("expected", []),
    )

    # Merge results
    comparison = {
        "model": fp16_model_path,
        "awq_path": awq_quantized_path,
        "device": device,
        "num_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "fp16": {
            "timing": fp16_results["timing"],
            "memory": fp16_results["memory"],
        },
        "awq": {
            "mean_tokens_per_sec": round(float(np.mean(tokens_per_sec_list)), 2) if tokens_per_sec_list else 0,
        },
        "evaluation": eval_results,
    }

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\nSaved comparison → {output_path}")

    # Print summary
    _print_awq_summary(comparison)

    return comparison


def generate_report(results_path: str | None = None) -> str:
    """Generate a human-readable summary from saved results."""
    from utils.memory import get_device

    # Generate a simple report
    if results_path and os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        print("No results found to report")
        return ""

    lines = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"- **Model:** {results.get('model', 'N/A')}")
    lines.append(f"- **Device:** {results.get('device', get_device())}")
    lines.append(f"- **Samples:** {results.get('num_prompts', 'N/A')}")
    lines.append("")

    eval_data = results.get("evaluation", {})
    metrics = eval_data.get("per_metric", {})

    lines.append("## Quality Metrics")
    lines.append("")
    lines.append("| Metric | FP16 | AWQ | Delta |")
    lines.append("|---|---|---|---|")
    for metric_name, scores in metrics.items():
        display = metric_name.replace("_", " ").title()
        fp16 = f"{scores.get('fp16_mean', 0)*100:.1f}%"
        awq = f"{scores.get('awq_mean', 0)*100:.1f}%"
        delta = f"{scores.get('delta', 0)*100:+.1f}%"
        lines.append(f"| {display} | {fp16} | {awq} | {delta} |")
    lines.append("")

    sem = eval_data.get("semantic_similarity", {})
    lines.append("## Semantic Similarity")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Mean cosine | {sem.get('mean', 'N/A'):.4f} |")
    lines.append(f"| Std | {sem.get('std', 'N/A'):.4f} |")
    lines.append("")

    report = "\n".join(lines)
    return report


def _print_summary(results: dict) -> None:
    """Print formatted summary table for baseline runs."""
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
    print(f"\n⚡ Throughput: {timing.get('mean_tokens_per_sec', '?'):.1f} tok/s")
    print(f"💾 Peak memory: {memory.get('peak_allocated_gb', '?'):.2f} GB\n")


def _print_awq_summary(comparison: dict) -> None:
    """Print formatted summary for AWQ comparison runs."""
    eval_data = comparison.get("evaluation", {})
    metrics = eval_data.get("per_metric", {})

    print("\n" + "=" * 60)
    print("AWQ COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n📊 Quality Metrics:")
    print(f"  {'Metric':<35} {'FP16':<10} {'AWQ':<10} {'Δ':<10}")
    print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10}")
    for metric_name, scores in metrics.items():
        display = metric_name.replace("_", " ").title()
        fp16 = f"{scores.get('fp16_mean', 0)*100:.1f}%"
        awq = f"{scores.get('awq_mean', 0)*100:.1f}%"
        delta = f"{scores.get('delta', 0)*100:+.1f}%"
        print(f"  {display:<35} {fp16:<10} {awq:<10} {delta:<10}")

    fp16_timing = comparison.get("fp16", {}).get("timing", {})
    awq_tps = comparison.get("awq", {}).get("mean_tokens_per_sec", 0)
    print(f"\n⚡ Throughput: FP16={fp16_timing.get('mean_tokens_per_sec', '?'):.1f} vs AWQ={awq_tps:.1f} tok/s")

    sem = eval_data.get("semantic_similarity", {})
    print(f"📐 Semantic cosine: {sem.get('mean', 0):.4f}\n")
