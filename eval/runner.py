"""FP16 baseline inference runner for Qwen3.5-2B on Apple Silicon MPS."""

import json
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


@torch.no_grad()
def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: str = "mps",
) -> tuple[str, float]:
    """Generate a single response from a prompt.

    Returns:
        (generated_text, tokens_per_second)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    input_len = inputs["input_ids"].size(1)

    if temperature == 0.0:
        do_sample = False
    else:
        do_sample = True

    t0 = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - t0

    generated_ids = outputs[0][input_len:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)

    new_tokens = len(generated_ids)
    tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0.0

    return generated.strip(), tokens_per_sec


def load_fp16_model(
    model_path: str,
    device: str = "mps",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen3.5-2B in FP16 on the specified device."""
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    return model, tokenizer


def run_fp16_baseline(
    model_path: str,
    prompts: list[str],
    expected: list[dict] | None = None,
    output_path: str | None = None,
    max_new_tokens: int = 512,
    device: str = "mps",
    progress_callback: Callable | None = None,
) -> dict:
    """Run FP16 inference on all prompts and save outputs.

    Args:
        model_path: Path to local model directory.
        prompts: List of formatted prompt strings.
        expected: Optional list of expected tool calls (for eval).
        output_path: Where to save results JSON.
        max_new_tokens: Max tokens per generation.
        device: Torch device string.
        progress_callback: Optional fn(i, total) for status updates.

    Returns:
        Benchmark result dict with outputs, timing, and memory stats.
    """
    model, tokenizer = load_fp16_model(model_path, device)

    # Record baseline memory
    torch.mps.empty_cache()
    peak_mem_before = _get_current_memory(device)

    outputs: list[str] = []
    latencies: list[float] = []
    tokens_per_sec_list: list[float] = []
    total_tokens = 0

    print(f"\nGenerating {len(prompts)} prompts...")
    t_start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        if progress_callback:
            progress_callback(i + 1, len(prompts))

        out, tps = generate_text(model, tokenizer, prompt, max_new_tokens, device=device)
        outputs.append(out)
        tokens_per_sec_list.append(tps)
        total_tokens += len(out.split())

    total_elapsed = time.perf_counter() - t_start

    # Memory after
    peak_mem_after = _get_current_memory(device)
    current_mem = _get_current_memory(device)

    # Build result
    result = {
        "model": model_path,
        "device": device,
        "dtype": "float16",
        "num_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "outputs": outputs,
        "expected": expected,
        "timing": {
            "total_seconds": round(total_elapsed, 2),
            "mean_tokens_per_sec": round(float(np.mean(tokens_per_sec_list)), 2),
            "median_tokens_per_sec": round(float(np.median(tokens_per_sec_list)), 2),
            "mean_latency_seconds": round(float(np.mean(latencies)), 3) if latencies else None,
            "total_tokens_generated": total_tokens,
        },
        "memory": {
            "peak_allocated_gb": round(peak_mem_after, 2),
            "current_allocated_gb": round(current_mem, 2),
            "savings_vs_model_size_gb": None,  # Will be filled by AWQ comparison
        },
    }

    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "fp16_outputs.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nSaved FP16 results → {output_path}")

    # Cleanup
    del model
    if device == "mps":
        torch.mps.empty_cache()

    return result


# ── Memory helpers ────────────────────────────────────────────────────

def _get_current_memory(device: str) -> float:
    """Get current memory allocation in GB."""
    if device == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    elif device == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FP16 baseline inference")
    parser.add_argument("--model-path", default="models/Qwen3.5-2B-FP16", help="Path to FP16 model")
    parser.add_argument("--prompts-path", default="data/processed/evaluation.json", help="Path to eval prompts JSON")
    parser.add_argument("--output-path", default=None, help="Where to save results")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from data.loader import load_splits, format_prompt

    _, eval_samples = load_splits(data_dir=os.path.dirname(args.prompts_path))
    prompts = [format_prompt(s) for s in eval_samples]
    expected = [{"name": "test", "arguments": {}} for _ in eval_samples]  # placeholder

    results = run_fp16_baseline(
        model_path=args.model_path,
        prompts=prompts,
        expected=expected,
        output_path=args.output_path,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        progress_callback=lambda i, n: print(f"  [{i}/{n}]", end="\r" if i < n else "\n"),
    )

    print(f"\nMean throughput: {results['timing']['mean_tokens_per_sec']:.1f} tok/s")
    print(f"Peak memory:     {results['memory']['peak_allocated_gb']:.2f} GB")
