"""FP16 baseline inference runner for Qwen3.5 models."""

import json
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.memory import get_device, log_memory, empty_cache
from utils.errors import retry_with_backoff


@retry_with_backoff(max_retries=3)
def load_fp16_model(
    model_path: str,
    device: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model in FP16 on the specified device.

    Args:
        model_path: Path to local model directory or HuggingFace repo ID.
        device: Torch device string. Auto-detected if None.

    Returns:
        (model, tokenizer)
    """
    if device is None:
        device = get_device()

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


@torch.no_grad()
def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: str | None = None,
) -> tuple[str, float]:
    """Generate a single response from a prompt.

    Returns:
        (generated_text, tokens_per_second)
    """
    if device is None:
        device = get_device()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    input_len = inputs["input_ids"].size(1)

    do_sample = temperature > 0.0

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


def run_fp16_baseline(
    model_path: str,
    prompts: list[str],
    expected: list[dict] | None = None,
    output_path: str | None = None,
    max_new_tokens: int = 512,
    device: str | None = None,
    progress_callback: Callable | None = None,
) -> dict:
    """Run FP16 inference on all prompts and save outputs.

    Args:
        model_path: Path to local model directory.
        prompts: List of formatted prompt strings.
        expected: Optional list of expected tool calls (for eval).
        output_path: Where to save results JSON. If None, not saved.
        max_new_tokens: Max tokens per generation.
        device: Torch device string. Auto-detected if None.
        progress_callback: Optional fn(i, total) for status updates.

    Returns:
        Benchmark result dict with outputs, timing, and memory stats.
    """
    if device is None:
        device = get_device()

    model, tokenizer = load_fp16_model(model_path, device)

    # Record baseline memory
    empty_cache()
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
        },
    }

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved FP16 results → {output_path}")

    # Cleanup
    del model
    empty_cache()

    return result


def _get_current_memory(device: str) -> float:
    """Get current memory allocation in GB."""
    if device == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    elif device == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1e9
        except Exception:
            return 0.0
    return 0.0
