"""AWQ calibration pass — memory-efficient version.

Runs calibration samples through the FP16 model and records
activation magnitudes per weight channel for each linear layer.

Aggregates channel importance ON-THE-FLY inside the hook,
so memory stays O(d_in) per layer instead of O(n_samples × tokens × d_in).
"""

import gc
import os
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.memory import memory_tracker, get_device


def is_linear_layer(module: torch.nn.Module) -> bool:
    """Check if a module is a linear layer (not embedding, not norm)."""
    return isinstance(module, torch.nn.Linear)


def register_calibration_hooks(
    model: AutoModelForCausalLM,
) -> tuple[dict[str, torch.Tensor], dict[str, int], list[Any]]:
    """Register forward hooks that aggregate channel importance on-the-fly.

    Unlike the naive approach of storing every activation tensor and
    concatenating later, this hook maintains a RUNNING SUM of
    |activation|.mean(0) per layer — only O(d_in) memory per layer.

    Args:
        model: The FP16 model.

    Returns:
        (running_sums, running_counts, hook_handle_list)
        - running_sums: {layer_name: running_sum_tensor}
        - running_counts: {layer_name: number_of_samples_seen}
        - hook_handles: list of hooks for cleanup
    """
    running_sums: dict[str, torch.Tensor] = {}
    running_counts: dict[str, int] = {}
    hooks: list[Any] = []

    def _make_hook(layer_name: str):
        def _hook(module, inputs, _outputs):
            """Aggregate channel importance on-the-fly."""
            x = inputs[0].detach()
            if x.dim() == 3:
                x = x.view(-1, x.size(-1))
            elif x.dim() != 2:
                return

            ci = x.abs().mean(dim=0).to("cpu")

            if layer_name not in running_sums:
                running_sums[layer_name] = ci
                running_counts[layer_name] = 1
            else:
                running_sums[layer_name] += ci
                running_counts[layer_name] += 1

        return _hook

    for name, module in model.named_modules():
        if is_linear_layer(module):
            hook = module.register_forward_hook(_make_hook(name))
            hooks.append(hook)

    return running_sums, running_counts, hooks


def run_calibration(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    calibration_prompts: list[str],
    output_path: str | None = None,
    max_length: int = 4096,
    device: str | None = None,
    verbose: bool = True,
    batch_size: int = 10,
) -> dict[str, torch.Tensor]:
    """Run AWQ calibration: forward calibration samples, collect activation stats.

    MEMORY-EFFICIENT: Aggregates channel importance on-the-fly in hooks,
    processes in batches, and runs cache cleanup between batches.

    Args:
        model: FP16 model on device.
        tokenizer: Model tokenizer.
        calibration_prompts: List of formatted prompt strings.
        output_path: Where to save calibration stats .pt file.
            If None, stats are not saved to disk (just returned).
        max_length: Max sequence length per sample.
        device: Torch device string. Auto-detected if None.
        verbose: Print progress.
        batch_size: Number of samples per batch (lower = less memory).

    Returns:
        dict of {layer_name: channel_importance_tensor}
    """
    if device is None:
        device = get_device()

    running_sums, running_counts, hooks = register_calibration_hooks(model)

    model.eval()
    n_prompts = len(calibration_prompts)

    if verbose:
        print(f"Running calibration on {n_prompts} samples "
              f"(batch_size={batch_size}, max_length={max_length}, device={device})...")

    # Process in batches to keep memory bounded
    with torch.no_grad():
        for batch_start in range(0, n_prompts, batch_size):
            batch = calibration_prompts[batch_start:batch_start + batch_size]
            batch_end = min(batch_start + batch_size, n_prompts)

            if verbose:
                print(f"  Batch [{batch_start + 1}-{batch_end}/{n_prompts}]...", end=" ", flush=True)

            for prompt in batch:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(device)

                # Forward — hooks aggregate channel importance
                model(**inputs)

                # Immediately free per-prompt memory
                del inputs

            if verbose:
                print("clearing cache...", flush=True)

            # Aggressive cleanup between batches
            gc.collect()
            _empty_device_cache(device)

    # Cleanup hooks
    for hook in hooks:
        hook.remove()

    # Average the running sums
    calibration_stats: dict[str, torch.Tensor] = {}
    for layer_name, running_sum in running_sums.items():
        count = running_counts.get(layer_name, 1)
        calibration_stats[layer_name] = running_sum / count

        if verbose:
            d_in = calibration_stats[layer_name].size(0)
            max_ci = calibration_stats[layer_name].max().item()
            mean_ci = calibration_stats[layer_name].mean().item()
            print(f"  {layer_name:<55} d_in={d_in:<5} samples={count:<4} "
                  f"max_ci={max_ci:.4f}  mean_ci={mean_ci:.4f}")

    # Save if output_path is provided
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(calibration_stats, output_path)
        if verbose:
            print(f"\nSaved calibration stats → {output_path}")
            print(f"  Layers captured: {len(calibration_stats)}")

    return calibration_stats


def _empty_device_cache(device: str) -> None:
    """Empty device cache based on backend."""
    if device == "cuda":
        torch.cuda.empty_cache()
