"""Memory management for CUDA and CPU.

Provides memory monitoring, cleanup, and tracking helpers that target CUDA
GPUs (with a CPU fallback). The Apple-Silicon/MPS path was removed when the
project pivoted to a CUDA-first runtime.
"""

import gc
import os
import subprocess
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch


def get_device() -> str:
    """Auto-detect the best available device.

    Returns:
        "cuda" if CUDA is available, "cpu" otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_count() -> int:
    """Return the number of available devices of the detected type."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_device_name(device: str | None = None) -> str:
    """Return a human-readable device name."""
    if device is None:
        device = get_device()
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"


def empty_cache() -> None:
    """Clear device cache and run Python GC (CUDA only; no-op on CPU)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory(device: str | None = None) -> dict:
    """Get current device memory stats in GB.

    Args:
        device: "cuda" or "cpu". Auto-detected if None.

    Returns:
        dict with keys 'current_gb', 'peak_gb', 'total_gb', 'free_gb' (CUDA),
        or 'current_gb', 'total_gb' (CPU).
    """
    if device is None:
        device = get_device()

    stats: dict[str, Any] = {}

    if device == "cuda":
        stats["current_gb"] = torch.cuda.memory_allocated() / 1e9
        stats["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        stats["total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        stats["free_gb"] = max(0, stats["total_gb"] - stats["current_gb"])
    else:
        stats["current_gb"] = 0.0
        stats["total_gb"] = _get_system_ram_gb()

    return stats


def log_memory(tag: str = "", device: str | None = None) -> None:
    """Print current memory stats to stderr."""
    if device is None:
        device = get_device()
    mem = get_memory(device)
    sys_mem = _get_system_memory()

    parts = []
    if tag:
        parts.append(f"[{tag}]")

    if device == "cuda":
        pct = (mem["current_gb"] / mem["total_gb"] * 100) if mem.get("total_gb", 0) > 0 else 0
        parts.append(f"GPU: {mem['current_gb']:.2f}/{mem['total_gb']:.1f} GB ({pct:.0f}%)")
    else:
        parts.append("CPU mode")

    if sys_mem and "free_gb" in sys_mem:
        parts.append(f"RAM free: {sys_mem['free_gb']:.1f} GB")

    print(" | ".join(parts), flush=True)


@contextmanager
def memory_tracker(tag: str = "track", device: str | None = None):
    """Context manager that logs memory before/after a block.

    Also empties cache on exit.

    Usage:
        with memory_tracker("calibration_pass"):
            run_calibration(...)
    """
    if device is None:
        device = get_device()
    log_memory(f"{tag}_start", device)
    try:
        yield
    finally:
        empty_cache()
        log_memory(f"{tag}_end", device)


def batch_generator(
    items: list,
    batch_size: int = 10,
    progress_callback: Callable | None = None,
):
    """Yield items in batches with optional progress callback."""
    total = len(items)
    for i in range(0, total, batch_size):
        if progress_callback:
            progress_callback(min(i + batch_size, total), total)
        yield items[i : i + batch_size]


# ── Private helpers (Linux only) ─────────────────────────────────────


def _get_system_ram_gb() -> float:
    """Get total system RAM in GB (Linux)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / 1e6
        return 16.0  # fallback
    except Exception:
        return 16.0


def _get_system_memory() -> dict:
    """Get system RAM usage stats (Linux /proc/meminfo)."""
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                if line.startswith("MemFree:"):
                    mem["free_gb"] = int(line.split()[1]) / 1e6
                if line.startswith("MemAvailable:"):
                    mem["avail_gb"] = int(line.split()[1]) / 1e6
            return mem
    except Exception:
        return {"free_gb": 0.0}