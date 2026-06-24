"""Device-agnostic memory management for CUDA, MPS, and CPU.

Provides memory monitoring, limiting, and tracking helpers
that automatically dispatch to the correct backend.
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
        "cuda" if CUDA is available,
        "mps" if Apple MPS is available,
        "cpu" otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_count() -> int:
    """Return the number of available devices of the detected type."""
    device = get_device()
    if device == "cuda":
        return torch.cuda.device_count()
    elif device == "mps":
        return 1
    return 0


def get_device_name(device: str | None = None) -> str:
    """Return a human-readable device name."""
    if device is None:
        device = get_device()
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    elif device == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


def empty_cache() -> None:
    """Clear device cache and run Python GC.

    Dispatches to the correct backend automatically.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_memory(device: str | None = None) -> dict:
    """Get current device memory stats in GB.

    Args:
        device: "cuda", "mps", or "cpu". Auto-detected if None.

    Returns:
        dict with keys specific to the backend.
        Common keys: 'current_gb', 'peak_gb', 'total_gb'.
    """
    if device is None:
        device = get_device()

    stats: dict[str, Any] = {}

    if device == "cuda":
        stats["current_gb"] = torch.cuda.memory_allocated() / 1e9
        stats["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        stats["total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        stats["free_gb"] = max(0, stats["total_gb"] - stats["current_gb"])
    elif device == "mps":
        try:
            stats["current_gb"] = torch.mps.current_allocated_memory() / 1e9
        except Exception:
            stats["current_gb"] = 0.0
        try:
            stats["driver_gb"] = torch.mps.driver_allocated_memory() / 1e9
        except Exception:
            stats["driver_gb"] = 0.0
        try:
            stats["recommended_max_gb"] = torch.mps.recommended_max_memory() / 1e9
        except Exception:
            stats["recommended_max_gb"] = 0.0
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
    elif device == "mps":
        max_gb = mem.get("recommended_max_gb", 0)
        pct = (mem["current_gb"] / max_gb * 100) if max_gb > 0 else 0
        parts.append(f"MPS: {mem['current_gb']:.2f}/{max_gb:.1f} GB ({pct:.0f}%)")
    else:
        parts.append("CPU mode")

    if sys_mem and "free_gb" in sys_mem:
        parts.append(f"RAM free: {sys_mem['free_gb']:.1f} GB")

    print(" | ".join(parts), flush=True)


def limit_memory(fraction: float = 0.7, device: str | None = None) -> None:
    """Limit device memory to a fraction of total.

    On MPS, this uses ``torch.mps.set_per_process_memory_fraction``.
    On CUDA, this is a no-op (use ``torch.cuda.set_per_process_memory_fraction``
    if available in your PyTorch version).
    On CPU, this is a no-op.

    Args:
        fraction: Fraction of total memory to allow (0.1 - 1.0).
        device: Target device. Auto-detected if None.
    """
    if device is None:
        device = get_device()
    fraction = max(0.1, min(1.0, fraction))

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(fraction)
        max_mem = torch.mps.recommended_max_memory()
        print(f"[MEM] MPS limited to {fraction:.0%} = {max_mem / 1e9:.1f} GB")
    elif device == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(fraction)
            total = torch.cuda.get_device_properties(0).total_memory
            print(f"[MEM] CUDA limited to {fraction:.0%} = {total * fraction / 1e9:.1f} GB")
        except Exception:
            print(f"[MEM] CUDA memory fraction not supported in this PyTorch version")
    else:
        print(f"[MEM] Memory limiting not supported on {device}")


@contextmanager
def memory_tracker(tag: str = "track", device: str | None = None):
    """Context manager that logs memory before/after a block.

    Also emptys cache on exit.

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


# ── Private helpers ──────────────────────────────────────────────────


def _get_system_ram_gb() -> float:
    """Get total system RAM in GB (cross-platform)."""
    import platform

    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            return int(result.stdout.strip()) / 1e9
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / 1e6
        return 16.0  # fallback
    except Exception:
        return 16.0


def _get_system_memory() -> dict:
    """Get system RAM usage stats."""
    import platform

    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = result.stdout.strip().split("\n")
            page_size = 16384
            stats: dict[str, float] = {}
            for line in lines:
                if "page size" in line.lower():
                    try:
                        page_size = int(line.split()[-1])
                    except (ValueError, IndexError):
                        pass
                if "free" in line.lower():
                    try:
                        stats["free_pages"] = int(line.split(":")[1].strip().rstrip("."))
                    except (ValueError, IndexError):
                        pass
                if "active" in line.lower() and "file-backed" not in line.lower():
                    try:
                        stats["active_pages"] = int(line.split(":")[1].strip().rstrip("."))
                    except (ValueError, IndexError):
                        pass
                if "wired" in line.lower():
                    try:
                        stats["wired_pages"] = int(line.split(":")[1].strip().rstrip("."))
                    except (ValueError, IndexError):
                        pass

            free_gb = stats.get("free_pages", 0) * page_size / 1e9
            active_gb = stats.get("active_pages", 0) * page_size / 1e9
            wired_gb = stats.get("wired_pages", 0) * page_size / 1e9
            return {
                "free_gb": round(free_gb, 1),
                "active_gb": round(active_gb, 1),
                "wired_gb": round(wired_gb, 1),
                "used_gb": round(active_gb + wired_gb, 1),
            }
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                mem = {}
                for line in f:
                    if line.startswith("MemFree:"):
                        mem["free_gb"] = int(line.split()[1]) / 1e6
                    if line.startswith("MemAvailable:"):
                        mem["avail_gb"] = int(line.split()[1]) / 1e6
                return mem
        return {"free_gb": 0.0}
    except Exception:
        return {"free_gb": 0.0}
