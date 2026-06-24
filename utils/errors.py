"""Error handling utilities for the AWQ pipeline.

Provides custom exception classes, retry decorators,
and validation helpers for pipeline stages.
"""

import time
import functools
from typing import Any, Callable, TypeVar

import torch

F = TypeVar("F", bound=Callable[..., Any])


# ── Custom Exceptions ─────────────────────────────────────────────────


class AWQError(Exception):
    """Base exception for all AWQ pipeline errors."""


class ModelLoadError(AWQError):
    """Raised when model loading fails."""


class CalibrationError(AWQError):
    """Raised when calibration pass fails."""


class ScaleError(AWQError):
    """Raised when scale computation fails."""


class QuantizationError(AWQError):
    """Raised when quantization fails."""


class InferenceError(AWQError):
    """Raised during inference with quantized weights."""


class ValidationError(AWQError):
    """Raised when pipeline input validation fails."""


# ── Retry Decorator ───────────────────────────────────────────────────


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        backoff_factor: Multiplier for delay after each retry.
        exceptions: Tuple of exception types that trigger a retry.

    Usage:
        @retry_with_backoff(max_retries=3)
        def load_model(path):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** (attempt - 1))
                        print(f"[RETRY] {func.__name__} failed (attempt {attempt}/{max_retries}): "
                              f"{e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        print(f"[RETRY] {func.__name__} failed after {max_retries} attempts: {e}")
            raise last_exception  # type: ignore
        return wrapper  # type: ignore
    return decorator


# ── Memory Diagnostics ───────────────────────────────────────────────


def diagnose_oom(device: str) -> None:
    """Print memory diagnostics during an OOM situation."""
    import gc

    gc.collect()
    print("\n[OOM] Memory diagnostics:")
    print(f"  Device: {device}")

    try:
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU allocated: {allocated:.2f}/{total:.1f} GB")
            print(f"  GPU cached:    {cached:.2f} GB")
        elif device == "mps":
            current = torch.mps.current_allocated_memory() / 1e9
            driver = torch.mps.driver_allocated_memory() / 1e9
            recommended = torch.mps.recommended_max_memory() / 1e9
            print(f"  MPS current:  {current:.2f} GB")
            print(f"  MPS driver:   {driver:.2f} GB")
            print(f"  MPS max:      {recommended:.2f} GB")
    except Exception as e:
        print(f"  (memory query failed: {e})")

    print(f"\n  Suggestion: reduce --batch-size, --max-length, or --samples")
    print()


# ── Pipeline Stage Validation ─────────────────────────────────────────


def require_calibration_stats(path: str) -> dict:
    """Load and validate calibration stats file.

    Raises ValidationError if the file is missing or malformed.
    """
    import os

    if not os.path.exists(path):
        raise ValidationError(
            f"Calibration stats not found at {path}. "
            f"Run 'awq calibrate' first."
        )
    try:
        stats = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ValidationError(
            f"Failed to load calibration stats from {path}: {e}"
        )

    if not isinstance(stats, dict):
        raise ValidationError(
            f"Expected dict of calibration stats, got {type(stats).__name__}"
        )
    if len(stats) == 0:
        raise ValidationError("Calibration stats dict is empty")

    # Validate structure: each value should be a 1D tensor
    for name, val in stats.items():
        if not isinstance(val, torch.Tensor):
            raise ValidationError(
                f"Calibration stat '{name}' is not a tensor (got {type(val).__name__})"
            )
        if val.dim() != 1:
            raise ValidationError(
                f"Calibration stat '{name}' has shape {val.shape}, expected 1D"
            )

    return stats


def require_scales(path: str) -> dict:
    """Load and validate AWQ scales file.

    Raises ValidationError if the file is missing or malformed.
    """
    import os

    if not os.path.exists(path):
        raise ValidationError(
            f"AWQ scales not found at {path}. "
            f"Run 'awq scales' first."
        )
    try:
        scales = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ValidationError(
            f"Failed to load scales from {path}: {e}"
        )

    if not isinstance(scales, dict):
        raise ValidationError(
            f"Expected dict of scales, got {type(scales).__name__}"
        )
    if len(scales) == 0:
        raise ValidationError("Scales dict is empty")

    return scales


def require_quantized_dir(path: str) -> str:
    """Validate that a quantized model directory exists and has the expected files.

    Returns the path to quantized_state.pt.
    """
    import os

    if not os.path.isdir(path):
        raise ValidationError(
            f"Quantized model directory not found at {path}. "
            f"Run 'awq quantize' first."
        )

    state_path = os.path.join(path, "quantized_state.pt")
    if not os.path.exists(state_path):
        raise ValidationError(
            f"Quantized state file not found at {state_path}. "
            f"Run 'awq quantize' first."
        )

    meta_path = os.path.join(path, "metadata.json")
    if not os.path.exists(meta_path):
        raise ValidationError(
            f"Quantized metadata not found at {meta_path}. "
            f"Run 'awq quantize' first."
        )

    return state_path


# ── Exit Codes ────────────────────────────────────────────────────────


EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_VALIDATION_ERROR = 2
