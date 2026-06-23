"""ToolACE data loader and prompt formatter for AWQ experimentation."""

import json
import os
from typing import Any

from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_toolace_splits(
    calibration_size: int = 200,
    eval_size: int = 100,
    seed: int = 42,
    cache_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load ToolACE dataset and split into calibration and evaluation sets.

    Args:
        calibration_size: Number of samples for AWQ calibration.
        eval_size: Number of samples for final evaluation.
        seed: Random seed for deterministic splitting.
        cache_dir: Optional cache directory for datasets.

    Returns:
        (calibration_samples, eval_samples) — each a list of raw ToolACE samples.
    """
    ds = load_dataset("lockon/ToolACE", split="train", cache_dir=cache_dir)
    total = calibration_size + eval_size
    if len(ds) < total:
        raise ValueError(
            f"ToolACE has {len(ds)} samples, need at least {total}. "
            f"Reduce calibration_size or eval_size."
        )

    shuffled = ds.shuffle(seed=seed)
    calibration = [shuffled[i] for i in range(calibration_size)]
    evaluation = [
        shuffled[i] for i in range(calibration_size, calibration_size + eval_size)
    ]
    return calibration, evaluation


def extract_system_prompt(sample: dict) -> str:
    """Extract the system prompt (with function definitions) from a ToolACE sample."""
    system = sample.get("system", "")
    return system


def extract_user_query(sample: dict) -> str:
    """Extract the user query from the conversation."""
    conversations = sample.get("conversations", [])
    for turn in conversations:
        if turn.get("from") == "user":
            return turn.get("value", "")
    return ""


def extract_expected_tool_calls(sample: dict) -> list[dict[str, Any]]:
    """Extract ground-truth tool call(s) that the assistant should produce.

    ToolACE format: assistant turn contains the tool call string, e.g.
    '[GetStockPrice(symbol="AAPL")]' or similar.

    Returns a list of dicts with keys 'name' and 'arguments'.
    """
    conversations = sample.get("conversations", [])
    expected: list[dict[str, Any]] = []
    for turn in conversations:
        if turn.get("from") == "assistant":
            raw = turn.get("value", "").strip()
            parsed = _parse_tool_call_string(raw)
            if parsed:
                expected.append(parsed)
    return expected


def format_prompt(sample: dict) -> str:
    """Format a ToolACE sample into a model input string.

    Uses the Qwen3 chat template style:
        <|im_start|>system
        ...function definitions...
        <|im_end|>
        <|im_start|>user
        ...query...
        <|im_end|>
        <|im_start|>assistant
    """
    system = extract_system_prompt(sample)
    user_query = extract_user_query(sample)

    parts = [
        "<|im_start|>system",
        system,
        "<|im_end|>",
        "<|im_start|>user",
        user_query,
        "<|im_end|>",
        "<|im_start|>assistant",
    ]
    return "\n".join(parts)


def save_splits(
    calibration: list[dict],
    evaluation: list[dict],
    output_dir: str | None = None,
) -> str:
    """Save calibration and evaluation splits as JSON files.

    Returns the output directory path.
    """
    if output_dir is None:
        output_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(output_dir, exist_ok=True)

    calib_path = os.path.join(output_dir, "calibration.json")
    eval_path = os.path.join(output_dir, "evaluation.json")

    with open(calib_path, "w") as f:
        json.dump(calibration, f, indent=2)
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2)

    print(f"Saved {len(calibration)} calibration samples → {calib_path}")
    print(f"Saved {len(evaluation)} evaluation samples     → {eval_path}")
    return output_dir


def load_splits(data_dir: str | None = None) -> tuple[list[dict], list[dict]]:
    """Load previously saved splits from disk."""
    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, "processed")
    calib_path = os.path.join(data_dir, "calibration.json")
    eval_path = os.path.join(data_dir, "evaluation.json")

    with open(calib_path) as f:
        calibration = json.load(f)
    with open(eval_path) as f:
        evaluation = json.load(f)

    return calibration, evaluation


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_tool_call_string(raw: str) -> dict | None:
    """Very basic parser for ToolACE-style tool call strings.

    Handles format like:  FunctionName(param1="value1", param2="value2")
    Falls back to returning the raw string if parsing fails.
    """
    raw = raw.strip().strip("[]")
    if not raw:
        return None

    # Try to extract function name and parenthesised arguments
    paren_idx = raw.find("(")
    if paren_idx == -1:
        return {"name": raw, "arguments": {}}

    name = raw[:paren_idx].strip()
    args_str = raw[paren_idx + 1 : raw.rfind(")")] if ")" in raw else raw[paren_idx + 1 :]

    import re

    args: dict[str, str] = {}
    if args_str.strip():
        # Match key="value" or key=value patterns
        pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|([^,)]+))'
        for match in re.finditer(pattern, args_str):
            key = match.group(1)
            value = match.group(2) if match.group(2) is not None else match.group(3).strip()
            args[key] = value

    return {"name": name, "arguments": args}


if __name__ == "__main__":
    # Quick smoke test
    calib, eval_ = load_toolace_splits(5, 3)
    print(f"Calibration: {len(calib)}, Evaluation: {len(eval_)}")
    sample = calib[0]
    print(f"\n=== Formatted prompt ===\n{format_prompt(sample)[:500]}...")
    print(f"\nExpected tool call: {extract_expected_tool_calls(sample)}")
    save_splits(calib, eval_)
