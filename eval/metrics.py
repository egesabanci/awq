"""Evaluation metrics for tool-calling AWQ experiments.

All metrics compare FP16 baseline outputs against AWQ-quantized outputs
on structured function-calling tasks.
"""

import json
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import torch


# ── Individual Metrics ────────────────────────────────────────────────


def check_json_validity(text: str) -> bool:
    """Check if the generated output contains valid JSON."""
    # Look for JSON-like structures in the output
    json_candidates = re.findall(r"\{[^{}]*\}", text)
    if not json_candidates:
        return False
    for candidate in json_candidates:
        try:
            json.loads(candidate)
            return True
        except json.JSONDecodeError:
            continue
    return False


def extract_tool_call(text: str) -> dict | None:
    """Extract the first valid function call from generated text.

    Tries multiple formats:
    1. JSON object like {"name": "func", "arguments": {...}}
    2. Function call syntax like FuncName(arg1="val1")
    3. Simple function name match against known patterns
    """
    # Format 1: JSON tool call
    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
    for match in re.finditer(json_pattern, text):
        try:
            parsed = json.loads(match.group())
            if "name" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # Format 2: FuncName(...) pattern (ToolACE format)
    func_pattern = r'(\w+)\s*\(([^)]*)\)'
    for match in re.finditer(func_pattern, text):
        name = match.group(1)
        args_str = match.group(2)
        args: dict[str, str] = {}
        if args_str.strip():
            arg_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|([^,)]+))'
            for arg_match in re.finditer(arg_pattern, args_str):
                key = arg_match.group(1)
                value = arg_match.group(2) if arg_match.group(2) else arg_match.group(3).strip()
                args[key] = value
        return {"name": name, "arguments": args}

    # Format 3: bare function name
    for word in re.findall(r'\b\w+\b', text):
        if word[0].isupper() and len(word) > 2:
            return {"name": word, "arguments": {}}

    return None


def check_function_name(expected_name: str, predicted: str) -> bool:
    """Check if the predicted output contains the correct function name."""
    extracted = extract_tool_call(predicted)
    if extracted is None:
        return False
    return extracted["name"].strip().lower() == expected_name.strip().lower()


def check_required_params(expected: dict, predicted: str) -> float:
    """Check what fraction of required params are present.

    Returns a score between 0.0 (none present) and 1.0 (all present).
    If no params are expected, returns 1.0.
    """
    expected_args = expected.get("arguments", {})
    if not expected_args:
        return 1.0

    extracted = extract_tool_call(predicted)
    if extracted is None:
        return 0.0

    predicted_args = extracted.get("arguments", {})
    required_keys = set(expected_args.keys())
    if not required_keys:
        return 1.0

    present = sum(1 for k in required_keys if k in predicted_args)
    return present / len(required_keys)


def check_param_values(expected: dict, predicted: str) -> float:
    """Check what fraction of parameter values match exactly.

    Returns a score between 0.0 and 1.0.
    """
    extracted = extract_tool_call(predicted)
    if extracted is None:
        return 0.0

    expected_args = expected.get("arguments", {})
    predicted_args = extracted.get("arguments", {})

    if not expected_args:
        return 1.0

    matches = 0
    for k, v in expected_args.items():
        if k in predicted_args:
            pred_v = predicted_args[k]
            # Normalize both to strings for comparison
            if str(pred_v).strip().lower() == str(v).strip().lower():
                matches += 1
            # Also try without quotes
            elif str(pred_v).strip().strip('"').lower() == str(v).strip().strip('"').lower():
                matches += 1

    return matches / len(expected_args)


# ── Semantic Similarity ───────────────────────────────────────────────


def get_embeddings(
    texts: list[str],
    model: torch.nn.Module | None = None,
    tokenizer: Any = None,
    device: str = "mps",
) -> torch.Tensor:
    """Extract embeddings for a list of texts using the model's hidden states.

    If model is None, falls back to simple bag-of-words embedding for
    offline semantic comparison (no extra model needed).
    """
    if model is not None and tokenizer is not None:
        return _get_model_embeddings(texts, model, tokenizer, device)

    return _get_bow_embeddings(texts)


def _get_model_embeddings(
    texts: list[str],
    model: torch.nn.Module,
    tokenizer: Any,
    device: str = "mps",
) -> torch.Tensor:
    """Extract embeddings from the last hidden state of the model."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            # Use mean of last hidden state as sentence embedding
            last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden]
            emb = last_hidden.mean(dim=1).cpu()  # [1, hidden]
            embeddings.append(emb)
    return torch.cat(embeddings, dim=0)


def _get_bow_embeddings(texts: list[str], vocab_size: int = 5000) -> torch.Tensor:
    """Simple bag-of-words embedding fallback.

    Creates a fixed-size vector based on word frequencies.
    Good enough for relative similarity comparisons.
    """
    embeddings = []
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        counts: dict[str, float] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1.0

        # Sort by frequency, take top-k
        sorted_words = sorted(counts.items(), key=lambda x: -x[1])[:vocab_size]
        emb = torch.zeros(vocab_size)
        for i, (_, freq) in enumerate(sorted_words):
            emb[i] = freq
        embeddings.append(emb)

    return torch.stack(embeddings)


def semantic_similarity(
    texts_a: list[str],
    texts_b: list[str],
    model: torch.nn.Module | None = None,
    tokenizer: Any = None,
    device: str = "mps",
) -> dict[str, float]:
    """Compute semantic similarity between paired texts.

    Returns mean, median, std, and min cosine similarity.
    """
    assert len(texts_a) == len(texts_b), "Lists must have same length"

    emb_a = get_embeddings(texts_a, model, tokenizer, device)
    emb_b = get_embeddings(texts_b, model, tokenizer, device)

    # Normalise
    emb_a = emb_a / emb_a.norm(dim=1, keepdim=True).clamp(min=1e-8)
    emb_b = emb_b / emb_b.norm(dim=1, keepdim=True).clamp(min=1e-8)

    cosines = (emb_a * emb_b).sum(dim=1).numpy()

    return {
        "mean": float(np.mean(cosines)),
        "median": float(np.median(cosines)),
        "std": float(np.std(cosines)),
        "min": float(np.min(cosines)),
        "max": float(np.max(cosines)),
    }


# ── Aggregation ────────────────────────────────────────────────────────


def evaluate_sample(expected: dict, predicted: str) -> dict:
    """Compute all metrics for a single sample.

    Args:
        expected: Ground-truth tool call dict with 'name' and 'arguments'.
        predicted: Raw text output from the model.

    Returns:
        Dict of metric_name → score for this sample.
    """
    return {
        "json_valid": float(check_json_validity(predicted)),
        "function_name_correct": float(check_function_name(expected.get("name", ""), predicted)),
        "param_presence": check_required_params(expected, predicted),
        "param_accuracy": check_param_values(expected, predicted),
        "has_tool_call": float(extract_tool_call(predicted) is not None),
    }


def run_full_evaluation(
    fp16_outputs: list[str],
    awq_outputs: list[str],
    expected: list[list[dict]],
) -> dict:
    """Compare FP16 vs AWQ outputs across all metrics.

    expected is a list of lists per sample (multiple possible tool calls).
    We evaluate against the first expected call per sample for simplicity.

    Returns a dict with aggregate comparison table.
    """
    assert len(fp16_outputs) == len(awq_outputs) == len(expected), "All lists must match in length"

    # Flatten: take first expected tool call per sample if available
    flat_expected = []
    for exp_list in expected:
        if exp_list and len(exp_list) > 0:
            flat_expected.append(exp_list[0])
        else:
            flat_expected.append({"name": "", "arguments": {}})

    fp16_scores: list[dict] = []
    awq_scores: list[dict] = []

    for exp, fp16_out, awq_out in zip(flat_expected, fp16_outputs, awq_outputs):
        fp16_scores.append(evaluate_sample(exp, fp16_out))
        awq_scores.append(evaluate_sample(exp, awq_out))

    # Aggregate
    metrics = ["json_valid", "function_name_correct", "param_presence", "param_accuracy", "has_tool_call"]
    table: dict[str, dict] = {}
    for m in metrics:
        fp16_vals = [s[m] for s in fp16_scores]
        awq_vals = [s[m] for s in awq_scores]
        table[m] = {
            "fp16_mean": float(np.mean(fp16_vals)),
            "awq_mean": float(np.mean(awq_vals)),
            "delta": float(np.mean(awq_vals) - np.mean(fp16_vals)),
            "fp16_std": float(np.std(fp16_vals)),
            "awq_std": float(np.std(awq_vals)),
        }

    # Semantic similarity between FP16 and AWQ outputs
    sem = semantic_similarity(fp16_outputs, awq_outputs)

    return {
        "n_samples": len(expected),
        "per_metric": table,
        "semantic_similarity": sem,
    }


# ── Perplexity (diagnostic) ────────────────────────────────────────────


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    device: str = "mps",
    max_length: int = 512,
) -> float:
    """Compute perplexity over a list of texts.

    Useful as a diagnostic measure of overall model quality degradation.
    Lower is better.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            n_tokens = (inputs["input_ids"] != tokenizer.pad_token_id).sum().item()
            if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
                n_tokens = (inputs["input_ids"] != tokenizer.pad_token_id).sum().item()
            else:
                n_tokens = inputs["input_ids"].size(1)

            total_loss += loss * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    return float(np.exp(avg_loss))
