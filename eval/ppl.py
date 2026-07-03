#!/usr/bin/env python3
"""AWQ quality measurement: perplexity + greedy generation.

Standalone analysis tool (NOT a CLI subcommand). Compares three configs of a
model against the wikitext-2 test split:

  fp16         — the FP16 reference baseline
  awq-dequant  — awq.inference.load_awq_model (dequantized-FP16; our quality)
  awq-runtime  — AutoAWQ real INT4 GEMM on the exported HF-AWQ model (#25)

Perplexity uses a sliding window (stride, max_length) over the concatenated
test set and reports exp(mean cross-entropy over non-padded tokens).

Usage:
  python eval/ppl.py --model /data/models/Qwen3-1.7B --config fp16
  python eval/ppl.py --model /data/models/Qwen3-1.7B --config awq-dequant \
      --quantized-state /data/out-1.7b/awq_quantized/quantized_state.pt
  python eval/ppl.py --model /data/models/Qwen3-1.7B --config awq-runtime \
      --export-dir /data/out-1.7b/awq_hf --gen
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Make this repo `awq` package win over the AutoAWQ same-named package for the
# fp16 / awq-dequant configs. The awq-runtime config reverses this.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Force this repo to the FRONT of sys.path even if the editable .pth already
# added it at the end (otherwise AutoAWQ site-packages/awq shadows it).
while _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)
_AUTOAWQ_SITE = "/data/.local/lib/python3.10/site-packages"
# Eagerly cache THIS repo's `awq` before transformers/datasets can pull in
# AutoAWQ's same-named package and shadow it in sys.modules.
import awq as _our_awq  # noqa: F401  (ours; cheap empty __init__)
_ = _our_awq

GEN_PROMPTS = [
    "The capital of France is",
    "Explain gradient descent in one sentence:",
    "def fibonacci(n):\n    ",
    "Once upon a time in a galaxy far away,",
]


def _load_fp16(model_path: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16,
        device_map="auto" if device == "cuda" else device,
        low_cpu_mem_usage=True, trust_remote_code=False,
    )
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model.eval()
    return model, tok


def _load_awq_dequant(quantized_state: str, model_path: str, device: str):
    from awq.inference import load_awq_model
    return load_awq_model(quantized_state, model_path, device)


def _load_awq_runtime(export_dir: str, device: str):
    """Load the exported HF-AWQ model with AutoAWQ's real INT4 GEMM runtime.

    AutoAWQ ships a top-level `awq` package that collides with this repo's
    `awq` import name. We bypass our editable-install finder so `import awq`
    resolves to AutoAWQ, then restore it after loading.
    """
    import torch
    from transformers import AutoTokenizer
    orig_meta = sys.meta_path
    # Purge our cached `awq` so AutoAWQ's same-named package can be imported.
    for k in [k for k in list(sys.modules) if k == "awq" or k.startswith("awq.")]:
        del sys.modules[k]
    sys.meta_path = [f for f in sys.meta_path if "editable" not in type(f).__name__.lower()]
    sys.path.insert(0, _AUTOAWQ_SITE)
    try:
        from awq import AutoAWQForCausalLM
        m = AutoAWQForCausalLM.from_quantized(export_dir, device_map="auto")
    finally:
        sys.meta_path = orig_meta
    tok = AutoTokenizer.from_pretrained(export_dir)
    m.model.eval()
    return m.model, tok


def perplexity(model, tok, texts, device: str, stride: int, maxlen: int) -> float:
    import torch
    enc = tok("\n\n".join(texts), return_tensors="pt")
    seq = enc.input_ids.to(next(model.parameters()).device)
    n = seq.size(1)
    losses, ntoks = [], 0
    prev_end = 0
    for begin in range(0, n, stride):
        end = min(begin + maxlen, n)
        trg = seq[:, begin:end]
        with torch.no_grad():
            out = model(trg, labels=trg)
        # out.loss is mean over (end-begin-1) tokens
        ntok = max(1, trg.size(1) - 1)
        losses.append(out.loss.item() * ntok)
        ntoks += ntok
        if end == n:
            break
    mean_loss = sum(losses) / max(1, ntoks)
    return math.exp(mean_loss)


def greedy_gen(model, tok, device: str, max_new: int = 64) -> dict[str, str]:
    import torch
    out = {}
    for p in GEN_PROMPTS:
        inp = tok(p, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        out[p] = tok.decode(o[0], skip_special_tokens=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="AWQ perplexity + generation measurement")
    ap.add_argument("--model", required=True, help="FP16 model path (for fp16 / awq-dequant)")
    ap.add_argument("--config", required=True,
                    choices=["fp16", "awq-dequant", "awq-runtime"],
                    help="Which model config to evaluate")
    ap.add_argument("--quantized-state", default=None, help="quantized_state.pt (for awq-dequant)")
    ap.add_argument("--export-dir", default=None, help="exported HF-AWQ dir (for awq-runtime)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--gen", action="store_true", help="Also run greedy generation on the 4-prompt set")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    if args.config == "fp16":
        model, tok = _load_fp16(args.model, args.device)
    elif args.config == "awq-dequant":
        assert args.quantized_state, "--quantized-state required for awq-dequant"
        model, tok, _ = _load_awq_dequant(args.quantized_state, args.model, args.device)
    elif args.config == "awq-runtime":
        assert args.export_dir, "--export-dir required for awq-runtime"
        model, tok = _load_awq_runtime(args.export_dir, args.device)

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if t.strip()]

    ppl = perplexity(model, tok, texts, args.device, args.stride, args.max_length)
    print(f"[{args.config}] wikitext-2 test PPL = {ppl:.4f}")

    if args.gen:
        print("\n--- Greedy generation ---")
        for p, o in greedy_gen(model, tok, args.device, args.max_new).items():
            print(f"\n# PROMPT: {p!r}\n{o}")


if __name__ == "__main__":
    main()