# AWQ Quality Benchmarks

Perplexity + generation measured on the EC2 `g6.xlarge` (NVIDIA L4, 24 GB),
PyTorch 2.6.0+cu124, transformers 5.12, AutoAWQ 0.2.9 (real INT4 GEMM runtime).
WikiText-2 **test** split, sliding window (stride 512, max_length 2048), greedy
generation (max_new_tokens 48). Artifacts produced with
`awq run --quantize-strategy all --group-size 128`.

> **Qwen3-7B does not exist** on the Hub; **Qwen3-8B** is used as the
> 7B-class secondary target (closest available, 16 GB FP16, fits the L4).

## Headline results

| Model | FP16 PPL | AWQ-runtime PPL | × FP16 | awq.inference PPL | bar |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B (neg. control) | ~coherent | coherent | — | garbage | below viability |
| **Qwen3-1.7B** | 16.79 | **20.90** | **1.245×** | 264120 (broken) | INVESTIGATE |
| **Qwen3-8B** (7B-class) | 9.75 | **10.08** | **1.034×** | 174464 (broken) | **PASS** |

`AWQ-runtime` = `awq export` → AutoAWQ real-INT4 GEMM. `awq.inference` = the
dequantized-FP16 path (`load_awq_model`). Pass bar: AWQ PPL ≤ 1.10× FP16.

**8B passes cleanly (1.034× FP16)** with coherent greedy generation across all
four prompts. 1.7B is at the INVESTIGATE/edge (1.245×) — smaller models are
less robust to INT4, as expected.

### 8B greedy generation (FP16 vs AWQ-runtime)

| Prompt | FP16 | AWQ-runtime (real INT4) |
| --- | --- | --- |
| "The capital of France is" | Paris, Rome, Madrid, Berlin, Amsterdam… | Paris, Washington DC, London, Berlin, Rome, Madrid… |
| "Explain gradient descent in one sentence:" | correct, coherent | correct, coherent |
| "def fibonacci(n):" | clean recursive fib | clean fib (`if n <= 1`) |
| "Once upon a time in a galaxy far away," | "planet Mathoria, base-12 number system" | "planet Mathoria, Number Navigators" |

AWQ-runtime stays on-task on every prompt; no degeneration.

## Verdict

The three algorithm fixes (scale direction `W·s`/`/s`, per-layer α grid search,
canonical verify) were re-validated on **viable** models on CUDA.

- **The exportable AWQ works and scales with model size.** `awq export` →
  AutoAWQ real-INT4 gives **1.034× FP16 on 8B** (PASS) and 1.245× on 1.7B
  (INVESTIGATE), with coherent generation.
- **`awq.inference` (the dequantized-FP16 path) is broken on viable models.**
  PPL 264120 (1.7B) / 174464 (8B). Root cause below.

### Root cause: `awq.inference` divides weights by `s`, amplifying error

`awq.inference.load_awq_model` injects `Q(W·s)/s` as the linear weight and runs
a normal FP16 forward (no norm folding). For channels where the AWQ scale
`s < 1`, the `1/s` factor **amplifies** the INT4 rounding error, inflating the
weight norm and concentrating error on those channels. Measured on
`model.layers.0.self_attn.q_proj` (Qwen3-1.7B, gs=128):

| dequant path | weight norm (orig = 72.03) | MSE vs FP16 | single-layer output rel. err. |
| --- | --- | --- | --- |
| `awq.inference` (`Q(W·s)/s`) | 79.38 (+10%) | 4.2e-4 | 29–46% |
| export (`Q(W·s)`, `s` folded into norm) | 72.27 (+0.3%) | 2.1e-5 | — |

A 29–46% per-layer output error compounds across 28–36 layers into garbage. The
**norm-fold** approach (store `Q(W·s)`, divide the *activation* by `s` via the
preceding norm — what `awq export` and the reference `mit-han-lab/llm-awq` do)
avoids the `1/s` amplification and is **20× more accurate** per weight.

**Recommendation.** `awq.inference` should fold `s` into the preceding norm
(shared per norm group, as `awq export` does) instead of dividing weights by
`s`. Until then, `awq.inference` is a sanity-check path only; the
`awq export` → AutoAWQ/vLLM path is the correct, runtime-loadable AWQ.

## Grid-search vs fixed-α ablation (Qwen3-1.7B, gs=128)

| scales | verify MSE (weight) | export runtime PPL |
| --- | --- | --- |
| grid search (--n-grid 20) | 0.000274 | 20.90 |
| fixed α=0.5 (--no-grid-search) | 0.000563 | 19.94 |

- **Grid search wins on weight reconstruction** (~2× lower verify MSE) — the
  grid-search fix is real and the scoring function is correctly weighted at the
  per-layer level.
- **Fixed-α wins on end-to-end export PPL** (19.94 vs 20.90). This is not a
  scoring bug: the export re-quantizes with a **shared-per-norm-group** `s`
  (geometric mean of the member linears), so the per-layer grid search's
  aggressive, layer-specific `s` does not propagate through the shared-s
  aggregation. Fixed-α produces smoother per-linear scales that aggregate into
  a better shared `s`.

**Implication.** To make the grid search improve *export* quality, the search
should optimize over **norm-groups** (a shared `s` per `input_layernorm` /
`post_attention_layernorm`) rather than per-linear — matching what the
exportable format actually uses. This is the natural follow-up to #25's
shared-s reconciliation.

## Reproducing

```bash
# 1.7B / 8B
awq run --model /data/models/Qwen3-1.7B --output-dir /data/out-1.7b \
  --dataset wikitext --samples 128 --max-length 2048 --device cuda \
  --quantize-strategy all --group-size 128 --verify-layers 5
awq export --model /data/models/Qwen3-1.7B \
  --from /data/out-1.7b/awq_quantized/quantized_state.pt \
  --to /data/out-1.7b/awq_hf --group-size 128        # add --device cuda for 8B
python eval/ppl.py --model /data/models/Qwen3-1.7B --config fp16
python eval/ppl.py --model /data/models/Qwen3-1.7B --config awq-dequant \
  --quantized-state /data/out-1.7b/awq_quantized/quantized_state.pt
python eval/ppl.py --model /data/models/Qwen3-1.7B --config awq-runtime \
  --export-dir /data/out-1.7b/awq_hf --gen
```