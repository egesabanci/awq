# Inference — running a quantized model

`awq`'s native artifact (`awq_quantized/quantized_state.pt`) is a custom INT4
layout no external runtime can consume. To **run** the quantized model, use
`awq export` to re-pack it into the **AutoAWQ / HF-AWQ GEMM on-disk format**
that real INT4 runtimes (AutoAWQ, vLLM) load directly.

## `awq export`

```bash
awq export --model Qwen/Qwen3-1.7B \
  --from out-1.7b/awq_quantized/quantized_state.pt --to out-1.7b/awq_hf \
  --group-size 128            # add --device cuda for large models
```

Produces a directory with AutoAWQ GEMM `qweight`/`qzeros`/`scales` (int32/fp16)
per quantized linear, a folded-`1/s` norm, `quantize_config.json`, and the
copied config/tokenizer. See [cli.md](cli.md) and [quantization.md](quantization.md).

## Scale reconciliation (the hard part)

Our pipeline computes an *independent* per-channel AWQ scale `s` for every
linear. A loadable HF-AWQ model folds `1/s` into the **preceding norm**'s
weight, but `q`/`k`/`v` share `input_layernorm` and `gate`/`up` share
`post_attention_layernorm` — three/five linears cannot each fold a different
`s` into one norm. The exporter therefore:

1. **Aggregates** the per-linear `s` into one shared scale per norm group
   (geometric mean over the member linears) and folds `1/s` into that norm.
2. **Re-quantizes** the 5 norm-following linears (`q`,`k`,`v`,`gate`,`up`)
   with the shared `s` into the AutoAWQ unsigned-int4 grid.
3. `o_proj` and `down_proj` have **no preceding norm**, so they are exported
   as plain RTN INT4 (`s = 1`).

The signed `[-7, 7]` grid is mapped onto AutoAWQ's unsigned `[0, 15]` grid by
storing `zero = 7` and `q_unsigned = q_signed + 7`, so the runtime's
`(q - zero) * scale` reproduces our group dequant exactly — **no
re-quantization error** on the AWQ-scaled linears.

## Loading the exported model

```python
# AutoAWQ real INT4 GEMM runtime (torch 2.6 compatible)
from awq import AutoAWQForCausalLM            # the AutoAWQ package, not this repo
m = AutoAWQForCausalLM.from_quantized("out-1.7b/awq_hf", device_map="auto")
# ... m.model.generate(...) ...

# or vLLM (requires vLLM's torch pin):
# from vllm import LLM; llm = LLM(model="out-1.7b/awq_hf", quantization="awq")
```

> **Name collision.** AutoAWQ ships a top-level `awq` package that collides
> with this repo's `awq` import name — you cannot `import` both at once. Run
> AutoAWQ in a separate process (or bypass the editable finder) as
> `eval/ppl.py` does. This is why `awq export` does **not** depend on AutoAWQ
> at runtime: the packing is reimplemented in pure torch inside `awq/export.py`
> and unit-tested against AutoAWQ's documented GEMM dequant.

> **No dequant-to-FP16 inference path.** An earlier `awq.inference` path
> dequantized weights back to FP16 and ran a plain forward (a MacBook/MPS
> workaround for the lack of INT4 kernels there). It amplified INT4 error on
> small-`s` channels and produced garbage on viable models (see
> [benchmarks.md](benchmarks.md)), so it was removed. Use `awq export` + a
> real INT4 runtime instead.

## Quality

See [benchmarks.md](benchmarks.md): the exported model hits **1.034× FP16
perplexity on Qwen3-8B** with coherent greedy generation, loaded in AutoAWQ's
real INT4 GEMM runtime.