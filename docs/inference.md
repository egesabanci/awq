# Inference

Dequantization and model loading for AWQ-quantized weights — `awq/inference.py`.

## `dequantize_layer(q)`

The canonical dequantizer (also imported by `awq.quantize.verify_reconstruction`,
so verify and inference share one path). For each group:

```
low  = packed & 0x0F;        high = (packed >> 4) & 0x0F
low  = low > 7  ? low - 16  : low      # two's-complement sign fix
high = high > 7 ? high - 16 : high
w_group = stack([low, high]) * group_scale      # ≈ (W·s) for that group
```

Concatenate groups, slice to `d_in`, then **divide by the AWQ scale**:
`Ŵ = w / s`. Result: an FP16 `[d_out, d_in]` weight approximating the original.

## `load_awq_model(quantized_path, model_path, device)`

1. Load the FP16 model shell via the shared `awq.models.load_model`.
2. `torch.load` the quantized state (CPU, `weights_only=True`).
3. For each quantized layer, `dequantize_layer` → overwrite the matching
   `nn.Linear.weight.data` with the dequantized FP16 weight.
4. `model.eval()`.

> **No INT4 kernel.** This is dequantized-FP16 inference: the model runs a
> normal FP16 forward with weights that happen to be INT4-derived. There is
> **no** inference speed or memory benefit — peak memory is ~one full FP16
> model. Use this path to sanity-check that the quantized artifact still
> produces coherent output. For real INT4 execution, **export** the artifact
> (see below) and hand the exported directory to AutoAWQ or vLLM.

## Exporting to a runtime-loadable INT4 model (`awq export`)

`quantized_state.pt` is a custom layout no external runtime can consume.
`awq export` re-packs it into the **AutoAWQ GEMM on-disk format**
(`qweight`/`qzeros`/`scales` + `quantize_config.json`) that
`AutoAWQForCausalLM` and vLLM's `awq` loader read directly:

```bash
awq export --model Qwen/Qwen3-1.7B \
  --from out-1.7b/awq_quantized/quantized_state.pt --to out-1.7b/awq_hf \
  --group-size 128
```

### Scale reconciliation (the hard part)

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

### Loading the exported model

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
> `eval/ppl.py` does. This is why `awq export` does **not** depend on
> AutoAWQ at runtime: the packing is reimplemented in pure torch inside
> `awq/export.py` and unit-tested against AutoAWQ's documented GEMM dequant.

## Generating from the quantized model (library use)

```python
from awq.models import load_model
from awq.inference import load_awq_model

# FP16 baseline
model, tok = load_model("Qwen/Qwen3-0.6B", "mps")
# ... model.generate(...) ...

# AWQ (dequantized-FP16)
qmodel, qtok, qstate = load_awq_model("out/awq/quantized_state.pt",
                                      "Qwen/Qwen3-0.6B", "mps")
# ... qmodel.generate(...) ...
```

Remember: throughput will be ~FP16 (no kernel), so this is for quality
inspection, not deployment.