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
> produces coherent output. For real INT4 execution, hand `quantized_state.pt`
> to an INT4-aware runtime (vLLM, TGI, MLX, TensorRT-LLM).

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