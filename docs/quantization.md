# Quantization

AWQ scale computation, skip strategies, INT4 packing, and reconstruction
verification — `awq/scales.py` and `awq/quantize.py`.

## AWQ scale (per layer)

`compute_awq_scale(weight, channel_importance, ...)` computes a per-input-channel
scale `s` of shape `[d_in]`.

**Candidate form** (activation-based, matching `mit-han-lab/llm-awq`
`_search_module_scale`):

```
s = (x_max ^ α) / (mean(x_max) ^ α)        # mean-normalised; absolute scale
                                           # is invariant under the group quantizer
```

**Grid search** (default, `grid_search=True`): iterate α over `n_grid=20`
values in `[0, 1)` and score each candidate by activation-weighted
reconstruction error of `Q(W·s)/s`:

```
err(s) = || (W - Q(W·s)/s) · diag(x_max) ||²      # ≈ output error ||(W - Ŵ)·x||²
```

`α = 0` gives `s = 1` (plain RTN), so the search can only do as well as RTN,
never worse. The weight is actually quantized during the search
(`_pseudo_quantize_int4` mirrors `quantize_layer_cpu`'s grid exactly), so the
search reflects the real quantizer.

`--no-grid-search` uses the fixed `--alpha` instead (closed form, weight not
quantized during scale selection).

## Direction (correct AWQ)

Quantize scales weights **up** by `s` before quantization; dequantize divides
by `s`:

```
quantize:   w_scaled = W * s          # salient channels occupy more of the INT4 range
dequantize: Ŵ = Q(W·s) / s ≈ W         # activation scaling is folded into the weight
```

This matches the reference implementation (`fc.weight.mul_(scales)` then
`/scales`). The activation-side `1/s` is folded into the stored weight, so the
dequantized weight is just an approximation of `W` — no activation hooks needed
at inference.

`_pseudo_quantize_int4` (used by the search) and `quantize_layer_cpu` (used
for real packing) share the same grid: `qscale = max|group| / 7`, round,
clamp to `[-7, 7]`.

## Skip strategies (`build_skip_set`)

Model-agnostic. Layer indices are derived from the calibration stats via
`re.search(r"model\.layers\.(\d+)\.", key)`, so depth-dependent strategies
adapt to any model:

| Strategy | Behavior |
| --- | --- |
| `all` | Quantize every linear except explicit skips. |
| `alternating` | Keep even-numbered layers in FP16. |
| `last_only` | Quantize the second half of the layers. |
| `first_only` | Quantize the first half. |

Explicit skips:
- `lm_head` (and `model.lm_head`, `model.model.lm_head`) — vocabulary projection
  is too sensitive; skipped unless `--no-skip-lm-head`.
- Qwen3.5-style hybrid-attention tiny projections
  (`model.layers.{i}.linear_attn.in_proj_{a,b,z}`) — skipped by name when
  present; on standard-attention models they match nothing.

## INT4 packing (`quantize_layer_cpu`)

For each group of `group_size` input channels:

1. `w_group = (W·s)[:, g:g+group_size]` (zero-pad the final group to `group_size`).
2. `qscale = max|w_group| / 7.0` (one FP16 scalar per group).
3. `w_int4 = round(w_group / qscale).clamp(-7, 7)` (signed, [-7, 7] — 15 codes).
4. Pack two INT4 into one uint8 via two's-complement low-nibble convention:
   `packed = (low & 0x0F) | ((high & 0x0F) << 4)`.

Stored per layer: `packed_weights` (list of uint8 tensors), `group_scales`
(list of FP16 scalars), `scale_factors` (the AWQ `s`, FP16 `[d_in]`),
`shape`, `group_size`.

## Reconstruction verification (`verify_reconstruction`)

Dequantizes `--verify-layers` layers via the **canonical `dequantize_layer`**
and reports MSE against the original weight read from `safetensors`. This is a
reconstruction-quality check, not an inference path — running the model is done
via `awq export` + a real INT4 runtime (which folds `s` into the norm).

## Compression (`metadata.json`)

The compression ratio counts packed INT4 weights **plus** per-group FP16
scales **plus** per-channel AWQ scales, so it reflects the true on-disk
footprint. For `group_size=32`, group scales are one FP16 scalar per 32
weights, so metadata overhead is small and the effective ratio is ~4× over
FP16 linear weights. Quantization itself runs on CPU, streaming
`safetensors` tensor-by-tensor; peak memory is one tensor + its packed
output.