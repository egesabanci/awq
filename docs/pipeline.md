# Pipeline

The `awq` pipeline is four linear phases. Each phase is independently runnable
and produces a discrete artifact on disk.

```txt
                 calibrate                scales                  quantize
FP16 model  ────────────────►  calibration_stats.pt  ────────────────►  awq_scales.pt  ────────────────►  awq_quantized/
 + prompts    (forward + hooks)              (1D |X|.mean(0)            (per-channel s          (packed INT4 +
              per linear layer)               per linear layer)          [d_in] per layer)         group scales +
                                                                                                  AWQ scales)
                                                                                                      │
                                                                                                      ▼
                                                                                              verify (MSE)
```

## Phases

### 1. Calibrate (`awq.calibrate`)
- Forward calibration prompts through the FP16 model with forward hooks on
  every `nn.Linear`.
- Each hook aggregates `|activation|.mean(0)` **on-the-fly** as a running sum,
  so memory is `O(d_in)` per layer, not `O(samples × tokens × d_in)`.
- Output: `calibration_stats.pt` — `{layer_name: 1D tensor of d_in}`.

### 2. Scales (`awq.scales`)
- For each linear layer, grid-search α ∈ [0, 1) and pick the scale
  `s = (x_max^α) / (mean(x_max)^α)` minimizing activation-weighted
  reconstruction error of `Q(W·s)/s` (the correct AWQ direction: weights scaled
  **up** by `s` before quantization, divided by `s` after).
- `α = 0` is plain RTN (`s = 1`), so the search never does worse than RTN.
- Output: `awq_scales.pt` — `{layer_name: s of shape [d_in]}`.

### 3. Quantize (`awq.quantize`)
- Stream each weight tensor from `safetensors` one at a time (CPU, memory-safe).
- Apply `W' = W·s`, group-wise INT4 quantize (`qscale = max|group|/7`, round,
  clamp to [-7, 7]), pack two INT4 per byte.
- Store packed weights + per-group FP16 scales + per-channel AWQ scales `s`.
- Output: `awq_quantized/quantized_state.pt` + `metadata.json`.

### 4. Verify (`awq.quantize.verify_reconstruction`)
- Dequantize a few layers via the **canonical** `dequantize_layer` path
  (so verification reports the real per-layer quantization MSE).
- Report MSE against the original FP16 weights on disk.

## Full pipeline (`awq run`)

`awq run` executes calibrate → scales → quantize (+ verify) in one process,
freeing the model and emptying the cache between phases. There is no
benchmark/evaluation phase — quality comparison is out of scope for this CLI.

## Invariants

- The verification dequant (`awq.quantize.dequantize_layer`) is used only by
  `verify_reconstruction`. Running the model is done via `awq export` + a real
  INT4 runtime, so there is no dequant-to-FP16 inference path to diverge from.
- Quantization never loads the full model into device memory; it reads
  `safetensors` tensor-by-tensor on CPU.
- A scales run that produces 0 layers raises `ScaleError` (it never silently
  writes an empty scales file).
- CLI errors propagate a non-zero exit code (`awq/__main__.py` does
  `sys.exit(main())`).