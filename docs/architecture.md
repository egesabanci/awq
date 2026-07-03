# Architecture

Module boundaries, dependency timing, and data flow for `awq`.

## Modules

| Module | Responsibility | Heavy imports? |
| --- | --- | --- |
| `awq/cli.py` | Argparse subcommands (`calibrate`, `scales`, `quantize`, `run`), dispatch, error handling. | No — heavy libs imported lazily inside each `cmd_*`. |
| `awq/__main__.py` | `python -m awq` entry; `sys.exit(main())` so errors propagate. | No. |
| `awq/models.py` | `load_model()` — shared FP16 causal LM + tokenizer loader. | `torch`, `transformers` (imported when `load_model` is called). |
| `awq/calibrate.py` | Forward hooks aggregating `|X|.mean(0)` per linear layer. | `torch`, `transformers` (inside `run_calibration`). |
| `awq/scales.py` | Per-layer AWQ scale computation + α grid search + skip-set builder. | `torch` (inside `compute_awq_scale`). |
| `awq/quantize.py` | `safetensors` streaming, INT4 packing, `dequantize_layer`, `verify_reconstruction`. | `safetensors` (inside `iter_weights`/`load_weight_from_safetensors`). |
| `awq/inference.py` | `load_awq_model` (dequantize + inject linear weights). | `torch`, `transformers`. |
| `utils/memory.py` | Device detection, MPS/CUDA memory limiting, tracking. | `torch`. |
| `utils/errors.py` | Exceptions, `retry_with_backoff`, validation, OOM diagnostics. | `torch` (only inside `require_*` / `diagnose_oom`). |
| `data/natural_calibration.json` | Bundled WikiText-2 calibration samples (loaded by path). | None. |

## Dependency timing

`awq` is import-light: `import awq` / `python -m awq --help` must not import
`torch`, `transformers`, or `datasets`. The CLI dispatch (`cmd_calibrate`,
`cmd_scales`, `cmd_quantize`, `cmd_run`) imports heavy modules **inside** the
function body, so a help print or an early validation failure never pays the
import cost. When adding a command, keep heavy imports inside the `cmd_*`
function, not at module top.

## Data flow

```txt
model.safetensors ─┐
                   ├─ calibrate (forward + hooks) ─► calibration_stats.pt
prompts ───────────┘                                       │
                                                           ▼
model.safetensors ─► scales (grid search over W) ──► awq_scales.pt
                   │                                       │
                   └───────────────────────────────────────►│
                                                           ▼
model.safetensors ─► quantize (W·s, INT4 pack) ───► quantized_state.pt + metadata.json
                                                           │
                                                           ▼
                                                   verify (dequant vs disk) ──► MSE
```

Calibration and scales both load the model (via `awq.models.load_model`);
quantize does **not** — it streams `safetensors` on CPU. `awq run` loads the
model, frees it, then re-loads it for the scales phase.

## Key invariants

- **Single dequant path.** `awq.quantize.dequantize_layer` is the canonical
  dequantizer; `awq.inference` imports it rather than re-implementing. Verify
  and inference therefore measure/incur the same error.
- **AWQ direction.** Quantization scales weights **up** by `s`
  (`w_scaled = W * s`) and dequantization divides by `s` (`W ≈ Q(W·s)/s`).
  This matches the reference implementation
  (`mit-han-lab/llm-awq`, `fc.weight.mul_(scales)` then `/scales`).
- **Memory-safe quantize.** Peak memory during `awq quantize` is one weight
  tensor + its packed output, not the full model.
- **No silent empty output.** `compute_all_scales` raises `ScaleError` if 0
  layers are produced; per-layer failures are surfaced even when `--quiet`.
- **Exit codes propagate.** `__main__.py` does `sys.exit(main())`; `main()`
  catches exceptions and returns 1.