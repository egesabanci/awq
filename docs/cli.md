# CLI

Command-line reference for `awq` (also invokable as `python -m awq`).

```bash
awq --help
awq <command> [options]
```

## Subcommands

| Command | Purpose |
| --- | --- |
| `calibrate` | Forward calibration samples; collect per-channel activation magnitudes. |
| `scales` | Compute per-layer AWQ scales (α grid search by default). |
| `quantize` | Pack linear weights to group-wise INT4; verify reconstruction. |
| `run` | Execute calibrate → scales → quantize (+ verify) in one process. |

There is no `benchmark`/eval command — quality comparison is out of scope for
this CLI.

## Global options

| Option | Default | Description |
| --- | --- | --- |
| `--model` | required | Local model directory or Hugging Face repo ID. |
| `--device` | auto | `cuda` / `mps` / `cpu`; auto-detected if omitted. Passing `--device cuda` on a non-CUDA host falls back to MPS/CPU with a warning. |
| `--quiet` | off | Suppress per-layer progress output. Errors and failures are still surfaced. |

## `awq calibrate`

| Option | Default | Description |
| --- | --- | --- |
| `--dataset` | `wikitext` | Calibration dataset (bundled WikiText-2 samples; falls back to HF `Salesforce/wikitext` if the bundle is absent). |
| `--output` | `results/calibration_stats.pt` | Output path for `calibration_stats.pt`. |
| `--samples` | `128` | Number of calibration samples. |
| `--batch-size` | `5` | Cache-clear cadence (see [calibration.md](calibration.md) — there is no real batching). |
| `--max-length` | `2048` | Max sequence length per sample (truncate). |

On MPS, `limit_memory(0.7, "mps")` is applied before the model loads.

## `awq scales`

| Option | Default | Description |
| --- | --- | --- |
| `--calibration-stats` | required | Path to `calibration_stats.pt`. |
| `--output` | `results/awq_scales.pt` | Output path for scales. |
| `--group-size` | `32` | INT4 group size; must match `awq quantize`. |
| `--alpha` | `0.5` | Fixed exponent, used only with `--no-grid-search`. |
| `--no-grid-search` | off | Use fixed `--alpha` instead of the per-layer α search. |
| `--n-grid` | `20` | Number of α candidates for the grid search (per AWQ paper). |
| `--quantize-strategy` | `alternating` | `all` / `alternating` / `last_only` / `first_only` (layer count derived from stats). |
| `--no-skip-lm-head` | off | Include `lm_head` in quantization (not recommended). |
| `--device` | auto | Device for the model load (weights read for grid search). |

If 0 scales are produced (all layers skipped/missing/failed), the command
raises `ScaleError` and exits non-zero.

## `awq quantize`

| Option | Default | Description |
| --- | --- | --- |
| `--scales` | required | Path to `awq_scales.pt`. |
| `--output-dir` | `results/awq_quantized` | Output directory. |
| `--group-size` | `32` | INT4 group size (must match scales). |
| `--verify-layers` | `3` | Layers to dequantize and compare vs disk. `0` skips. |

Quantization runs on CPU, reading `safetensors` one tensor at a time.

## `awq run`

Runs the three phases with a single model load freed between phases. Accepts
the union of `calibrate` + `scales` + `quantize` options: `--model`,
`--dataset`, `--output-dir`, `--samples`, `--batch-size`, `--max-length`,
`--group-size`, `--alpha`, `--no-grid-search`, `--n-grid`,
`--quantize-strategy`, `--device`, `--quiet`.

## Failure behavior

- Validation errors (`utils.errors.ValidationError`) print a message and exit 1.
- OOM triggers `utils.errors.diagnose_oom` which prints device memory stats.
- `KeyboardInterrupt` exits 1 cleanly.
- A 0-layer scales result raises `ScaleError` (never a silent success).

## Examples

```bash
# Full pipeline, model-agnostic, all layers quantized
awq run --model Qwen/Qwen3-0.6B --dataset wikitext \
  --output-dir out --quantize-strategy all --samples 32 --max-length 512

# Step-by-step
awq calibrate --model Qwen/Qwen3-0.6B --dataset wikitext \
  --output out/stats.pt --samples 128
awq scales --model Qwen/Qwen3-0.6B --calibration-stats out/stats.pt \
  --output out/scales.pt --group-size 32 --quantize-strategy all
awq quantize --model Qwen/Qwen3-0.6B --scales out/scales.pt \
  --output-dir out/awq --group-size 32 --verify-layers 3
```