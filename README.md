<div align="center">

# awq

**Model-agnostic Activation-aware Weight Quantization (AWQ) for causal LMs.**

[Quick Start](#quick-start) |
[Workflow](#workflow) |
[CLI Reference](#cli-reference) |
[Artifacts](#artifacts) |
[Technical Docs](#technical-docs) |
[Development](#development) |
[License](#license)

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Runtime](<https://img.shields.io/badge/Runtime-PyTorch%20(MPS%20%7C%20CUDA%20%7C%20CPU)-orange>)
![Package](https://img.shields.io/badge/Package-awq-green)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

`awq` is a from-scratch, pure-PyTorch implementation of **Activation-aware
Weight Quantization** (Lin et al., 2023) exposed as a small CLI. It quantizes
the linear layers of any Hugging Face causal LM to group-wise INT4 with
per-channel AWQ scaling, and verifies the result by reconstruction.

Use it when you want to study or produce an AWQ-quantized weight artifact
**without** depending on AutoAWQ / AutoGPTQ. The implementation is
model-agnostic: it derives the transformer layer layout from the model rather
than hardcoding layer counts, and it reads weights one tensor at a time from
`safetensors` so the quantizer itself stays memory-safe.

> **Inference note.** `awq` produces a quantized weight artifact and verifies
> its reconstruction quality. Loading that artifact (via `awq.inference`)
> dequantizes weights back to FP16 and runs a standard forward — there is no
> INT4 kernel, so there is no speed/memory benefit at inference time on its
> own. The artifact is what you'd hand to an INT4-aware runtime (vLLM, TGI,
> MLX, TensorRT-LLM, …) for real INT4 execution. See
> [docs/inference.md](docs/inference.md).

## Highlights

- **Pure PyTorch AWQ** — no external quantization libraries.
- **Model-agnostic** — works on any HF causal LM whose linears are named
  `model.layers.{i}.*`; layer count and skip sets are derived from the model.
- **Per-layer α grid search** — the proper AWQ scale search (not a fixed
  exponent): each layer picks the α that minimizes activation-weighted
  reconstruction error of `Q(W·s)/s`.
- **Memory-safe quantizer** — streams `safetensors` one tensor at a time;
  quantization runs on CPU and never loads the full model.
- **Reconstruction verification** — `awq quantize` dequantizes a few layers and
  reports MSE against the original weights, using the exact dequant path
  inference uses.
- **Device-agnostic** — auto-detects CUDA, MPS, or CPU; MPS memory is capped
  via `torch.mps.set_per_process_memory_fraction`.
- **Import-light CLI** — `torch` / `transformers` are only imported inside the
  subcommand that needs them; `awq --help` is instant.

## Quick Start

Prerequisites:

- macOS (Apple Silicon), Linux with CUDA, or CPU-only
- Python 3.11+
- PyTorch 2.4+
- A local FP16 model in `safetensors` (or a Hugging Face repo ID)

Install:

```bash
git clone git@github.com:egesabanci/awq.git
cd awq
pip install -e .
```

Check the CLI:

```bash
python -m awq --help
```

## Workflow

The pipeline is intentionally linear and inspectable.

| Step | What happens | Artifact |
| --- | --- | --- |
| 1. Calibrate | Forward calibration samples through the FP16 model; aggregate per-input-channel activation magnitude `\|X\|.mean(0)` on-the-fly via hooks. | `calibration_stats.pt` |
| 2. Scales | For each linear layer, grid-search α and compute per-channel AWQ scales `s = (x_max^α) / (mean^α)`, scored by activation-weighted reconstruction of `Q(W·s)/s`. | `awq_scales.pt` |
| 3. Quantize | Read each weight tensor from `safetensors`, scale `W·s`, pack to group-wise INT4 (two INT4 per byte), store group scales + AWQ scales. | `awq_quantized/quantized_state.pt`, `metadata.json` |
| 4. Verify | Dequantize a few layers (same path as inference) and report MSE vs. the original weights. | printed MSE |

Run all steps with one command:

```bash
awq run --model Qwen/Qwen3-0.6B --dataset wikitext --output-dir ./out
```

Or run steps individually:

```bash
awq calibrate --model Qwen/Qwen3-0.6B --dataset wikitext --output out/stats.pt
awq scales     --model Qwen/Qwen3-0.6B --calibration-stats out/stats.pt --output out/scales.pt
awq quantize   --model Qwen/Qwen3-0.6B --scales out/scales.pt --output-dir out/awq
```

## CLI Reference

```bash
python -m awq --help
```

### Global options

| Option | Default | Description |
| --- | --- | --- |
| `--model` | required | Model path or Hugging Face repo ID. |
| `--device` | auto | `cuda`, `mps`, or `cpu`. Auto-detected if omitted. |
| `--quiet` | off | Suppress progress output. |

### `awq calibrate`

| Option | Default | Description |
| --- | --- | --- |
| `--dataset` | `wikitext` | Calibration dataset (`wikitext`, `c4`). |
| `--output` | `results/calibration_stats.pt` | Output path for activation stats. |
| `--samples` | `128` | Number of calibration samples. |
| `--batch-size` | `5` | Cache-clear cadence between samples (see calibration docs). |
| `--max-length` | `2048` | Max sequence length per sample. |

### `awq scales`

| Option | Default | Description |
| --- | --- | --- |
| `--calibration-stats` | required | Path to `calibration_stats.pt`. |
| `--output` | `results/awq_scales.pt` | Output path for scale factors. |
| `--group-size` | `32` | INT4 group size; must match `awq quantize`. |
| `--alpha` | `0.5` | Fixed exponent, used only with `--no-grid-search`. |
| `--no-grid-search` | off | Disable per-layer α search; use `--alpha` instead. |
| `--quantize-strategy` | `alternating` | Which layers to quantize: `all`, `alternating`, `last_only`, `first_only`. |
| `--no-skip-lm-head` | off | Include `lm_head` in quantization (not recommended). |

### `awq quantize`

| Option | Default | Description |
| --- | --- | --- |
| `--scales` | required | Path to `awq_scales.pt`. |
| `--output-dir` | `results/awq_quantized` | Output directory for the quantized model. |
| `--group-size` | `32` | INT4 group size (32–128). |
| `--verify-layers` | `3` | Layers to verify post-quantization. `0` to skip. |

### `awq run` — full pipeline

Runs calibrate → scales → quantize (+ verify). Accepts the union of the
options above (`--model`, `--dataset`, `--output-dir`, `--samples`,
`--batch-size`, `--max-length`, `--group-size`, `--alpha`, `--no-grid-search`,
`--quantize-strategy`, `--device`, `--quiet`).

## Artifacts

```txt
out/
  calibration_stats.pt    # {layer_name: |X|.mean(0)}  (1D tensor per linear layer)
  awq_scales.pt           # {layer_name: per-channel AWQ scale s  ([d_in])}
  awq_quantized/
    quantized_state.pt    # {layer_name: packed INT4 + group scales + AWQ scales}
    metadata.json         # group size, layer count, compression ratio
```

`metadata.json` reports the compression ratio counting packed INT4 weights
**plus** per-group FP16 scales and per-channel AWQ scales, so the number
reflects the true on-disk footprint of the quantized linear weights
(typically ~4× over FP16 for `group_size=32`).

## Model compatibility

`awq` is model-agnostic for any HF causal LM whose transformer linears are
named `model.layers.{i}.*` (the standard Llama/Qwen/Mistral convention). The
skip-set logic (`build_skip_set`) derives the layer count from the
calibration stats, so `--quantize-strategy alternating` / `last_only` /
`first_only` work correctly regardless of depth.

`lm_head` is skipped by default (vocabulary projection is too sensitive to
INT4). Qwen3.5-style hybrid-attention tiny projections
(`linear_attn.in_proj_{a,b,z}`) are also skipped by name when present; on
standard-attention models they simply match nothing.

Tested on: Qwen3-0.6B (28 layers). Larger models (7B+) require a CUDA box —
see [docs/ec2.md](docs/ec2.md).

## Technical Docs

Maintainer-focused reference documentation is in [docs/index.md](docs/index.md):
pipeline, architecture, CLI, calibration, AWQ scales & INT4 quantization,
inference, EC2 deployment, and development.

## References

- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (Lin et al., 2023)](https://arxiv.org/abs/2306.00978)
- Reference scale search: [`mit-han-lab/llm-awq`](https://github.com/mit-han-lab/llm-awq) (`awq/quantize/auto_scale.py`)

## Repository Layout

```txt
awq/
  __init__.py
  __main__.py      # python -m awq
  cli.py           # CLI dispatcher (calibrate / scales / quantize / run)
  models.py        # shared FP16 model loader
  calibrate.py     # on-the-fly activation statistics via hooks
  scales.py        # per-layer AWQ scale computation + α grid search
  quantize.py      # memory-safe INT4 quantizer + reconstruction verify
  inference.py     # dequantization + AWQ model loading
data/
  natural_calibration.json   # bundled WikiText-2 calibration samples
utils/
  memory.py        # device detection, MPS/CUDA memory limiting & tracking
  errors.py        # exceptions, retry decorator, validation, OOM diagnostics
tests/
  test_pipeline.py
docs/
  index.md         # documentation map
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q tests/
python -m awq --help
python -c "from utils.memory import get_device; print(get_device())"
```

## License

Released under the MIT License. See [LICENSE](LICENSE).