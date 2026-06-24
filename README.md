<div align="center">

# AWQ Qwen Experiment

**Activation-aware weight quantization research pipeline.**

[Quick Start](#quick-start) |
[Workflow](#workflow) |
[CLI Reference](#cli-reference) |
[Supported Models](#supported-models) |
[Metrics](#metrics) |
[Technical Docs](#technical-docs) |
[References](#references) |
[Development](#development) |
[License](#license)

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Runtime](https://img.shields.io/badge/Runtime-PyTorch%20(MPS%20%7C%20CUDA)-orange)
![Package](https://img.shields.io/badge/Package-awq--qwen--experiment-green)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

AWQ Qwen Experiment applies **Activation-aware Weight Quantization** to
small language models (2B-7B) and evaluates quality on tool-calling tasks.
It runs fully locally on Apple Silicon (MPS) or CUDA GPUs (EC2).

Use it when you want to study how INT4 compression affects model output
quality without relying on existing quantization toolkits (AutoGPTQ, etc.).

**Key finding from v0.1:** INT4 at 2B scale produces per-layer cosine ~0.90
which compounds destructively across 24 layers. The method works best on
models 7B+.

## Highlights

- **Pure PyTorch AWQ implementation** — no external quantization libs needed
- **Device-agnostic** — auto-detects CUDA, MPS, or CPU
- **Memory-safe quantizer** — reads safetensors from disk one tensor at a time
- **CLI pipeline** — single command: `awq run --model ... --dataset ...`
- **Structured telemetry** — calibration stats, scales, quantized weights,
  and comparison benchmark all saved as standard formats
- **Import-light** — CLIs only import heavy libs (torch, transformers) when
  a subcommand actually needs them

## Quick Start

Prerequisites:

- macOS (Apple Silicon) or Linux with CUDA
- Python 3.11+
- PyTorch 2.4+
- An FP16 model (tested with Qwen3.5-2B)

Install:

```bash
git clone git@github.com:egesabanci/awq-qwen-experiment.git
cd awq-qwen-experiment
pip install -e .
```

Check the CLI:

```bash
python -m awq --help
```

## Workflow

The pipeline is intentionally linear and inspectable.

| Step | What happens | Evidence |
| --- | --- | --- |
| 1. Calibrate | Forward calibration samples through FP16 model; collect activation channel importance. | `calibration_stats.pt` |
| 2. Scales | Compute per-channel AWQ scaling factors from activation statistics. | `awq_scales.pt` |
| 3. Quantize | Read model safetensors from disk, quantize to INT4 with group-wise packing. | `quantized_state.pt`, `metadata.json` |
| 4. Benchmark | Run FP16 baseline + AWQ model on eval set; compare metrics. | `comparison.json` |

Run all four steps with one command:

```bash
awq run --model Qwen/Qwen3.5-2B --dataset wikitext --output-dir ./results
```

Or run steps individually:

```bash
awq calibrate --model Qwen/Qwen3.5-2B --dataset wikitext --output results/stats.pt
awq scales --model Qwen/Qwen3.5-2B --calibration-stats results/stats.pt --output results/scales.pt
awq quantize --model Qwen/Qwen3.5-2B --scales results/scales.pt --output-dir results/quantized
awq benchmark --model Qwen/Qwen3.5-2B --awq-dir results/quantized --output results/comparison.json
```

## CLI Reference

Run:

```bash
python -m awq --help
```

### Global Options

| Option | Default | Description |
| --- | --- | --- |
| `--model` | required | Model path or Hugging Face repo ID. |
| `--dataset` | `wikitext` | Calibration/eval dataset (`wikitext`, `c4`, `toolace`). |
| `--device` | auto | Compute device (`cuda`, `mps`, `cpu`). Auto-detected. |
| `--quiet` | off | Suppress progress output. |

### Calibrate (`awq calibrate`)

| Option | Default | Description |
| --- | --- | --- |
| `--output` | `results/calibration_stats.pt` | Output path for activation stats. |
| `--samples` | `128` | Number of calibration samples. |
| `--batch-size` | `5` | Calibration batch size (lower = less memory). |
| `--max-length` | `2048` | Max sequence length per sample. |

### Scales (`awq scales`)

| Option | Default | Description |
| --- | --- | --- |
| `--calibration-stats` | required | Path to `calibration_stats.pt`. |
| `--output` | `results/awq_scales.pt` | Output path for scale factors. |
| `--alpha` | `0.5` | AWQ scaling strength (0.0 = no scaling). |
| `--quantize-strategy` | `alternating` | Which layers to quantize: `all`, `alternating`, `last_only`, `first_only`. |
| `--no-skip-lm-head` | off | Include `lm_head` in quantization (not recommended). |

### Quantize (`awq quantize`)

| Option | Default | Description |
| --- | --- | --- |
| `--scales` | required | Path to `awq_scales.pt`. |
| `--output-dir` | `results/awq_quantized` | Output directory for quantized model. |
| `--group-size` | `32` | INT4 group size (32-128). |
| `--verify-layers` | `3` | Layers to verify post-quantization. 0 to skip. |

### Benchmark (`awq benchmark`)

| Option | Default | Description |
| --- | --- | --- |
| `--awq-dir` | `results/awq_quantized` | Directory with quantized model. |
| `--output` | `results/comparison.json` | Output path for comparison results. |
| `--eval-samples` | `10` | Number of evaluation samples. |
| `--max-new-tokens` | `256` | Max new tokens per generation. |

### Run (`awq run`) — Full Pipeline

| Option | Default | Description |
| --- | --- | --- |
| `--output-dir` | `results` | Output directory for all artifacts. |
| `--samples` | `128` | Number of calibration samples. |
| `--eval-samples` | `10` | Number of evaluation samples. |
| `--batch-size` | `5` | Calibration batch size. |
| `--max-length` | `2048` | Max sequence length. |
| `--max-new-tokens` | `256` | Max tokens per generation in benchmark. |
| `--group-size` | `32` | INT4 group size. |
| `--alpha` | `0.5` | AWQ scaling strength. |
| `--quantize-strategy` | `alternating` | Which layers to quantize. |
| `--skip-benchmark` | off | Skip the benchmark step after quantization. |

## Supported Models

| Model | Parameters | FP16 Size | Status |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-2B` | 1.88B | 4.2 GB | Tested (INT4 fails — see findings) |
| `Qwen/Qwen3-4B` | 3.9B | ~8 GB | Downloaded (partial) |

INT4 quantization is expected to work on 7B+ models. Testing on larger models
is planned but requires appropriate hardware.

## Metrics

Every run produces structured artifacts under the output directory:

```txt
results/
  calibration_stats.pt    # Activation channel importance (128+ tensors)
  awq_scales.pt           # Per-layer AWQ scale factors
  awq_quantized/
    quantized_state.pt    # Packed INT4 weights (~25% of original size)
    metadata.json         # Compression ratio, group size, layer count
  comparison.json         # FP16 vs AWQ quality comparison
```

The comparison JSON includes per-metric deltas:

| Field | Description |
| --- | --- |
| `evaluation.per_metric` | Per-metric FP16/AWQ/delta scores |
| `evaluation.semantic_similarity` | Cosine similarity between FP16 and AWQ outputs |
| `fp16.timing` | FP16 throughput (tok/s) |
| `fp16.memory` | FP16 peak memory (GB) |
| `awq.mean_tokens_per_sec` | AWQ throughput (tok/s) |

## Technical Docs

Maintainer-focused reference documentation is available in
[docs/index.md](docs/index.md). It covers the pipeline, architecture,
calibration, AWQ scale computation, INT4 quantization, inference,
benchmarking, and EC2 deployment.

## References

- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [ToolACE: A Toolkit for Evaluating Tool-Augmented LLMs](https://huggingface.co/datasets/lockon/ToolACE)
- [A Practical Guide to INT4 Quantization for SLMs (Microsoft)](https://medium.com/data-science-at-microsoft/a-practical-guide-to-int4-quantization-for-slms-gptq-vs-awq-olive-and-real-world-results-2f63d6963d1d)

## Repository Layout

```txt
awq-qwen-experiment/
  README.md
  LICENSE
  pyproject.toml
  awq/
    __init__.py
    __main__.py      # Entry point: python -m awq
    cli.py           # CLI dispatcher with subcommands
    calibrate.py     # Activation statistics collection
    scales.py        # AWQ scale computation
    quantize.py      # Memory-safe INT4 quantizer
    inference.py     # Dequantization and model loading
  data/
    __init__.py
    loader.py        # ToolACE dataset loader
    natural_calibration.json  # WikiText-2 samples
  eval/
    __init__.py
    runner.py        # FP16 inference runner
    benchmark.py     # FP16 vs AWQ comparison
    metrics.py       # Evaluation metrics suite
  utils/
    __init__.py
    memory.py        # Device-agnostic memory management
    errors.py        # Error handling, retries, validation
  tests/
    test_pipeline.py # 28 unit tests
  docs/
    index.md         # Documentation map
  report/
    EXPERIMENT_REPORT.md  # Full experimental findings
```

## Development

Install dev dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest -q tests/
```

Run CLI smoke check:

```bash
python -m awq --help
python -c "from utils.memory import get_device; print(get_device())"
```

Check formatting before committing:

```bash
git diff --check
```

## EC2 Deployment

See [docs/ec2.md](docs/ec2.md) for installation and usage on AWS EC2 with
CUDA GPUs.

## License

AWQ Qwen Experiment is released under the MIT License. See [LICENSE](LICENSE).
