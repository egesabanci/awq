# AWQ Qwen Experiment Technical Documentation

This directory contains maintainer-focused reference documentation for the
AWQ quantization pipeline. It describes the behavior implemented in
`awq/`, `eval/`, `data/`, and `utils/`.

The pipeline is deliberately linear:

```txt
calibrate → scales → quantize → benchmark
```

The package is also import-light. Importing `awq` or its CLI must not import
`torch`, `transformers`, or `datasets` until the subcommand that needs them
executes.

## Documentation Map

| Document | Purpose |
| --- | --- |
| [Pipeline](pipeline.md) | End-to-end execution flow and phase responsibilities. |
| [Architecture](architecture.md) | Module boundaries, dependency timing, data flow, and invariants. |
| [CLI](cli.md) | Command-line reference with examples and failure behavior. |
| [Calibration](calibration.md) | Activation statistics collection, hooks, memory efficiency. |
| [AWQ Scales](quantization.md) | Scale formula, strategy options, and layer skipping. |
| [INT4 Quantization](quantization.md) | Memory-safe quantizer, packing format, reconstruction quality. |
| [Inference](inference.md) | Dequantization, weight injection, and memory-efficient wrapper. |
| [Benchmark](benchmark.md) | FP16 vs AWQ comparison methodology and metrics. |
| [EC2 Deployment](ec2.md) | Installation and usage on AWS EC2 with CUDA GPUs. |
| [Development](development.md) | Test commands, import-safety checks, and extension workflow. |

## Supported Models

| Model | Architecture | AWQ Status |
| --- | --- | --- |
| Qwen3.5-2B | 24-layer dense, mixed linear + full attention | Full pipeline tested. INT4 quality loss severe. |
| Qwen3-4B | Dense | FP16 partially downloaded. |

## Quick Maintainer Commands

```bash
pip install -e ".[dev]"
python -m pytest -q tests/
python -m awq --help
python -c "from utils.memory import get_device; print(get_device())"
git diff --check
```
