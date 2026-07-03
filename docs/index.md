# awq — Technical Documentation

Maintainer-focused reference documentation for `awq`, a model-agnostic
Activation-aware Weight Quantization (AWQ) CLI implemented in pure PyTorch.

The pipeline is deliberately linear:

```txt
calibrate → scales → quantize (+ verify)
```

The package is import-light: importing `awq` or running the CLI does not import
`torch`, `transformers`, or `datasets` until the subcommand that needs them
executes. `awq --help` is instant and side-effect-free.

## Documentation Map

| Document | Purpose |
| --- | --- |
| [Pipeline](pipeline.md) | End-to-end execution flow and phase responsibilities. |
| [Architecture](architecture.md) | Module boundaries, dependency timing, data flow, invariants. |
| [CLI](cli.md) | Command-line reference with examples and failure behavior. |
| [Calibration](calibration.md) | Activation statistics collection, hooks, memory efficiency. |
| [Quantization](quantization.md) | AWQ scale search, skip strategies, INT4 packing, reconstruction. |
| [Inference](inference.md) | Dequantization, weight injection, and exporting to a runtime-loadable INT4 model. |
| [EC2 Deployment](ec2.md) | Installation and usage on AWS EC2 with CUDA GPUs. |
| [Benchmarks](benchmarks.md) | AWQ vs FP16 perplexity + generation results on Qwen3-1.7B/8B. |
| [Development](development.md) | Test commands, import-safety checks, and extension workflow. |

## What `awq` is (and is not)

`awq` quantizes the linear layers of a Hugging Face causal LM to group-wise
INT4 with per-channel AWQ scaling, and verifies the result by reconstruction.

It is **not** an INT4 inference engine. Loading the quantized artifact
(`awq.inference.load_awq_model`) dequantizes weights back to FP16 and runs a
standard forward — useful for sanity-checking quality, but there is no INT4
kernel, so no inference speed/memory win. Hand the artifact to an INT4-aware
runtime (vLLM, TGI, MLX, TensorRT-LLM) for real INT4 execution. See
[inference.md](inference.md).

## Model compatibility

Works on any HF causal LM whose transformer linears are named
`model.layers.{i}.*` (Llama/Qwen/Mistral convention). `build_skip_set` derives
the layer count from the calibration stats, so depth-dependent strategies
(`alternating`, `last_only`, `first_only`) adapt automatically. `lm_head` is
skipped by default.

## Quick Maintainer Commands

```bash
pip install -e ".[dev]"
python -m pytest -q tests/
python -m awq --help
python -c "from utils.memory import get_device; print(get_device())"
git diff --check
```