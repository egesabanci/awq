# EC2 Deployment

Install and run `awq` on AWS EC2 with NVIDIA GPUs. CUDA is where you can
calibrate and quantize larger models (7B+) that don't fit comfortably on a
laptop, and where you'd later hand the quantized artifact to an INT4-aware
runtime for real INT4 execution.

## Prerequisites

- AWS EC2 instance with an NVIDIA GPU (A10G, L4, A100)
- Ubuntu 22.04+ AMI (Deep Learning AMI recommended)
- Python 3.11+
- CUDA 12.x + cuDNN

## Instance recommendations

| Instance | GPU | VRAM | Comfortable model size |
| --- | --- | --- | --- |
| `g5.xlarge` | A10G | 24 GB | 7B |
| `g5.2xlarge` | A10G | 24 GB | 7B (+ CPU offload headroom) |
| `g4dn.xlarge` | T4 | 16 GB | 2B–3B |
| `p3.2xlarge` | V100 | 16 GB | 2B–3B |

## Installation

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git

git clone git@github.com:egesabanci/awq.git
cd awq

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### Verify installation

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m awq --help
```

## Running the pipeline

### Get a model

`awq` accepts a Hugging Face repo ID or a local directory. To pin a local copy:

```bash
huggingface-cli download Qwen/Qwen3-1.7B --local-dir /data/models/Qwen3-1.7B
```

### Full pipeline

```bash
python -m awq run \
  --model /data/models/Qwen3-1.7B \
  --dataset wikitext \
  --output-dir /data/out/qwen17 \
  --device cuda \
  --samples 128 \
  --batch-size 8 \
  --max-length 2048 \
  --group-size 32 \
  --quantize-strategy all
```

### Individual steps

```bash
# Calibrate
python -m awq calibrate --model /data/models/Qwen3-1.7B \
  --dataset wikitext --output /data/out/stats.pt --device cuda \
  --batch-size 8 --samples 128 --max-length 2048

# Scales (per-layer α grid search)
python -m awq scales --model /data/models/Qwen3-1.7B \
  --calibration-stats /data/out/stats.pt --output /data/out/scales.pt \
  --group-size 32 --quantize-strategy all

# Quantize (CPU, memory-safe) + verify
python -m awq quantize --model /data/models/Qwen3-1.7B \
  --scales /data/out/scales.pt --output-dir /data/out/awq \
  --group-size 32 --verify-layers 3
```

There is no `benchmark`/eval step in the CLI. To inspect generation quality,
`awq export` the artifact and load it in a real INT4 runtime (AutoAWQ/vLLM);
see [inference.md](inference.md). The bundled `eval/ppl.py` measures
perplexity + greedy generation against the FP16 baseline.

## Memory tuning

| GPU VRAM | Suggested `--batch-size` | Suggested `--samples` |
| --- | --- | --- |
| 16 GB | 4–8 | 64–128 |
| 24 GB | 8–16 | 128 |
| 40 GB+ | 16–32 | 128–256 |

If you hit OOM:

1. Reduce `--batch-size` (most impactful — clears cache more often).
2. Reduce `--max-length` from 2048 to 1024.
3. Reduce `--samples` from 128 to 64.
4. Use `--group-size 128` instead of 32 (coarser, faster quantization).

> Note: `--batch-size` controls cache-clear cadence, not real batching (each
> sample is forwarded individually). See [calibration.md](calibration.md).

### CUDA memory

There is no per-process VRAM cap on CUDA — `limit_memory` (an Apple-Silicon/MPS
helper) was removed. `load_model` with `device_map="auto"` allocates immediately
and a dedicated GPU box should use its full VRAM; capping it prematurely causes
OOM on models that genuinely need it (a 7B/8B FP16 load is ~14-16 GB on a 24 GB
card). `log_memory` still reports GPU usage for observability. If you share the
GPU, set `torch.cuda.set_per_process_memory_fraction` manually before importing
the model.

## Large-model notes

- Calibration loads the full FP16 model on the GPU. A 7B model ≈ 14 GB FP16.
- `awq run` loads the model for calibrate, frees it, then re-loads for scales.
- Quantization is CPU and streams `safetensors` one tensor at a time — it does
  not need the GPU.
- `--quantize-strategy last_only` / `alternating` keep some layers in FP16 if
  you want a mixed-precision artifact.

## Portability

Artifacts are plain `.pt`/`.json` and move freely between EC2 and a laptop:

```bash
scp -i ~/.ssh/key.pem ubuntu@<ip>:/data/out/awq/quantized_state.pt ./
```

## Troubleshooting

| Problem | Cause | Solution |
| --- | --- | --- |
| `CUDA out of memory` | VRAM exhausted | Reduce `--batch-size` / `--max-length` / `--samples`. |
| `No module named 'awq'` | Not installed | `pip install -e .` |
| `model.safetensors.index.json not found` | Sharded model | `awq` handles both sharded and single-file `safetensors`. |
| HuggingFace download timeout | Slow network | Pre-download and pass `--model /path/to/local`. |
| `ScaleError: 0 scales` | No layers matched | Check `--quantize-strategy` and that the model's linears are named `model.layers.{i}.*`. |