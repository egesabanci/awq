# EC2 Deployment

Install and run the AWQ quantization pipeline on AWS EC2 with CUDA GPUs.

## Prerequisites

- AWS EC2 instance with NVIDIA GPU (A10G, L4, A100)
- Ubuntu 22.04+ AMI (Deep Learning AMI recommended)
- Python 3.11+
- CUDA 12.x + cuDNN

## Instance Recommendations

| Instance | GPU | VRAM | Max model size | Cost profile |
| --- | --- | --- | --- | --- |
| `g5.xlarge` | A10G | 24 GB | 7B models | Cost-effective |
| `g5.2xlarge` | A10G | 24 GB | 7B models (+ CPU offload) | Good for 7B |
| `g4dn.xlarge` | T4 | 16 GB | 2B-3B models | Cheap option |
| `p3.2xlarge` | V100 | 16 GB | 2B-3B models | Legacy |

## Installation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and git
sudo apt install -y python3.11 python3.11-venv git

# Clone the repo
git clone git@github.com:egesabanci/awq-qwen-experiment.git
cd awq-qwen-experiment

# Create venv and install
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# Check CLI
python -m awq --help
```

## Running the Pipeline

### Download a Model

The pipeline auto-downloads models from HuggingFace. First run will cache
the model:

```bash
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-2B')"
```

Or download to a specific directory:

```bash
huggingface-cli download Qwen/Qwen3.5-2B --local-dir /data/models/Qwen3.5-2B
```

### Run Full Pipeline

```bash
python -m awq run \
  --model Qwen/Qwen3.5-2B \
  --dataset wikitext \
  --output-dir /data/results/awq-experiment \
  --device cuda \
  --batch-size 16 \
  --samples 128 \
  --eval-samples 50 \
  --group-size 32
```

### Run Individual Steps

```bash
# Calibrate
python -m awq calibrate \
  --model Qwen/Qwen3.5-2B \
  --dataset wikitext \
  --output /data/results/stats.pt \
  --device cuda \
  --batch-size 16

# Scales
python -m awq scales \
  --model Qwen/Qwen3.5-2B \
  --calibration-stats /data/results/stats.pt \
  --output /data/results/scales.pt

# Quantize (always on CPU for memory safety)
python -m awq quantize \
  --model Qwen/Qwen3.5-2B \
  --scales /data/results/scales.pt \
  --output-dir /data/results/quantized

# Benchmark
python -m awq benchmark \
  --model Qwen/Qwen3.5-2B \
  --awq-dir /data/results/quantized \
  --output /data/results/comparison.json \
  --device cuda \
  --eval-samples 50
```

## Memory Tuning

| GPU VRAM | Suggested `--batch-size` | Suggested `--samples` |
| --- | --- | --- |
| 16 GB | 4-8 | 64-128 |
| 24 GB | 8-16 | 128 |
| 40 GB+ | 16-32 | 128-256 |

If you hit OOM:

1. Reduce `--batch-size` (most impactful)
2. Reduce `--max-length` from 2048 to 1024
3. Reduce `--samples` from 128 to 64
4. Use `--group-size 128` instead of 32 (faster, coarser quantization)

## Large Model Notes

For 7B+ models:

- Calibration: expect ~14 GB VRAM at batch_size=8
- Benchmark: expect ~16 GB VRAM for dual model loading
- Consider using `AWQModelWrapper` for memory-efficient inference
- The `--quantize-strategy last_only` may work better on large models

## Data Transfer

Results are fully portable between EC2 and local macOS:

```bash
# From EC2 to local
scp -i ~/.ssh/key.pem ubuntu@<ip>:/data/results/comparison.json ./results/

# Analyze locally
python -c "import json; r = json.load(open('results/comparison.json')); print(r['evaluation']['per_metric'])"
```

## Troubleshooting

| Problem | Cause | Solution |
| --- | --- | --- |
| `CUDA out of memory` | VRAM exhausted | Reduce `--batch-size` or `--max-length` |
| `No module named 'awq'` | Not installed | `pip install -e .` |
| `model.safetensors.index.json not found` | Sharded model | The pipeline handles both sharded and single-file |
| HuggingFace download timeout | Slow network | Use `--model /path/to/local/model` with pre-downloaded weights |
| `AssertionError: Scales dict is empty` | No layers matched | Check `--quantize-strategy` and model architecture |
