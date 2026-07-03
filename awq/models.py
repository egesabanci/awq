"""Model loading for the AWQ pipeline.

Provides ``load_model()``: load an FP16 causal LM + tokenizer on the chosen
device. Used by the calibrate/scales/run subcommands and by ``awq.export`` so
there is a single loading code path.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.memory import get_device
from utils.errors import retry_with_backoff


@retry_with_backoff(max_retries=3)
def load_model(
    model_path: str,
    device: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM in FP16 on the target device.

    Args:
        model_path: Local model directory or HuggingFace repo ID.
        device: Torch device string. Auto-detected if None.

    Returns:
        (model, tokenizer) with the model in eval() mode.
    """
    if device is None:
        device = get_device()

    print(f"Loading model from {model_path}...")
    # device_map="auto" for CUDA (single/multi GPU); device string for CPU.
    device_map = "auto" if device == "cuda" else device
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    return model, tokenizer