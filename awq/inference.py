"""AWQ inference — load quantized INT4 weights and run forward passes.

Provides:
- dequantize_layer(): Convert packed INT4 back to FP16 (re-exported from awq.quantize)
- load_awq_model(): Load the FP16 model shell and replace its linear weights
  with dequantized AWQ ones.

Inference is dequantized-FP16: there is no INT4 kernel, so there is no
speed/memory benefit at inference time on its own. The quantized artifact is
what you hand to an INT4-aware runtime (vLLM, TGI, MLX, TensorRT-LLM, …) for
real INT4 execution. See docs/inference.md.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.memory import get_device
from awq.quantize import dequantize_layer  # canonical dequant (shared with verify)


def load_awq_model(
    quantized_path: str,
    model_path: str,
    device: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, dict[str, dict]]:
    """Load the FP16 model shell and replace its linear weights with dequantized AWQ ones.

    Args:
        quantized_path: Path to quantized_state.pt
        model_path: Path to FP16 model directory (for config + tokenizer)
        device: Target device. Auto-detected if None. On CUDA the model is
            loaded with ``device_map="auto"``, so each dequantized weight is
            placed on the module's existing device to respect sharding.

    Returns:
        (model, tokenizer, quantized_state)

    Note:
        This is dequantized-FP16 inference (no INT4 kernel) — peak memory is
        ~one full FP16 model, not the INT4 footprint. Packed INT4 weights stay
        on CPU and are dequantized one tensor at a time, so the quantizer's
        memory profile is not re-incurred here.
    """
    if device is None:
        device = get_device()

    # Load the FP16 model shell via the shared loader, then overwrite linear
    # weights with dequantized AWQ ones.
    from awq.models import load_model
    model, tokenizer = load_model(model_path, device)

    print(f"Loading quantized weights from {quantized_path}...")
    quantized_state = torch.load(quantized_path, map_location="cpu", weights_only=True)
    print(f"  {len(quantized_state)} layers loaded")

    # Dequantize and inject weights. Place each weight on the module's own
    # device so CUDA device_map="auto" sharding is preserved.
    named_modules = {n: m for n, m in model.named_modules()}

    for layer_name, q in quantized_state.items():
        mod = named_modules.get(layer_name)
        if mod is None:
            alt_name = layer_name.replace("model.", "model.language_model.")
            mod = named_modules.get(alt_name)
            if mod is None:
                continue

        if isinstance(mod, torch.nn.Linear):
            w = dequantize_layer(q)
            mod.weight.data = w.to(device=mod.weight.device, dtype=torch.float16)

    model.eval()
    return model, tokenizer, quantized_state