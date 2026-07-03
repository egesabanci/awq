# Calibration

Activation statistics collection for AWQ — `awq/calibrate.py`.

## Goal

AWQ protects weight channels whose corresponding **activations** are large.
Calibration collects, per linear layer, the per-input-channel activation
magnitude `|X|.mean(0)` — the `channel_importance` / `x_max` used by the scale
search.

## How it works

`register_calibration_hooks` attaches a forward hook to every `nn.Linear`.
The hook runs `x.abs().mean(dim=0)` on the input and accumulates it as a
**running sum** keyed by layer name, with a running count. After all samples,
each running sum is divided by its count to get the mean activation magnitude.

```python
ci = x.abs().mean(dim=0).cpu()          # [d_in]
running_sums[name] += ci                 # O(d_in) per layer, on CPU
running_counts[name] += 1
```

This is the key memory property: the hook stores `O(d_in)` per layer, **not**
`O(samples × tokens × d_in)`. You can calibrate on hundreds of samples without
holding any activation tensor beyond a single forward.

## Memory efficiency

- `torch.no_grad()` + `model.eval()` — no autograd graph, no training state.
- Per-prompt: tokenize → forward → `del inputs` immediately.
- Between `batch_size` samples: `gc.collect()` + device cache empty
  (`torch.cuda.empty_cache()` on CUDA; no-op on CPU).

> `--batch-size` does **not** batch tokenization or the forward — each sample
> is forwarded one at a time. It only controls how often the cache is cleared.
> Memory is therefore essentially independent of `--batch-size`; only speed
> changes (more clears = slower). Naming is preserved for backwards
> compatibility.

## Output

`calibration_stats.pt` is a `{layer_name: 1D tensor of shape [d_in]}` dict
saved with `torch.save`. Keys are the `named_modules()` names of each linear
layer (e.g. `model.layers.0.self_attn.q_proj`, `lm_head`).

## Device

`run_calibration` moves the per-channel statistic to CPU inside the hook so
the running sums never accumulate on the device. The model itself stays on
the chosen device (CUDA or CPU).