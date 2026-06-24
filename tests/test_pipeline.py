"""Unit tests for AWQ pipeline modules.

These tests validate the core logic without loading large models.
They run on CPU and should complete in under 10 seconds.
"""

import os
import sys
import tempfile

import pytest
import torch

# Ensure package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMemory:
    """Tests for utils/memory.py — device detection and helpers."""

    def test_get_device_returns_string(self):
        from utils.memory import get_device
        device = get_device()
        assert device in ("cuda", "mps", "cpu")

    def test_get_device_name(self):
        from utils.memory import get_device_name
        name = get_device_name("cpu")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_empty_cache_does_not_crash(self):
        from utils.memory import empty_cache
        empty_cache()  # Should not raise

    def test_get_memory_returns_dict(self):
        from utils.memory import get_memory
        mem = get_memory("cpu")
        assert isinstance(mem, dict)

    def test_log_memory_does_not_crash(self):
        from utils.memory import log_memory
        log_memory("test", "cpu")

    def test_batch_generator_yields_batches(self):
        from utils.memory import batch_generator
        items = list(range(10))
        batches = list(batch_generator(items, batch_size=3))
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[-1] == [9]

    def test_memory_tracker_does_not_crash(self):
        from utils.memory import memory_tracker
        with memory_tracker("test", "cpu"):
            pass


class TestErrors:
    """Tests for utils/errors.py — error handling utilities."""

    def test_retry_success(self):
        from utils.errors import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def succeeds_on_first():
            nonlocal call_count
            call_count += 1
            return 42

        result = succeeds_on_first()
        assert result == 42
        assert call_count == 1

    def test_retry_eventually_succeeds(self):
        from utils.errors import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausts(self):
        from utils.errors import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("always")

        with pytest.raises(ValueError):
            always_fails()
        assert call_count == 2

    def test_custom_exceptions(self):
        from utils.errors import (
            AWQError, CalibrationError, ScaleError,
            QuantizationError, ValidationError,
        )
        assert issubclass(CalibrationError, AWQError)
        assert issubclass(ScaleError, AWQError)
        assert issubclass(QuantizationError, AWQError)
        assert issubclass(ValidationError, AWQError)

    def test_require_calibration_stats_missing(self):
        from utils.errors import require_calibration_stats, ValidationError
        with pytest.raises(ValidationError):
            require_calibration_stats("/nonexistent/path.pt")

    def test_require_calibration_stats_valid(self):
        from utils.errors import require_calibration_stats
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"layer.0": torch.zeros(128)}, f.name)
            result = require_calibration_stats(f.name)
        assert isinstance(result, dict)
        assert "layer.0" in result

    def test_exit_codes(self):
        from utils.errors import EXIT_SUCCESS, EXIT_RUNTIME_ERROR, EXIT_VALIDATION_ERROR
        assert EXIT_SUCCESS == 0
        assert EXIT_RUNTIME_ERROR == 1
        assert EXIT_VALIDATION_ERROR == 2


class TestCalibrate:
    """Tests for awq/calibrate.py — calibration utilities."""

    def test_is_linear_layer(self):
        from awq.calibrate import is_linear_layer
        assert is_linear_layer(torch.nn.Linear(10, 20))
        assert not is_linear_layer(torch.nn.Embedding(100, 32))
        assert not is_linear_layer(torch.nn.LayerNorm(128))

    def test_register_hooks_tiny_model(self):
        from awq.calibrate import register_calibration_hooks

        model = torch.nn.Sequential()
        model.add_module("linear1", torch.nn.Linear(8, 4))
        model.add_module("linear2", torch.nn.Linear(4, 2))

        sums, counts, hooks = register_calibration_hooks(model)
        assert len(hooks) == 2

        # Run a forward pass
        x = torch.randn(1, 8)
        with torch.no_grad():
            model(x)

        assert "linear1" in counts
        assert "linear2" in counts
        assert counts["linear1"] == 1
        assert counts["linear2"] == 1
        assert sums["linear1"].shape == (8,)

        # Cleanup
        for h in hooks:
            h.remove()


class TestScales:
    """Tests for awq/scales.py — AWQ scale computation."""

    def test_compute_awq_scale_basic(self):
        from awq.scales import compute_awq_scale

        weight = torch.randn(64, 128)
        channel_importance = torch.randn(128).abs()

        s = compute_awq_scale(weight, channel_importance, alpha=0.5)
        assert s.shape == (128,)
        assert s.min() >= 0.1
        assert s.max() <= 10.0

    def test_compute_awq_scale_identity(self):
        from awq.scales import compute_awq_scale

        weight = torch.randn(64, 128)
        channel_importance = torch.ones(128)  # uniform → identity scales

        s = compute_awq_scale(weight, channel_importance, alpha=0.5)
        assert torch.allclose(s, torch.ones(128), atol=1e-4)

    def test_compute_awq_scale_dim_mismatch(self):
        from awq.scales import compute_awq_scale

        weight = torch.randn(64, 128)
        channel_importance = torch.randn(64)  # wrong dim

        with pytest.raises(ValueError, match="Channel mismatch"):
            compute_awq_scale(weight, channel_importance)

    def test_compute_awq_scale_not_2d(self):
        from awq.scales import compute_awq_scale

        weight = torch.randn(64)  # 1D
        channel_importance = torch.randn(64)

        with pytest.raises(ValueError, match="Expected 2D weight"):
            compute_awq_scale(weight, channel_importance)

    def test_build_skip_set_alternating(self):
        from awq.scales import build_skip_set

        stats = {
            "model.layers.0.mlp.gate_proj": torch.zeros(128),
            "model.layers.1.mlp.gate_proj": torch.zeros(128),
            "model.layers.2.mlp.gate_proj": torch.zeros(128),
            "model.layers.3.mlp.gate_proj": torch.zeros(128),
        }

        skips = build_skip_set(stats, quantize_strategy="alternating")
        assert "model.layers.0.mlp.gate_proj" in skips  # even
        assert "model.layers.1.mlp.gate_proj" not in skips  # odd
        assert "model.layers.2.mlp.gate_proj" in skips  # even
        assert "model.layers.3.mlp.gate_proj" not in skips  # odd

    def test_build_skip_set_all(self):
        from awq.scales import build_skip_set

        stats = {"model.layers.0.mlp.gate_proj": torch.zeros(128)}

        skips = build_skip_set(stats, quantize_strategy="all")
        assert "model.layers.0.mlp.gate_proj" not in skips  # not skipped in "all"

    def test_build_skip_set_lm_head(self):
        from awq.scales import build_skip_set

        skips = build_skip_set({}, skip_lm_head=True)
        assert "lm_head" in skips

    def test_build_skip_set_tiny_projections(self):
        from awq.scales import build_skip_set

        skips = build_skip_set({}, skip_tiny_projections=True)
        assert "model.layers.0.linear_attn.in_proj_a" in skips
        assert "model.layers.0.linear_attn.in_proj_b" in skips


class TestQuantize:
    """Tests for awq/quantize.py — INT4 quantization."""

    def test_quantize_layer_cpu(self):
        from awq.quantize import quantize_layer_cpu

        weight = torch.randn(32, 128, dtype=torch.float16)
        scale_factors = torch.randn(128).abs() + 0.5

        q = quantize_layer_cpu(weight, scale_factors, group_size=32)
        assert "packed_weights" in q
        assert "group_scales" in q
        assert "scale_factors" in q
        assert "shape" in q
        assert q["shape"] == (32, 128)
        assert q["group_size"] == 32

    def test_quantize_reconstruction(self):
        from awq.quantize import quantize_layer_cpu, _dequantize_group

        torch.manual_seed(42)
        weight = torch.randn(32, 64, dtype=torch.float16)
        scale_factors = torch.ones(64, dtype=torch.float16)  # identity scales

        q = quantize_layer_cpu(weight, scale_factors, group_size=32)
        d_out, d_in = q["shape"]

        # Dequantize
        deq_parts = []
        for packed, qscale in zip(q["packed_weights"], q["group_scales"]):
            w_deq = _dequantize_group(packed, qscale, d_out, q["group_size"])
            deq_parts.append(w_deq)
        deq_weight = torch.cat(deq_parts, dim=1)[:, :d_in]

        mse = (weight - deq_weight).pow(2).mean().item()
        assert mse < 0.02, f"MSE too high: {mse}"

    def test_normalize_safetensors_name(self):
        from awq.quantize import normalize_safetensors_name

        assert normalize_safetensors_name("model.layers.0.self_attn.q_proj.weight") == \
            "model.layers.0.self_attn.q_proj"
        assert normalize_safetensors_name("model.language_model.layers.0.mlp.gate_proj.weight") == \
            "model.layers.0.mlp.gate_proj"


class TestInference:
    """Tests for awq/inference.py — dequantization and model loading."""

    def test_dequantize_layer(self):
        from awq.inference import dequantize_layer
        from awq.quantize import quantize_layer_cpu

        torch.manual_seed(42)
        weight = torch.randn(32, 64, dtype=torch.float16)
        scale_factors = torch.ones(64, dtype=torch.float16)

        q = quantize_layer_cpu(weight, scale_factors, group_size=32)
        deq_weight = dequantize_layer(q)

        assert deq_weight.shape == (32, 64)
        mse = (weight - deq_weight).pow(2).mean().item()
        assert mse < 0.02, f"MSE too high: {mse}"
