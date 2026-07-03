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
        assert device in ("cuda", "cpu")

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


class TestDequant:
    """Tests for awq.quantize.dequantize_layer — the canonical dequant (verify path)."""

    def test_dequantize_layer(self):
        from awq.quantize import dequantize_layer
        from awq.quantize import quantize_layer_cpu

        torch.manual_seed(42)
        weight = torch.randn(32, 64, dtype=torch.float16)
        scale_factors = torch.ones(64, dtype=torch.float16)

        q = quantize_layer_cpu(weight, scale_factors, group_size=32)
        deq_weight = dequantize_layer(q)

        assert deq_weight.shape == (32, 64)
        mse = (weight - deq_weight).pow(2).mean().item()
        assert mse < 0.02, f"MSE too high: {mse}"


class TestAwqCorrectness:
    """Regression tests for the three core AWQ algorithm-correctness bugs.

    Bug 1: AWQ scale direction (weight scaled UP before quant, divided after).
    Bug 2: scale search uses the weight (grid search over alpha).
    Bug 3: verify_reconstruction uses the full dequant (includes 1/s).
    """

    def test_awq_direction_protects_salient_channel(self):
        """Bug 1: correct direction (W*s, /s) must beat inverted (W/s, *s)
        on activation-weighted error for a salient channel."""
        from awq.scales import compute_awq_scale, _pseudo_quantize_int4

        torch.manual_seed(2)
        d_out, d_in = 8, 64
        W = torch.randn(d_out, d_in) * 0.05
        s_x = torch.full((d_in,), 0.05)
        s_x[0] = 3.0  # channel 0 is salient

        s = compute_awq_scale(W, s_x, grid_search=True, group_size=32)

        # Correct direction (what the pipeline now does)
        err_correct = (((W - _pseudo_quantize_int4(W * s, 32) / s) ** 2)
                       * s_x).sum().item()
        # Inverted direction (the old bug)
        err_inverted = (((W - _pseudo_quantize_int4(W / s, 32) * s) ** 2)
                        * s_x).sum().item()

        assert err_correct < err_inverted, (
            f"Correct direction should be better: {err_correct} vs {err_inverted}")

    def test_grid_search_beats_rtn(self):
        """Bug 2: grid search uses the weight and never does worse than RTN."""
        from awq.scales import compute_awq_scale, _pseudo_quantize_int4

        torch.manual_seed(5)
        d_out, d_in = 32, 64
        W = torch.randn(d_out, d_in) * 0.05  # small weights: RTN is lossy
        s_x = torch.rand(d_in) + 0.01
        s_x[0] = 5.0  # a salient channel

        s = compute_awq_scale(W, s_x, grid_search=True, group_size=32)

        def weighted_err(scale):
            w_hat = _pseudo_quantize_int4(W * scale, 32) / scale
            return (((W - w_hat) ** 2) * s_x).sum().item()

        err_best = weighted_err(s)
        err_rtn = weighted_err(torch.ones(d_in))  # alpha=0, no scaling

        assert err_best <= err_rtn, (
            f"Grid search should beat RTN: {err_best} vs {err_rtn}")

    def test_compute_awq_scale_uses_weight(self):
        """Bug 2: scales must depend on the weight, not just activations."""
        from awq.scales import compute_awq_scale

        torch.manual_seed(4)
        s_x = torch.rand(64) * 2 + 0.1
        s_x[0] = 5.0
        W_small = torch.randn(32, 64) * 0.02   # tiny weights
        W_large = torch.randn(32, 64) * 2.0     # large weights

        s_small = compute_awq_scale(W_small, s_x, grid_search=True, group_size=32)
        s_large = compute_awq_scale(W_large, s_x, grid_search=True, group_size=32)

        # With the old code (weight ignored) these were identical; the search
        # over the weight's quantization error must now differentiate them.
        assert not torch.allclose(s_small, s_large, atol=1e-3), (
            "Scales should differ when weights differ (weight must be used)")

    def test_dequantize_layer_roundtrip_nonidentity(self):
        """Bug 3 / consistency: dequant must invert quant with non-identity scales.

        If dequantize_layer forgot the 1/s back-apply, this MSE would blow up."""
        from awq.quantize import dequantize_layer
        from awq.quantize import quantize_layer_cpu

        torch.manual_seed(3)
        W = torch.randn(16, 64, dtype=torch.float16)
        s = torch.linspace(0.5, 2.0, 64, dtype=torch.float16)

        q = quantize_layer_cpu(W, s, group_size=32)
        W_hat = dequantize_layer(q).float()

        mse = (W.float() - W_hat).pow(2).mean().item()
        # Missing 1/s would give MSE ~ ||W - Q(W*s)||^2 ≈ 0.25 here; correct is
        # the INT4 quant error (~0.02-0.03 in fp16). 0.05 cleanly separates them.
        assert mse < 0.05, f"Round-trip MSE too high (dequant missing 1/s?): {mse}"

    def test_verify_reconstruction_uses_full_dequant(self, tmp_path):
        """Bug 3: verify_reconstruction must report quant error, not scale magnitude.

        Old code dequantized via _dequantize_group (no 1/s) and compared against
        W, yielding MSE ~ ||W - Q(W*s)|| which is dominated by the scale factor.
        With the fix it uses dequantize_layer (includes 1/s) → small MSE.
        """
        from safetensors.torch import save_file
        from awq.quantize import quantize_layer_cpu, verify_reconstruction

        torch.manual_seed(7)
        d_out, d_in = 16, 64
        W = torch.randn(d_out, d_in, dtype=torch.float16)
        s = torch.linspace(0.5, 2.0, d_in, dtype=torch.float16)

        name = "model.layers.0.mlp.gate_proj.weight"
        save_file({name: W}, str(tmp_path / "model.safetensors"))

        q = quantize_layer_cpu(W, s, group_size=32)
        key = "model.layers.0.mlp.gate_proj"
        errors = verify_reconstruction({key: q}, str(tmp_path), num_layers=1)

        assert key in errors
        # Old (buggy) verify reported ~0.3-1.0 here; correct quant error is tiny.
        assert errors[key] < 0.05, (
            f"verify MSE too high (missing 1/s back-apply?): {errors[key]}")


class TestPipelineRobustness:
    """Regression tests for the bugs found during the real-model test run.

    T11: build_skip_set must be model-agnostic (derive layer count from stats).
    T5/T6: compute_all_scales must surface failures and raise on 0 layers.
    """

    @staticmethod
    def _stats(n_layers, proj="mlp.gate_proj"):
        return {f"model.layers.{i}.{proj}": torch.zeros(64) for i in range(n_layers)}

    def test_build_skip_set_alternating_28_layers(self):
        # T11: must not be hardcoded to 24 layers.
        from awq.scales import build_skip_set
        stats = self._stats(28)
        skips = build_skip_set(stats, quantize_strategy="alternating")
        assert "model.layers.0.mlp.gate_proj" in skips   # even
        assert "model.layers.1.mlp.gate_proj" not in skips  # odd
        assert "model.layers.26.mlp.gate_proj" in skips   # even, beyond old 24 cap
        assert "model.layers.27.mlp.gate_proj" not in skips  # odd, beyond old 24 cap

    def test_build_skip_set_last_only_28_layers(self):
        from awq.scales import build_skip_set
        stats = self._stats(28)
        skips = build_skip_set(stats, quantize_strategy="last_only")
        # first half (0..13) skipped, last half (14..27) quantized
        assert "model.layers.0.mlp.gate_proj" in skips
        assert "model.layers.13.mlp.gate_proj" in skips
        assert "model.layers.14.mlp.gate_proj" not in skips
        assert "model.layers.27.mlp.gate_proj" not in skips

    def test_build_skip_set_tiny_proj_derives_layer_count(self):
        # T11: tiny-projection skips should cover all detected layers, not just 0..23.
        from awq.scales import build_skip_set
        stats = {f"model.layers.{i}.linear_attn.in_proj_a": torch.zeros(8)
                 for i in range(28)}
        skips = build_skip_set(stats, skip_tiny_projections=True)
        assert "model.layers.27.linear_attn.in_proj_a" in skips  # beyond old 24 cap

    def test_compute_all_scales_raises_on_zero_layers(self):
        # T5/T6: a silent "0 layers" success must become a loud error.
        from awq.scales import compute_all_scales
        from utils.errors import ScaleError
        model = torch.nn.Sequential()
        model.add_module("real_name", torch.nn.Linear(8, 4))
        stats = {"nonexistent_layer": torch.zeros(8)}  # matches no model linear
        with pytest.raises(ScaleError):
            compute_all_scales(model, stats, skip_set=set(), grid_search=False, verbose=False)

    def test_compute_all_scales_succeeds_on_match(self):
        from awq.scales import compute_all_scales
        model = torch.nn.Sequential()
        model.add_module("lin", torch.nn.Linear(8, 4))
        stats = {"lin": torch.rand(8) * 2 + 0.1}
        scales = compute_all_scales(model, stats, skip_set=set(),
                                    grid_search=False, verbose=False)
        assert "lin" in scales
        assert scales["lin"].shape == (8,)


@pytest.mark.skipif(
    not os.environ.get("AWQ_INTEGRATION"),
    reason="Set AWQ_INTEGRATION=1 (and optionally AWQ_INTEGRATION_MODEL) to run the end-to-end CLI smoke test; needs network + a real model.",
)
class TestIntegrationCLI:
    """End-to-end smoke test: invoke `awq run` as a subprocess on a real model.

    Opt-in because it downloads/loads a real model. Intended for CUDA/EC2 or
    any machine with enough RAM. Run with:
        AWQ_INTEGRATION=1 pytest -q tests/test_pipeline.py -k integration
    Override the model and sample budget via env vars:
        AWQ_INTEGRATION_MODEL (default Qwen/Qwen3-0.6B)
        AWQ_INTEGRATION_SAMPLES (default 8)
        AWQ_INTEGRATION_MAX_LENGTH (default 512)
    """

    def test_awq_run_end_to_end(self, tmp_path):
        import subprocess

        model = os.environ.get("AWQ_INTEGRATION_MODEL", "Qwen/Qwen3-0.6B")
        samples = os.environ.get("AWQ_INTEGRATION_SAMPLES", "8")
        max_length = os.environ.get("AWQ_INTEGRATION_MAX_LENGTH", "512")
        out_dir = tmp_path / "out"

        cmd = [
            sys.executable, "-m", "awq", "run",
            "--model", model,
            "--dataset", "wikitext",
            "--output-dir", str(out_dir),
            "--samples", samples,
            "--max-length", max_length,
            "--verify-layers", "1",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(__file__)) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)

        assert result.returncode == 0, (
            f"awq run exited {result.returncode}\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
        )
        # All three artifacts must exist.
        assert (out_dir / "calibration_stats.pt").exists()
        assert (out_dir / "awq_scales.pt").exists()
        assert (out_dir / "awq_quantized" / "quantized_state.pt").exists()
        assert (out_dir / "awq_quantized" / "metadata.json").exists()
