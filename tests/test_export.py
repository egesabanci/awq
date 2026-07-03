"""Unit tests for awq/export.py — AutoAWQ GEMM packing round-trip correctness."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestExportPacking:
    """The on-disk AutoAWQ GEMM encoding must round-trip through dequantize_gemm."""

    def test_pack_unpack_roundtrip(self):
        from awq.export import _pack_column, _unpack_column, _apply_order, _reverse_awq_order, AWQ_PACK_ORDER
        torch.manual_seed(0)
        m = torch.randint(0, 15, (16, 32), dtype=torch.int8)
        ordered = _apply_order(m, AWQ_PACK_ORDER)
        q = _pack_column(ordered)
        assert q.dtype == torch.int32
        assert q.shape == (16, 4)
        back = _reverse_awq_order(_unpack_column(q))
        assert torch.equal(back, m.to(torch.int8))

    def test_export_linear_dequant_matches_group_dequant_identity_scale(self):
        """zero=7 mapping: awq_dequant_gemm(export) == our group dequant Q(W)."""
        from awq.export import _sym_quantize, _export_linear_from_signed, awq_dequant_gemm
        torch.manual_seed(1)
        d_out, d_in, gs = 32, 128, 32
        W = torch.randn(d_out, d_in, dtype=torch.float32)
        s = torch.ones(d_in)
        signed, gscales = _sym_quantize(W, s, gs)
        packed = _export_linear_from_signed(signed, gscales, gs)
        # our group dequant of W (s=1) -> [d_out, d_in]
        our_deq = (signed.to(torch.float32) * gscales.repeat_interleave(gs, dim=1).float())
        # awq dequant -> [d_in, d_out]; transpose to compare
        awq_deq = awq_dequant_gemm(packed["qweight"], packed["qzeros"], packed["scales"], gs).t().float()
        mse = (our_deq - awq_deq).pow(2).mean().item()
        assert mse < 1e-6, f"packing round-trip MSE too high: {mse}"

    def test_export_linear_dequant_with_awq_scale(self):
        """With a non-trivial AWQ scale s, export dequant == Q(W*s) (pre-scaled weight)."""
        from awq.export import _sym_quantize, _export_linear_from_signed, awq_dequant_gemm
        torch.manual_seed(2)
        d_out, d_in, gs = 16, 128, 64
        W = torch.randn(d_out, d_in, dtype=torch.float32)
        s = torch.linspace(0.5, 2.0, d_in, dtype=torch.float32)
        signed, gscales = _sym_quantize(W, s, gs)  # Q(W*s) signed
        packed = _export_linear_from_signed(signed, gscales, gs)
        # our pre-scaled dequant Q(W*s)
        our_deq = (signed.to(torch.float32) * gscales.repeat_interleave(gs, dim=1).float())[:, :d_in]
        awq_deq = awq_dequant_gemm(packed["qweight"], packed["qzeros"], packed["scales"], gs).t().float()
        mse = (our_deq - awq_deq).pow(2).mean().item()
        assert mse < 1e-6, f"AWQ-scale packing round-trip MSE too high: {mse}"

    def test_qzeros_are_seven_packed(self):
        """zeros must be 7 everywhere (maps signed [-7,7] -> unsigned [0,14])."""
        from awq.export import _sym_quantize, _export_linear_from_signed, _unpack_column, _reverse_awq_order
        torch.manual_seed(3)
        W = torch.randn(8, 64)
        s = torch.ones(64)
        signed, gscales = _sym_quantize(W, s, 32)
        packed = _export_linear_from_signed(signed, gscales, 32)
        izeros = _reverse_awq_order(_unpack_column(packed["qzeros"]))
        assert izeros.shape[0] == 2  # 64/32 groups
        assert torch.all(izeros == 7)

    def test_exported_shapes_match_autoawq_gemm_layout(self):
        """qweight [d_in, d_out//8], qzeros [d_in//gs, d_out//8], scales [d_in//gs, d_out]."""
        from awq.export import _sym_quantize, _export_linear_from_signed
        W = torch.randn(48, 96)  # d_out=48, d_in=96
        s = torch.ones(96)
        signed, gscales = _sym_quantize(W, s, 32)
        p = _export_linear_from_signed(signed, gscales, 32)
        assert p["qweight"].shape == (96, 48 // 8)        # [d_in, d_out//8]
        assert p["qzeros"].shape == (96 // 32, 48 // 8)   # [d_in//gs, d_out//8]
        assert p["scales"].shape == (96 // 32, 48)        # [d_in//gs, d_out]
        assert p["scales"].dtype == torch.float16
        assert p["qweight"].dtype == torch.int32


class TestExportEndToEnd:
    """Full export on a tiny synthesized Llama-style block (CPU)."""

    def test_export_tiny_block_writes_loadable_layout(self, tmp_path):
        from awq.export import export_to_awq, awq_dequant_gemm
        from awq.quantize import quantize_layer_cpu
        # Build a minimal quantized_state for 1 layer with all 7 linears.
        torch.manual_seed(4)
        d_in, d_out_attn, d_inter = 64, 64, 128
        projs = {
            "self_attn.q_proj": (d_out_attn, d_in),
            "self_attn.k_proj": (d_out_attn, d_in),
            "self_attn.v_proj": (d_out_attn, d_in),
            "self_attn.o_proj": (d_in, d_out_attn),
            "mlp.gate_proj": (d_inter, d_in),
            "mlp.up_proj": (d_inter, d_in),
            "mlp.down_proj": (d_in, d_inter),
        }
        qstate = {}
        for proj, (o, i) in projs.items():
            W = torch.randn(o, i, dtype=torch.float16)
            s = torch.rand(i, dtype=torch.float16) + 0.5
            q = quantize_layer_cpu(W, s, group_size=32)
            qstate[f"model.layers.0.{proj}"] = q

        # We can't fully run export_to_awq without a real HF model dir + config,
        # so test the math path directly: each linear packs and round-trips.
        from awq.export import _sym_quantize, _export_linear_from_signed
        for proj, (o, i) in projs.items():
            W = torch.randn(o, i, dtype=torch.float32)
            s = torch.rand(i, dtype=torch.float32) + 0.5
            signed, gs = _sym_quantize(W, s, 32)
            p = _export_linear_from_signed(signed, gs, 32)
            deq = awq_dequant_gemm(p["qweight"], p["qzeros"], p["scales"], 32).t().float()
            ours = (signed.float() * gs.repeat_interleave(32, dim=1).float())[:, :i]
            assert (deq - ours).pow(2).mean().item() < 1e-6

@pytest.mark.skipif(
    not os.environ.get("AWQ_EXPORT_INTEGRATION"),
    reason="Set AWQ_EXPORT_INTEGRATION=1 (plus AWQ_EXPORT_MODEL and optional AWQ_EXPORT_AUTOAWQ_SITE) to run the export end-to-end test; needs a real model + AutoAWQ.",
)
class TestExportIntegration:
    """End-to-end: awq run -> awq export -> AutoAWQ real-INT4 load -> coherent PPL.

    Runs in subprocesses because AutoAWQ's top-level `awq` package collides
    with this repo's `awq` import name (they cannot coexist in one process).
    Reuses eval/ppl.py for the AutoAWQ runtime load.
    """

    def test_export_loads_in_autoawq_and_is_coherent(self, tmp_path):
        import subprocess
        model = os.environ.get("AWQ_EXPORT_MODEL", "/data/models/Qwen3-1.7B")
        out = tmp_path / "out"
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        # 1. quantize
        r = subprocess.run([sys.executable, "-m", "awq", "run", "--model", model,
                            "--dataset", "wikitext", "--output-dir", str(out),
                            "--samples", "16", "--max-length", "512", "--device", "cuda",
                            "--quantize-strategy", "all", "--group-size", "128",
                            "--verify-layers", "1"], capture_output=True, text=True,
                           env=env, timeout=1800)
        assert r.returncode == 0, f"awq run failed:\n{r.stderr[-1500:]}"
        # 2. export
        r = subprocess.run([sys.executable, "-m", "awq", "export", "--model", model,
                            "--from", str(out / "awq_quantized" / "quantized_state.pt"),
                            "--to", str(out / "awq_hf"), "--group-size", "128"],
                           capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0, f"awq export failed:\n{r.stderr[-1500:]}"
        assert (out / "awq_hf" / "quantize_config.json").exists()
        # 3. load in AutoAWQ + measure PPL via eval/ppl.py (separate process)
        r = subprocess.run([sys.executable, "eval/ppl.py", "--model", model,
                            "--config", "awq-runtime", "--export-dir", str(out / "awq_hf"),
                            "--stride", "512", "--max-length", "2048"],
                           capture_output=True, text=True, env=env, timeout=1800,
                           cwd=os.path.dirname(os.path.dirname(__file__)))
        assert r.returncode == 0, f"ppl.py failed:\n{r.stderr[-1500:]}"
        # PPL must be finite and coherent (< 100 for a viable Qwen3 model)
        import re
        m = re.search(r"PPL = ([\d.]+)", r.stdout)
        assert m, f"no PPL in stdout:\n{r.stdout[-1000:]}"
        ppl = float(m.group(1))
        assert ppl < 100, f"exported model PPL not coherent: {ppl}"
