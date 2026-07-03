# Development

## Setup

```bash
pip install -e ".[dev]"          # installs pytest, ruff
```

## Tests

```bash
python -m pytest -q tests/
```

Tests are CPU-only, load no real models, and run in a few seconds. They cover
memory helpers, error utilities, calibration hooks, AWQ scale math (including
the grid-search and direction regression tests), quantize/dequant round-trip, the AutoAWQ GEMM packing round-trip
(test_export.py), and the model-agnostic skip-set logic.

## Import-safety check

The CLI must stay import-light. Verify that `awq --help` does not pull in
`torch` / `transformers`:

```bash
python -m awq --help            # should be instant
python -c "from utils.memory import get_device; print(get_device())"
```

## Formatting

```bash
git diff --check
ruff check .
```

## Extending the pipeline

- **New subcommand:** add a subparser in `awq/cli.py`, a `cmd_<name>` function,
  and a dispatch branch in `main()`. Keep heavy imports inside the `cmd_*`
  body so `--help` stays light. Return `0` on success, `1` on error
  (`__main__.py` does `sys.exit(main())`).
- **New model architecture:** `build_skip_set` derives layer count from the
  calibration stats, so depth-dependent strategies adapt automatically. If a
  model uses non-standard linear names (not `model.layers.{i}.*`), the
  skip-set regex won't match — calibrate/scales/quantize still run (they key
  by exact `named_modules()` names), only the depth-dependent strategies fall
  back to `all`.
- **Real INT4 inference:** handled by `awq export`, which re-packs
  `quantized_state.pt` into a runtime-loadable AutoAWQ/HF-AWQ model (load it in
  AutoAWQ or vLLM). See [inference.md](inference.md).

## Releasing

The distribution/package is `awq` (`pyproject.toml`). The import package is
`awq`. Bump `version` in `pyproject.toml` for releases.