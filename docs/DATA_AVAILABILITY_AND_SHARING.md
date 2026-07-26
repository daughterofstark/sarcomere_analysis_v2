# Data Availability And Sharing

This repository is intended to be shared as code, tests, configuration templates, and documentation.

## What Is Not Included

- Raw widefield microscopy images.
- Raw confocal microscopy images.
- Generated result tables and preview images under `results/`.
- Review-pack zip files.
- Local virtual environments, caches, and machine-specific paths.

## Local Data Paths

The shared default config uses a placeholder input path:

```text
/path/to/local/widefield/raw
```

Users should either edit a private local config copy or pass CLI overrides, for example:

```bash
../sarcgraph-env/bin/python scripts/build_manifest.py \
  --config configs/default.yaml \
  --image-dir /path/to/local/widefield/raw
```

For confocal pilot workflows, pass the local confocal directory explicitly:

```bash
../sarcgraph-env/bin/python scripts/run_confocal_baseline_audit.py \
  --config configs/default.yaml \
  --confocal-root /path/to/local/confocal
```

## What To Commit

- `src/`
- `scripts/`
- `tests/`
- `configs/default.yaml`
- `docs/`
- `templates/`
- `README.md`
- project metadata such as `pyproject.toml` and `.gitignore`

## What Not To Commit

- Raw microscopy data: `.tif`, `.tiff`, `.czi`, `.lif`, `.nd2`, `.lsm`
- `results/`
- Generated zip files
- Local environment folders such as `.venv/`, `venv/`, or `env/`
- Cache folders such as `__pycache__/` and `.pytest_cache/`

The GitHub repository alone does not contain enough information to reproduce the original private dataset outputs. It provides the reproducible pipeline and audit tooling; users must provide their own local image data.
