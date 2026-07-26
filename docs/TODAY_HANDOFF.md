# Today Handoff

Project path: `<repo-root>`

## Current Status

The minimal classical image-analysis pipeline is runnable end to end. It processes archival myocardium TIFFs into per-patch and per-image computational metrics, QC previews, and provenance. The full 145-image batch completed successfully.

This is not a clinical/statistical analysis and does not make biological conclusions.

## Implemented Stages

0. Scaffold: clean project structure, package metadata, config, scripts, tests, docs.
1. Config/calibration: config loader, pixel size `0.1299 um/px`, spacing band derived from config.
2. IO/manifest: TIFF discovery, filename parsing, donor/image IDs, manifest building, TIFF loading.
3. Preprocessing: float conversion, percentile clipping, `[0, 1]` scaling, Gaussian background subtraction, optional mild denoising, no CLAHE in measurement path.
4. Masks/QC/patch grid: tissue mask, fixed patch grid, patch-level QC gates and invalid reasons.
5. Structure tensor/OOP: gradients, tensor smoothing, axial orientation, coherence, energy, image and patch OOP.
6. Spacing scaffold: conservative directional autocorrelation primary estimator and FFT scaffold/cross-check; weak patches return `NaN`.
7. Standardized outputs/provenance: per-image/per-patch tables, preview PNGs, run provenance JSON.
8. CLI/batch runner: one-image CLI, batch CLI, shared `pipeline.py` orchestration.
9. Schema/test hardening: centralized schemas, deterministic output ordering, 66 tests, development smoke script.
10. Full-batch operational audit: full 145-image batch run and audit summary generated.

## Reproduction Commands

Run tests:

```bash
../sarcgraph-env/bin/python -m pytest
```

Run development smoke:

```bash
PYTHON=../sarcgraph-env/bin/python bash scripts/dev_smoke.sh
```

Build manifest:

```bash
../sarcgraph-env/bin/python scripts/build_manifest.py --config configs/default.yaml
```

One-image run:

```bash
../sarcgraph-env/bin/python scripts/run_image_metrics.py --config configs/default.yaml --image-id 2.007-1 --write-all
```

Three-image batch smoke:

```bash
../sarcgraph-env/bin/python scripts/run_batch_metrics.py --config configs/default.yaml --limit 3 --write-tables --write-provenance --continue-on-error
```

Full batch:

```bash
../sarcgraph-env/bin/python scripts/run_batch_metrics.py --config configs/default.yaml --write-tables --write-provenance --continue-on-error
```

Audit:

```bash
../sarcgraph-env/bin/python scripts/audit_batch_outputs.py --config configs/default.yaml
```

## Current Test Count

`66 passed`

## Full Batch Audit Numbers

- Expected images: `145`
- Processed images: `145`
- OK: `145`
- Errors: `0`
- Per-image rows: `145`
- Per-patch rows: `32625`
- Donors represented: `29`
- Valid spacing patch rows: `338`
- Mean runtime/image: `2.21 s`

## Key Output Files

- `results/tables/manifest.csv`
- `results/tables/per_image_metrics.csv`
- `results/tables/per_patch_metrics.csv`
- `results/tables/batch_run_summary.csv`
- `results/tables/batch_audit_summary.json`
- `results/tables/batch_audit_summary.txt`
- `results/provenance/{image_id}_run_provenance.json`
- `results/previews/{image_id}_tissue_mask_overlay.png`
- `results/previews/{image_id}_orientation.png`
- `results/previews/{image_id}_coherence.png`
- `results/previews/{image_id}_oop_heatmap.png`
- `results/previews/{image_id}_spacing_heatmap.png`

## Schema Locks

Schema constants live in `src/sarcomere_analysis/schemas.py`:

- `MANIFEST_COLUMNS`: manifest identity, source path, calibration-derived spacing columns.
- `PATCH_METRICS_COLUMNS`: patch coordinates, QC, orientation/OOP, spacing scaffold outputs.
- `IMAGE_METRICS_COLUMNS`: image-level tissue fraction, OOP, spacing summaries.
- `BATCH_RUN_SUMMARY_COLUMNS`: per-image batch status, errors, runtime, output paths.

Writers fill missing optional columns before writing and preserve deterministic column order. Required core columns fail loudly.

## Known Limitations

- The spacing scaffold is conservative and preliminary.
- Many spacing values are `NaN` by design when confidence is weak or QC fails.
- No FIJI/manual sarcomere-length validation has been performed yet.
- No clinical statistics have been performed.
- Per-patch rows are not independent biological samples.
- OOP and spacing are computational metrics, not biological conclusions yet.

## Next Recommended Implementation Steps

1. Inspect output distributions and preview gallery for QC sanity.
2. Improve visual QC gallery/index, not algorithms.
3. Build FIJI validation harness for manual sarcomere length.
4. Only after validation: feature aggregation/per-donor tables.
5. Only after aggregation: statistics.
6. Only after core validation: benchmarking tools.
