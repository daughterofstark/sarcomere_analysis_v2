# Sarcomere Analysis

Classical image-analysis tools for sarcomeric organisation audits in myocardium microscopy images.

This repository contains code, tests, and documentation only. Raw microscopy images and generated analysis results are intentionally not included.

Current scope includes widefield preprocessing/QC/orientation/OOP, exploratory spacing scaffolds, confocal pilot selected-region audits, metadata/provenance helpers, annotation-pack utilities, and validation-status reporting.

Not included: raw image data, generated result tables/previews/review zips, clinical/statistical inference, ML, cell segmentation, or publication figures.

## Start Here

- [Today handoff](docs/TODAY_HANDOFF.md)
- [Runbook](docs/RUNBOOK.md)
- [Run classical pipeline](docs/RUN_CLASSICAL_PIPELINE.md)
- [Metric definitions](docs/metric_definitions.md)
- [Data availability and sharing](docs/DATA_AVAILABILITY_AND_SHARING.md)
- [Share-ready audit](docs/SHARE_READY_AUDIT.md)
- [Next Codex prompts](docs/NEXT_CODEX_PROMPTS.md)

## Data And Paths

Raw microscopy images are external local inputs. Set local paths in a private config copy or pass CLI overrides such as `--image-dir` or `--confocal-root`.

The default config uses share-safe placeholders/relative outputs:

- widefield input placeholder: `/path/to/local/widefield/raw`
- generated outputs: `results/`

Do not commit raw image files, generated `results/`, review-pack zip files, or local environment folders.

## Analysis Status

Widefield and confocal workflows are separate.

- Widefield: production measurement pipeline is frozen; spacing was low-yield and remains exploratory.
- Confocal pilot: selected-region OOP/coherence and calibrated spacing were promising in visually reviewed regions, but remain exploratory/manual-review-dependent.

No clinical, disease-comparison, or biological claims are made from this repository alone.

## Common Commands

```bash
../sarcgraph-env/bin/python -m pytest
PYTHON=../sarcgraph-env/bin/python bash scripts/dev_smoke.sh
../sarcgraph-env/bin/python scripts/build_manifest.py --config configs/default.yaml --image-dir /path/to/local/widefield/raw
../sarcgraph-env/bin/python scripts/run_image_metrics.py --config configs/default.yaml --image-id 2.007-1 --write-all
../sarcgraph-env/bin/python scripts/run_batch_metrics.py --config configs/default.yaml --write-tables --write-provenance --continue-on-error
../sarcgraph-env/bin/python scripts/audit_batch_outputs.py --config configs/default.yaml
../sarcgraph-env/bin/python scripts/generate_qc_previews.py --config configs/default.yaml --continue-on-error
../sarcgraph-env/bin/python scripts/build_qc_gallery.py --config configs/default.yaml --write-index --write-html
../sarcgraph-env/bin/python scripts/run_classical_pipeline.py --config configs/default.yaml --skip-existing
```

Raw TIFFs are external local inputs and must not be copied into this repository. Per-patch rows are computational QC/measurement rows, not independent biological samples.

The QC gallery indexes existing preview PNGs for human inspection. It is diagnostic navigation only, not publication figure generation.
Preview generation recomputes maps for visualization but does not change saved metric tables; scientific interpretation still requires FIJI validation.
