# Sarcomere Analysis

Minimal classical image-analysis pipeline for archival human myocardium TIFFs.

Current scope: manifest/IO, preprocessing, tissue mask/QC, patch grid, structure-tensor orientation/OOP, conservative spacing scaffold, standardized outputs, provenance, single-image CLI, batch CLI, schema locks, and smoke tests.

Not implemented: FIJI validation, benchmarking, clinical/statistical inference, ML, cell segmentation, or publication figures.

## Start Here

- [Today handoff](docs/TODAY_HANDOFF.md)
- [Runbook](docs/RUNBOOK.md)
- [Run classical pipeline](docs/RUN_CLASSICAL_PIPELINE.md)
- [Metric definitions](docs/metric_definitions.md)
- [Next Codex prompts](docs/NEXT_CODEX_PROMPTS.md)

## Common Commands

```bash
../sarcgraph-env/bin/python -m pytest
PYTHON=../sarcgraph-env/bin/python bash scripts/dev_smoke.sh
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
