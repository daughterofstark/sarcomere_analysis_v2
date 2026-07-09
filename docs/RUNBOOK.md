# Runbook

## Environment

Run commands from:

```bash
/Users/medhasharma/sarcomere_tools/sarcomere-analysis
```

Use the project environment:

```bash
../sarcgraph-env/bin/python
```

## Outputs

Generated outputs go under `results/`:

- `results/tables/`
- `results/previews/`
- `results/provenance/`

Raw TIFFs remain external local inputs and are not copied.

## One Image

```bash
../sarcgraph-env/bin/python scripts/run_image_metrics.py --config configs/default.yaml --image-id 2.007-1 --write-all
```

## Full Batch

```bash
../sarcgraph-env/bin/python scripts/run_batch_metrics.py --config configs/default.yaml --write-tables --write-provenance --continue-on-error
```

Do not add `--write-preview` for full batch unless debugging; previews are large.

## Audit Batch

```bash
../sarcgraph-env/bin/python scripts/audit_batch_outputs.py --config configs/default.yaml
```

Audit files:

- `results/tables/batch_audit_summary.json`
- `results/tables/batch_audit_summary.txt`

## QC Gallery

Generate missing preview PNGs:

```bash
../sarcgraph-env/bin/python scripts/generate_qc_previews.py --config configs/default.yaml --continue-on-error
```

Build an index of existing preview PNGs:

```bash
../sarcgraph-env/bin/python scripts/build_qc_gallery.py --config configs/default.yaml --write-index --write-html
```

Outputs:

- `results/tables/qc_gallery_index.csv`
- `results/qc_gallery/index.html`

Open `results/qc_gallery/index.html` in a browser after rebuilding the gallery.

The QC gallery is for diagnostic inspection of masks, orientation maps, OOP heatmaps, and spacing heatmaps. It is not publication figure generation. Preview generation recomputes maps for visualization but does not change saved metric tables. Scientific interpretation still requires FIJI validation.

## Spacing Diagnostics

Run diagnostic reporting for the current spacing scaffold:

```bash
../sarcgraph-env/bin/python scripts/diagnose_spacing.py --config configs/default.yaml --write-summary
```

For a smaller smoke check with per-patch diagnostic CSVs:

```bash
../sarcgraph-env/bin/python scripts/diagnose_spacing.py --config configs/default.yaml --limit 5 --write-summary --write-patch-diagnostics
```

Outputs:

- `results/diagnostics/spacing_diagnostic_summary.csv`
- `results/diagnostics/spacing_diagnostic_by_image.csv`
- `results/diagnostics/{image_id}_spacing_patch_diagnostics.csv` when `--write-patch-diagnostics` is used

Spacing diagnostics expose current internal quantities such as expected lag bounds, selected lag, peak score, autocorrelation baseline, confidence, and rejection stage. They do not validate spacing. Repeated accepted spacing values near one bound require algorithm review before FIJI validation or biological interpretation.

To write a single autocorrelation debug plot for one patch:

```bash
../sarcgraph-env/bin/python scripts/diagnose_spacing.py --config configs/default.yaml --limit 1 --write-summary --debug-image-id 2.007-1 --debug-patch-id 2.007-1_p00000 --write-autocorr-debug
```

## Development Smoke

```bash
PYTHON=../sarcgraph-env/bin/python bash scripts/dev_smoke.sh
```

## Clean Generated Outputs Safely

Generated outputs are ignored by git. To clean local generated outputs, remove subfolders under `results/` only:

```bash
rm -rf results/tables results/previews results/provenance
```

Do not remove raw TIFF directories.

## Do Not Commit

- Raw TIFFs
- Virtual environments
- Generated `results/tables/`, `results/previews/`, or `results/provenance/`
- NPZ/cache outputs
- Scratch notebooks or temporary files
