# Confocal Calibrated Spacing Audit

This module estimates sarcomere/Z-disc spacing in microns for confocal images using per-image confocal pixel calibration.

It is exploratory. It does not change the frozen widefield pipeline, does not tune thresholds, and does not introduce a new spacing algorithm.

## Command

```bash
../sarcgraph-env/bin/python scripts/run_confocal_spacing_audit.py \
  --config configs/default.yaml \
  --write-previews
```

## Inputs

- `results/confocal_metadata/confocal_metadata_calibration.csv`
- `results/confocal_same_grid_oop/confocal_same_grid_oop_per_patch.csv`
- confocal source images referenced by the calibration table

The audit uses only per-image confocal pixel sizes. The widefield pixel size is never used as fallback.

## Method

The existing autocorrelation spacing helper is reused with a per-image configuration where the expected 1.5-2.4 µm spacing band is converted to pixels using that image's pixel size.

Spacing is evaluated primarily inside moderate confident-striation candidate patches from the same 128 px grid used in the confocal selected-region OOP audit. Non-candidate patches are retained in the output table but marked `not_candidate_region`.

## Outputs

Outputs are written under `results/confocal_spacing_audit/`:

- `confocal_spacing_per_patch.csv`
- `confocal_spacing_per_image.csv`
- `confocal_spacing_summary.json`
- `confocal_spacing_summary.txt`
- optional previews under `previews/`

## Interpretation

If the existing estimator produces low yield, that is reported directly. Low-yield spacing remains exploratory and should not become a biological endpoint without manual review. Micron spacing is only valid for images with valid per-image confocal calibration.
