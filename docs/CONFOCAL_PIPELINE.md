# Confocal Pipeline

This command is a confocal-first orchestration wrapper. It runs the current selected confocal workflow end to end using the primary `moderate` confident-striation gate and writes consolidated outputs.

It does not introduce a new segmentation algorithm, tune thresholds, change widefield outputs, adopt `moderate_relaxed_combined` as primary, or create publication figures.

## Command

```bash
../sarcgraph-env/bin/python scripts/run_confocal_pipeline.py \
  --config configs/default.yaml \
  --confocal-root /path/to/local/confocal \
  --output-dir results/confocal_pipeline \
  --write-previews
```

## Workflow

1. Discover confocal images and write a manifest.
2. Extract per-image pixel calibration from image metadata.
3. Compute the confocal confident-striation mask and sensitivity variants.
4. Apply the primary `moderate` gate.
5. Compute same-grid OOP/orientation on the selected grid.
6. Compute calibrated spacing in selected moderate candidate regions only.
7. Write consolidated per-patch and per-image tables.
8. Write preview overlays if requested.
9. Write a summary JSON/TXT report.

## Outputs

Outputs are written under the selected `--output-dir`:

- `confocal_pipeline_manifest.csv`
- `confocal_pipeline_per_patch.csv`
- `confocal_pipeline_per_image.csv`
- `confocal_pipeline_summary.json`
- `confocal_pipeline_summary.txt`
- `previews/`, when `--write-previews` is used

Intermediate step outputs are kept under `_intermediate/` inside the same output directory so older confocal audit folders are not overwritten.

## Gate Policy

- Primary gate: `moderate`
- Secondary/sensitivity gate: `moderate_relaxed_combined`
- `moderate_relaxed_combined` is not used for primary summaries.
- Relaxed-gate spacing should not be reported unless a separate refreshed audit is explicitly run and reviewed.

## Calibration Policy

Micron spacing uses per-image confocal calibration only. The widefield pixel size is never used as a fallback. Images without valid per-image calibration have micron spacing disabled and are reported clearly.

## Interpretation

Confocal selected-region OOP and calibrated spacing are exploratory and manual-review-needed. This wrapper improves reproducibility of the current confocal pilot workflow; it does not create clinical, disease-comparison, or biological claims.
