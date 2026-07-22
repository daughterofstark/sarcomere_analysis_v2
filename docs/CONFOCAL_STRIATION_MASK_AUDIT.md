# Confocal Striation Mask Audit

This module creates an exploratory candidate mask for confocal regions likely to contain confident striated/Z-disc signal.

It answers Natalia's feasibility question in a deliberately conservative way: yes, it is feasible to restrict analysis to candidate confident-striation regions, but the current output is a QC/audit mask, not a validated biological segmentation.

## Why This Exists

The frozen widefield QC gate barely admitted confocal patches in the baseline audit. Confocal images have different contrast, resolution, noise, and local tissue quality, so the widefield patch-validity rules are not transferable unchanged.

This module therefore creates a separate confocal-only candidate mask using local classical evidence:

- tissue/signal fraction
- local gradient energy
- local orientation coherence
- local intensity contrast
- local intensity standard deviation
- saturation rejection

No diagnosis, health label, or expected-positive label is used to decide candidate status.

## Run

```bash
../sarcgraph-env/bin/python scripts/run_confocal_striation_mask_audit.py \
  --config configs/default.yaml \
  --confocal-manifest results/confocal_baseline/confocal_manifest.csv \
  --write-previews
```

Optional parameters:

```bash
--patch-size
--stride
--min-gradient-energy
--min-orientation-coherence
--min-intensity-std
--max-saturation-fraction
```

These are audit parameters only. They do not change the frozen widefield pipeline.

## Outputs

Outputs are written under:

```text
results/confocal_striation_mask/
```

Files:

- `confocal_striation_mask_per_patch.csv`
- `confocal_striation_mask_per_image.csv`
- `confocal_striation_mask_summary.json`
- `confocal_striation_mask_summary.txt`
- optional previews under `results/confocal_striation_mask/previews/`

Preview types:

- normalized image
- candidate mask overlay
- patch grid with candidate patches highlighted
- rejection map

## Images To Inspect

Natalia identified:

- `6052-CLEAR_STRIPES.tif` and `5138.tif` as expected positive examples
- `3112.tif` as a complex image where Z-disc-like structures may not form striations

The audit reports these separately, but it does not hard-code them to pass.

## Calibration

The confocal pixel size is currently unknown. This module operates in pixels only and does not report sarcomere length in microns.

## Interpretation

This is not a validated Z-disc segmentation, not a spacing endpoint, and not a biological result. It is a candidate-region audit to decide whether future analysis should focus on confident striated regions rather than whole-image or widefield-style patch gates.
