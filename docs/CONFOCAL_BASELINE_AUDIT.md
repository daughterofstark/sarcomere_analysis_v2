# Confocal Baseline Audit

This workflow performs a non-destructive baseline intake of Natalia's confocal pilot images. It is separate from the frozen widefield pipeline outputs.

The goal is to answer a practical question before building anything new: do the confocal images look suitable for the existing orientation/OOP machinery, and is a later confident-striation/Z-disc-region mask likely needed?

## Inputs

Default confocal root:

```text
/path/to/local/confocal
```

Supported files:

- `.tif`
- `.tiff`
- `.png`
- `.jpg`
- `.jpeg`

TIFF microscopy formats are preferred when available.

## Run

```bash
../sarcgraph-env/bin/python scripts/run_confocal_baseline_audit.py \
  --config configs/default.yaml \
  --confocal-root /path/to/local/confocal \
  --write-previews
```

Outputs are written under:

```text
results/confocal_baseline/
```

## Outputs

- `confocal_manifest.csv`
- `confocal_baseline_per_image.csv`
- `confocal_baseline_per_patch.csv`
- `confocal_baseline_summary.json`
- `confocal_baseline_summary.txt`
- optional preview PNGs under `results/confocal_baseline/previews/`

The manifest flags filenames/sample IDs containing:

- `6052` or `5138` as expected positive examples
- `3112` as a noted complex example

## Calibration

The widefield pixel size is not assumed to apply to confocal images.

Orientation/OOP can be computed in pixel coordinates. Calibrated sarcomere spacing in microns is not reported unless a confocal pixel size is explicitly known. In the current baseline audit, spacing is marked as not computed due missing confocal calibration.

## Interpretation

This is a baseline transfer audit only.

No confocal-specific optimisation has been performed. No confident Z-disc mask, segmentation model, threshold tuning, clinical analysis, or biological interpretation is implemented here.

The output is meant to guide the next decision: whether to create a confident-striation/Z-disc-region mask for confocal images before attempting validated organisation or spacing endpoints.
