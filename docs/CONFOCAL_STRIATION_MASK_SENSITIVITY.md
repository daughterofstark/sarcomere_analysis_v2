# Confocal Striation Mask Sensitivity

This module audits whether a stricter confocal confident-striation candidate gate is plausible using the existing patch features from `results/confocal_striation_mask/confocal_striation_mask_per_patch.csv`.

It does not recompute image measurements, tune production thresholds, change the widefield pipeline, or make biological claims.

## Inputs

- `results/confocal_striation_mask/confocal_striation_mask_per_patch.csv`
- `results/confocal_striation_mask/confocal_striation_mask_per_image.csv`

The audit uses existing patch-level values:

- `gradient_energy`
- `orientation_coherence`
- `intensity_std`
- `contrast`
- `signal_fraction`
- `saturation_fraction`

## Variants

The grid includes:

- `lenient`
- `default_current`
- `moderate`
- `strict`
- `very_strict`

The default/current variant mirrors the previous broad gate. The stricter variants use quantiles from the confocal patch table. This is sensitivity/QC only, not final tuning.

## Classification

Variants are classified as:

- `too_broad`: median candidate fraction is near whole-image, or many images are above 0.90 candidate fraction
- `too_sparse`: expected positive examples lose nearly all candidate patches
- `plausible_for_review`: expected positives retain meaningful candidate fractions, 3112 remains lower, and the median candidate fraction is not near whole-image
- `uninformative_low_yield`: none of the above, but not clearly useful

The expected positives `5138` and `6052-CLEAR_STRIPES`, and the complex example `3112`, are used only for reporting/classification. They are not hard-coded to pass.

## Run

```bash
../sarcgraph-env/bin/python scripts/run_confocal_striation_sensitivity.py \
  --config configs/default.yaml \
  --write-previews
```

Optional inputs:

```bash
--patch-table
--image-table
--output-dir
--max-preview-variants
```

## Outputs

Outputs are written under:

```text
results/confocal_striation_sensitivity/
```

Files:

- `confocal_striation_sensitivity_variants.csv`
- `confocal_striation_sensitivity_per_image.csv`
- `confocal_striation_sensitivity_summary.json`
- `confocal_striation_sensitivity_summary.txt`
- optional previews under `results/confocal_striation_sensitivity/previews/`

## Interpretation

This audit asks whether a more selective confident-region gate is plausible. It does not validate a segmentation, does not report sarcomere length in microns, and does not change any production output.
