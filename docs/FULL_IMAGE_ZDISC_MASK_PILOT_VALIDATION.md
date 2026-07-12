# Full-Image Z-Disc Mask Pilot Validation

This module compares sparse manually drawn full-image Z-disc/striation mask summaries against existing automated image-level OOP/orientation metrics.

It is pilot validation only. It is not clinical analysis, not final expert validation, not disease comparison, not a publication figure workflow, and not a replacement for the frozen production pipeline.

## Inputs

- `results/full_image_zdisc_annotation/full_image_zdisc_annotation_features.csv`
- `results/tables/features_per_image.csv`
- optionally `results/tables/analysis_per_image.csv` as metadata in later workflows

Rows are joined by `image_id`. `donor_id` is preserved as a string and used as a consistency check.

## Command

```bash
../sarcgraph-env/bin/python scripts/validate_full_image_zdisc_masks.py --config configs/default.yaml
```

Optional:

```bash
../sarcgraph-env/bin/python scripts/validate_full_image_zdisc_masks.py \
  --config configs/default.yaml \
  --min-n-for-correlation 10
```

## Outputs

- `results/validation/full_image_zdisc_mask_validation_matched.csv`
- `results/validation/full_image_zdisc_mask_validation_summary.json`
- `results/validation/full_image_zdisc_mask_validation_summary.txt`

## Metrics

The matched table keeps one row per manual full-image annotation and adds automated image-level metrics where a matching image exists.

The summary reports:

- matching counts and donor mismatch counts
- number of images with visible Z-disc labels
- number of orientation-estimable masks
- axial orientation error for estimable manual masks
- automated image OOP medians by manual annotation status
- optional exploratory Spearman association between manual `zdisc_pixel_fraction` and automated `image_oop`

No p-values are used for group comparisons. No disease or clinical comparisons are performed.

## Orientation Agreement

Manual orientation comes from the rough PCA proxy over label-1 full-image mask pixels. Automated orientation comes from `image_mean_orientation_deg`.

Axial angular error is:

```text
min(abs(a - b), 180 - abs(a - b))
```

after treating angles as axial 0-180 degree quantities.

## Cautions

- This is pilot validation only.
- The number of orientation-estimable images is small.
- The manual masks were drawn by the user, not an independent blinded expert.
- Full-image masks are sparse annotations, not exhaustive Z-disc segmentation.
- These outputs should not be treated as final publication validation.
- Empty masks are valid negative or unclear examples.
- This does not validate spacing.
- Spacing remains `exploratory_low_yield`.
