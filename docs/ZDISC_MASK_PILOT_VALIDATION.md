# Z-Disc Mask Pilot Validation

This module compares manually drawn Z-disc/striation mask summaries against existing automated patch-level OOP/orientation metrics for the same crops.

It is a pilot validation signal only. It is not clinical analysis, not final expert validation, not a publication figure workflow, and not a replacement for the frozen production pipeline.

## Inputs

- `results/zdisc_annotation/zdisc_annotation_features.csv`
- `results/zdisc_annotation/zdisc_annotation_index.csv`
- `results/tables/features_per_patch.csv`

Rows are joined by `image_id` and `patch_id`. `donor_id` is preserved as a string and used as a consistency check.

## Command

```bash
../sarcgraph-env/bin/python scripts/validate_zdisc_masks.py --config configs/default.yaml
```

Optional:

```bash
../sarcgraph-env/bin/python scripts/validate_zdisc_masks.py \
  --config configs/default.yaml \
  --min-n-for-correlation 10
```

## Outputs

- `results/validation/zdisc_mask_validation_matched.csv`
- `results/validation/zdisc_mask_validation_summary.json`
- `results/validation/zdisc_mask_validation_summary.txt`

## Metrics

The matched table keeps one row per manual annotation mask and adds automated patch metrics where a matching patch exists.

The summary reports:

- matching counts and donor mismatch counts
- number of masks with visible Z-disc labels
- number of orientation-estimable manual masks
- axial orientation error for estimable manual masks
- automated patch OOP medians by manual annotation status
- an optional exploratory Spearman association between manual `zdisc_pixel_fraction` and automated `patch_oop`

No p-values are used for group comparisons. No disease or clinical comparisons are performed.

## Orientation Agreement

Manual orientation comes from the rough PCA proxy over label-1 mask pixels. Automated orientation comes from `patch_mean_orientation_deg`.

Axial angular error is:

```text
min(abs(a - b), 180 - abs(a - b))
```

after treating angles as axial 0-180 degree quantities.

## Cautions

- This is pilot validation only.
- The number of orientation-estimable masks is small.
- The manual masks were drawn by the user, not an independent blinded expert.
- These outputs should not be treated as final publication validation.
- Empty masks are valid negative/unclear examples.
- This does not validate spacing.
- Spacing remains `exploratory_low_yield`.
