# Full-Image Patch Mask Validation

This module performs pilot local validation by cropping sparse full-image manual Z-disc/striation masks into the same patch windows used by the frozen automated patch pipeline.

It is validation scaffolding only. It does not change production OOP, orientation, spacing, feature, or analysis tables.

## Inputs

- `results/full_image_zdisc_annotation/full_image_annotation_index.csv`
- `results/full_image_zdisc_annotation/masks/`
- `results/tables/features_per_patch.csv`

Rows are matched by `image_id` and `patch_id`. `donor_id` is preserved as a string and checked for consistency.

## Command

```bash
../sarcgraph-env/bin/python scripts/validate_full_image_patch_masks.py --config configs/default.yaml
```

Optional:

```bash
../sarcgraph-env/bin/python scripts/validate_full_image_patch_masks.py \
  --config configs/default.yaml \
  --min-zdisc-pixels 10 \
  --min-n-for-correlation 10
```

## Outputs

- `results/validation/full_image_patch_mask_validation_matched.csv`
- `results/validation/full_image_patch_mask_validation_summary.json`
- `results/validation/full_image_patch_mask_validation_summary.txt`

## Manual Patch Features

For each annotated full-image mask, the module crops the mask to each automated patch window from `features_per_patch.csv` and computes:

- label-1 Z-disc pixel count and fraction
- label-2 ignore pixel count and fraction
- patch annotation status: `empty`, `zdisc_labeled`, `ignore_only`, or `mixed`
- rough manual patch orientation from label-1 pixels when enough pixels are present

Label 2 is treated as ignore/uncertain, not as Z-disc signal.

## Agreement Summaries

For patches where manual orientation is estimable and automated patch orientation is finite, axial angular error is computed as:

```text
min(abs(a - b), 180 - abs(a - b))
```

The summary also reports automated `patch_oop` medians by manual patch status. Group medians are descriptive only; no p-values or clinical statistics are computed.

If enough matched rows are available, a Spearman association between manual Z-disc pixel fraction and automated patch OOP may be reported as exploratory pilot context. It is not a validation claim or threshold-selection rule.

## Interpretation

Patch-level comparison is more appropriate than full-image orientation comparison for these sparse manual labels because the user drew visible local structures, not exhaustive global segmentation.

Cautions:

- This is pilot/local validation only.
- Manual masks were drawn by the user, not an independent blinded expert.
- Full-image masks are sparse annotations, not exhaustive segmentation.
- No clinical or biological claims are made here.
- This does not validate spacing.
- Spacing remains `exploratory_low_yield`.
