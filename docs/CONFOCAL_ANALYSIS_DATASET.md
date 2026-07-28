# Confocal Analysis Dataset

This is the downstream-safe confocal table and manual endpoint-QC template. It packages existing frozen outputs only; it does not rerun algorithms, tune thresholds, adopt the relaxed gate, or change widefield outputs.

## Outputs

- `results/confocal_analysis_dataset/confocal_analysis_per_image.csv`
- `results/confocal_analysis_dataset/confocal_manual_review_template.csv`
- `results/confocal_analysis_dataset/confocal_analysis_dataset_summary.json`
- `results/confocal_analysis_dataset/confocal_analysis_dataset_summary.txt`

## Summary

- Images total: 42
- OOP allowed count: 41
- Spacing allowed count: 7
- Review template rows: 17

Spacing-reportable images:
- `8A793.tif`
- `94217.tif`
- `B23E3_1.tif`
- `E0ABF.tif`
- `E0ABF_1.tif`
- `E0ABF_2.tif`
- `EB98A_1.tif`

## Rules

- `oop_value_allowed_for_downstream` follows `oop_reportable`.
- `spacing_value_allowed_for_downstream` follows `spacing_reportable`.
- Non-reportable spacing values are retained but marked with `spacing_downstream_warning = not_reportable_endpoint_low_yield`.
- Reportable spacing values remain selected-region/subset endpoints and are marked `selected_region_spacing_only`.

## Manual Review Template

The manual review template is for endpoint QC, not ML training. Allowed response values:
- `selected_regions_valid`: yes / partial / no / unclear
- `valid_spacing_patches_valid`: yes / partial / no / unclear / not_applicable
- `image_suitable_for_spacing`: yes / no / unclear
- `image_suitable_for_oop`: yes / no / unclear
- `reviewer_confidence`: 1 / 2 / 3 / 4 / 5

## Caveats

- This is a downstream-safe data packaging layer only.
- Endpoint flags must be respected in downstream work.
- OOP/coherence is broad; spacing is subset/selected-region only.
- Spacing numeric values are retained even when not reportable, with warning flags controlling interpretation.
- No disease, clinical, statistical, ML, or biological inference is performed here.
- The manual review template is for endpoint QC, not ML training.
