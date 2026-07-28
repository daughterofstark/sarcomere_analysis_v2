# Confocal Freeze Report

Generated: `2026-07-28T12:33:43.206875+00:00`

## Frozen Decision

- Primary gate: `moderate`
- Relaxed gate: `moderate_relaxed_combined_sensitivity_only_not_primary`
- Widefield calibration used: `False`

## Larger Dataset

- Images: 42
- Errors: 0
- Calibrated images: 42
- Total patches: 40362
- Selected candidate patches: 10546
- Valid selected spacing patches: 786
- Selected spacing valid fraction: 0.0745306277261521
- Median selected OOP: 0.7068473614369967
- Median selected spacing: 2.239569763717925 um

## Endpoint Result

- OOP reportable: 41/42
- Spacing reportable: 7/42
- Failed/unusable: 0
- Endpoint class counts: `{'spacing_eligible_moderate': 7, 'spacing_eligible_low_confidence': 25, 'oop_only_spacing_low_yield': 5, 'low_candidate_review_needed': 5, 'failed_or_unusable': 0}`

Spacing-reportable images:
- `8A793.tif`
- `94217.tif`
- `B23E3_1.tif`
- `E0ABF.tif`
- `E0ABF_1.tif`
- `E0ABF_2.tif`
- `EB98A_1.tif`

## Manual Visual Spot-Check

- Reviewed in-chat: 6/7
- Reviewed acceptable: 6/6
- `E0ABF.tif` was not manually reviewed in-chat and remains algorithmic/pending visual confirmation.
- `94217.tif` passed with a broad-selection caveat.
- `EB98A_1.tif` passed with a not-perfect caveat.

## Final Frozen Interpretation

- Widefield spacing remains low-yield/negative.
- Widefield OOP is not validated as expert organisation.
- Confocal pipeline scales technically.
- Confocal OOP/coherence is broadly reportable.
- Confocal spacing is not universal.
- Confocal spacing is reportable only as a selected-region/subset endpoint for spacing-eligible images.
- Do not run clinical/disease statistics yet.
- Do not use ML yet.
- Next scientific action would be expert review/confirmation or additional labels/images, not more unsupervised tuning.

## Allowed Downstream Use

- Report confocal OOP/coherence broadly with caveats.
- Report spacing only for spacing-reportable selected regions.
- Use endpoint flags in downstream tables.
- Share the endpoint review pack for expert QC.

## Not Allowed Claims

- Whole-cohort spacing claims.
- Disease or clinical inference.
- Relaxed gate as primary.
- Widefield spacing claims.
- Claiming OOP is validated biological organisation.
- ML claims.

## No-Change Statement

This freeze report is documentation/reporting only. It does not rerun analysis, change thresholds, adopt the relaxed gate, modify widefield outputs, or alter production algorithms.
