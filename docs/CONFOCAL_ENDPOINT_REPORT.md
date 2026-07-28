# Confocal Endpoint Report

This report classifies the larger confocal cohort into endpoint-aware analysis groups. It reads existing confocal pipeline and cohort-audit outputs only; it does not rerun algorithms, tune thresholds, adopt the relaxed gate, or change widefield outputs.

## Cohort Status

- Images processed: 42
- Errors: 0
- Calibrated images: 42
- Endpoint class counts: `{'spacing_eligible_low_confidence': 25, 'spacing_eligible_moderate': 7, 'low_candidate_review_needed': 5, 'oop_only_spacing_low_yield': 5}`
- OOP-reportable images: 41
- Spacing-reportable images: 7
- OOP-only images: 5
- Review-needed images: 30

## Decision Rules

- `oop_reportable`: per-image calibration exists and at least 25 selected candidate patches are available.
- `spacing_reportable`: at least 10 valid selected spacing patches and selected spacing valid fraction at least 0.10.
- `spacing_eligible_moderate`: spacing is reportable under the current endpoint rule.
- `spacing_eligible_low_confidence`: at least 5 valid spacing patches exist, but the fraction is below 0.10.
- `oop_only_spacing_low_yield`: OOP/coherence is available, but spacing is low-yield.
- `low_candidate_review_needed`: selected candidate fraction is below 0.05 or too few candidate patches are available.
- `failed_or_unusable`: processing error or missing per-image calibration.

## Interpretation

- The 42-image confocal pipeline ran successfully.
- OOP/coherence endpoint is broadly available for images with sufficient selected candidate patches.
- Spacing endpoint is available only for a subset of images meeting both count and fraction rules.
- No images met the robust spacing threshold in the cohort audit; seven met moderate spacing criteria.
- Spacing should not be treated as a universal endpoint for the larger confocal cohort.
- Downstream analysis should be endpoint-aware: OOP/coherence across the broader cohort, spacing only in spacing-eligible images/regions.
- No disease, clinical, threshold-tuning, or biological claims are made by this endpoint report.

## Spacing-Reportable Images

- `8A793.tif`: `spacing_eligible_moderate`, spacing fraction `0.1283185840707964`, valid spacing patches `29`
- `94217.tif`: `spacing_eligible_moderate`, spacing fraction `0.1317073170731707`, valid spacing patches `54`
- `B23E3_1.tif`: `spacing_eligible_moderate`, spacing fraction `0.1391304347826087`, valid spacing patches `32`
- `E0ABF.tif`: `spacing_eligible_moderate`, spacing fraction `0.2024691358024691`, valid spacing patches `82`
- `E0ABF_1.tif`: `spacing_eligible_moderate`, spacing fraction `0.171195652173913`, valid spacing patches `63`
- `E0ABF_2.tif`: `spacing_eligible_moderate`, spacing fraction `0.1653543307086614`, valid spacing patches `105`
- `EB98A_1.tif`: `spacing_eligible_moderate`, spacing fraction `0.1311475409836065`, valid spacing patches `32`

## OOP-Only Images

- `10763_2.tif`: `oop_only_spacing_low_yield`, spacing fraction `0.0`, valid spacing patches `0`
- `31331_2.tif`: `oop_only_spacing_low_yield`, spacing fraction `0.1320754716981132`, valid spacing patches `7`
- `7156A_2.tif`: `oop_only_spacing_low_yield`, spacing fraction `0.0169491525423728`, valid spacing patches `1`
- `899B1.tif`: `oop_only_spacing_low_yield`, spacing fraction `0.0135135135135135`, valid spacing patches `4`
- `C970D_2.tif`: `oop_only_spacing_low_yield`, spacing fraction `0.0227272727272727`, valid spacing patches `3`

## Review-Needed Images

- `10763.tif`: `low_candidate_review_needed`, spacing fraction `0.0975609756097561`, valid spacing patches `4`
- `10763_1.tif`: `low_candidate_review_needed`, spacing fraction `0.0`, valid spacing patches `0`
- `31331.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.078125`, valid spacing patches `10`
- `31331_1.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0494505494505494`, valid spacing patches `9`
- `5155D.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.073469387755102`, valid spacing patches `18`
- `5155D_1.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0509803921568627`, valid spacing patches `13`
- `5155D_2.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0295698924731182`, valid spacing patches `11`
- `7156A.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0636363636363636`, valid spacing patches `14`
- `7156A_1.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0397350993377483`, valid spacing patches `6`
- `72ADB.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0614886731391585`, valid spacing patches `19`
- `72ADB_1.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0509708737864077`, valid spacing patches `21`
- `72ADB_2.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0453563714902807`, valid spacing patches `21`
- `899B1_1.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0408163265306122`, valid spacing patches `14`
- `899B1_2.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.044280442804428`, valid spacing patches `12`
- `8A793_1.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0656934306569343`, valid spacing patches `9`
- `8A793_2.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0649350649350649`, valid spacing patches `5`
- `94217_1.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0972972972972973`, valid spacing patches `36`
- `94217_2.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0364372469635627`, valid spacing patches `18`
- `B23E3.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.045045045045045`, valid spacing patches `15`
- `B23E3_2.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.057788944723618`, valid spacing patches `23`
- `BF793.tif`: `low_candidate_review_needed`, spacing fraction `0.1`, valid spacing patches `1`
- `BF793_1.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0484848484848484`, valid spacing patches `8`
- `BF793_2.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0602409638554216`, valid spacing patches `5`
- `C970D.tif`: `low_candidate_review_needed`, spacing fraction `0.048780487804878`, valid spacing patches `2`
- `C970D_1.tif`: `low_candidate_review_needed`, spacing fraction `0.03125`, valid spacing patches `1`
- `CB8A5.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0504451038575667`, valid spacing patches `17`
- `CB8A5_1.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0325443786982248`, valid spacing patches `11`
- `CB8A5_2.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.025`, valid spacing patches `11`
- `EB98A.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0818505338078291`, valid spacing patches `23`
- `EB98A_2.tif`: `spacing_eligible_low_confidence`, spacing fraction `0.0395348837209302`, valid spacing patches `17`

## Recommendation

Use OOP/coherence as the broad confocal endpoint family and restrict spacing summaries to spacing-reportable images/regions. Next action should be visual review of spacing-eligible and low-yield examples, plus more confocal images if spacing is intended as a downstream endpoint.
