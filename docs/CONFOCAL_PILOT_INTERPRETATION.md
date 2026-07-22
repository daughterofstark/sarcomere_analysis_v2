# Confocal Pilot Interpretation

This report consolidates existing confocal pilot outputs only. It does not change algorithms, thresholds, widefield outputs, or production tables.

## 1. Confocal Dataset Intake

- Images processed: 11
- Confocal files present: 11 TIFFs, not 10.
- Processed OK: 11
- Processed errors: 0
- Expected positives: [{'confocal_image_id': '5138', 'filename': '5138.tif'}, {'confocal_image_id': '6052-CLEAR_STRIPES', 'filename': '6052-CLEAR_STRIPES.tif'}]
- Complex example: [{'confocal_image_id': '3112', 'filename': '3112.tif'}]

## 2. Baseline Transfer Audit

- Baseline patch rows: 2475
- Valid orientation patch count total: 2
- Median valid orientation patch fraction: 0.0
- Conclusion: `widefield_qc_not_transferable_unchanged`
- Existing widefield patch QC barely admitted confocal patches (2/2475), so widefield QC should not be transferred unchanged.

## 3. Selective Confident-Striation Mask

- Default gate assessment: `too_broad`
- Why default was too broad: The default/current gate was too broad because median candidate fraction or many per-image candidate fractions were near whole-image.
- Moderate gate classification: `plausible_for_review`
- Candidate fractions: {'5138': 0.3215400624349636, '6052_CLEAR_STRIPES': 0.2924037460978148, '3112': 0.0967741935483871, '7028': 0.6222684703433923}
- Selected variant: `moderate`
- Selected candidate patch count: 2330

## 4. Same-Grid Selected-Region OOP

- Same-grid patch rows: 10571
- Patches processed OK/errors: 10571/0
- Candidate patch count: 2330
- Selected-vs-all OOP summary: {'median_selected_region_oop_128': 0.7084587633058819, 'median_all_region_oop_128': 0.6139508754227327, 'median_selected_vs_all_oop_difference_128': 0.0653476819313511, 'median_selected_region_coherence_128': 0.677074134349823, 'median_all_region_coherence_128': 0.6394568979740143}
- Selective coherence/gradient summary: {'median_selected_region_coherence': 0.7259525954723358, 'median_all_region_coherence': 0.6811765432357788, 'median_selected_region_gradient_energy': 0.0053753014653921, 'median_all_region_gradient_energy': 0.0030924435704946, 'median_selected_region_oop': None, 'median_all_region_oop': None, 'median_selected_vs_all_oop_difference': None, 'oop_available': False}
- 5138: {'confocal_image_id': '5138', 'filename': '5138.tif', 'total_patches': 961, 'candidate_patch_count': 309, 'candidate_patch_fraction': 0.3215400624349636, 'selected_region_median_oop_128': 0.8391437533025475, 'selected_region_iqr_oop_128': 0.0841178830295195, 'all_region_median_oop_128': 0.7773011702430936, 'all_region_iqr_oop_128': 0.20121673319534594, 'selected_vs_all_oop_difference_128': 0.06184258305945389, 'selected_region_median_orientation_valid_pixels_128': 2892.0, 'all_region_median_orientation_valid_pixels_128': 2475.0, 'selected_region_median_coherence_128': 0.6879969239234924, 'all_region_median_coherence_128': 0.6394568979740143, 'expected_positive_example': True, 'noted_complex_example': False, 'interpretation_flag': 'review_needed'}
- 6052-CLEAR_STRIPES: {'confocal_image_id': '6052-CLEAR_STRIPES', 'filename': '6052-CLEAR_STRIPES.tif', 'total_patches': 961, 'candidate_patch_count': 281, 'candidate_patch_fraction': 0.2924037460978148, 'selected_region_median_oop_128': 0.6792985573540838, 'selected_region_iqr_oop_128': 0.1733039985059588, 'all_region_median_oop_128': 0.6139508754227327, 'all_region_iqr_oop_128': 0.24785953896257473, 'selected_vs_all_oop_difference_128': 0.0653476819313511, 'selected_region_median_orientation_valid_pixels_128': 2726.0, 'all_region_median_orientation_valid_pixels_128': 2317.0, 'selected_region_median_coherence_128': 0.7574900984764099, 'all_region_median_coherence_128': 0.7264613211154938, 'expected_positive_example': True, 'noted_complex_example': False, 'interpretation_flag': 'review_needed'}
- 3112: {'confocal_image_id': '3112', 'filename': '3112.tif', 'total_patches': 961, 'candidate_patch_count': 93, 'candidate_patch_fraction': 0.0967741935483871, 'selected_region_median_oop_128': 0.6833980237614709, 'selected_region_iqr_oop_128': 0.13792176827289793, 'all_region_median_oop_128': 0.6762861735361223, 'all_region_iqr_oop_128': 0.2181148761385916, 'selected_vs_all_oop_difference_128': 0.007111850225348548, 'selected_region_median_orientation_valid_pixels_128': 1754.0, 'all_region_median_orientation_valid_pixels_128': 523.0, 'selected_region_median_coherence_128': 0.7174859642982483, 'all_region_median_coherence_128': 0.6760668754577637, 'expected_positive_example': False, 'noted_complex_example': True, 'interpretation_flag': 'review_needed'}
- 7028: {'confocal_image_id': '7028', 'filename': '7028.tif', 'total_patches': 961, 'candidate_patch_count': 598, 'candidate_patch_fraction': 0.6222684703433923, 'selected_region_median_oop_128': 0.825695890684641, 'selected_region_iqr_oop_128': 0.08936761438382235, 'all_region_median_oop_128': 0.8025646684852366, 'all_region_iqr_oop_128': 0.12591401092193388, 'selected_vs_all_oop_difference_128': 0.023131222199404444, 'selected_region_median_orientation_valid_pixels_128': 2902.0, 'all_region_median_orientation_valid_pixels_128': 2636.0, 'selected_region_median_coherence_128': 0.6960753798484802, 'all_region_median_coherence_128': 0.6842988729476929, 'expected_positive_example': False, 'noted_complex_example': False, 'interpretation_flag': 'broad_candidate_fraction_review_needed'}

## 5. Answer To Natalia

Yes, selective confident-region analysis appears feasible on the confocal images. Analysing candidate striated regions gives cleaner OOP/coherence summaries than analysing all signal, but this remains exploratory and needs visual/manual review before biological claims.

## 6. Calibration And Spacing

- Confocal pixel size: `unknown`
- Spacing in microns reported: False
- No spacing in microns is reported because confocal pixel calibration is unknown. Spacing may become feasible later only if pixel calibration and clear Z-discs are available.

## 7. Next Recommended Steps

- Ask Natalia for confocal pixel size or Leica metadata if available.
- Manually review moderate overlays for 5138, 6052-CLEAR_STRIPES, 3112, and 7028.
- Optionally create a small confocal annotation pack.
- Only after review, consider a confocal-specific validated configuration.
- Do not merge confocal thresholds into the widefield default configuration.

## 8. Allowed Claims

- The confocal transfer pilot processed all available confocal images.
- The existing widefield QC gate did not transfer unchanged to confocal images.
- The moderate selective mask is plausible for visual review.
- Selected regions show higher OOP/coherence than all regions in the current same-grid audit.
- Selective-region analysis appears feasible but remains exploratory.

## 9. Claims Not Allowed

- Confocal OOP is biologically validated.
- The moderate mask is a final Z-disc or striation segmentation.
- Spacing in microns is measured from the confocal images.
- Disease or healthy conclusions can be drawn from this pilot.
- Widefield conclusions are overturned by this confocal pilot.
