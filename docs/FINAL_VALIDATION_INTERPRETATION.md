# Final Validation Interpretation

This report consolidates the validation evidence without changing algorithms, thresholds, feature tables, annotations, masks, or outputs.

## 1. Dataset And Pipeline Status

- Images: 145
- Donors: 29
- Patch feature rows: 32625
- Image feature rows: 145
- Donor feature rows: 29
- Full pipeline runs successfully: True
- Production pipeline frozen: True
- Test status from stored audit: 178 passed

## 2. Why Object-Level Z-Disc Detection Was Not The Primary Route

Object-level Z-disc detection and sarcomere spacing are not the right primary route for this widefield archival dataset.

- Z-discs are often faint, blurred, discontinuous, or locally ambiguous.
- Object-based Z-disc detection and spacing are fragile on these widefield images.
- The corrected conservative spacing estimator found too few confident patches.
- Spacing status: `exploratory_low_yield`
- Valid spacing patches: 14
- Primary spacing endpoint allowed: False
- No mean sarcomere length should be reported as a primary endpoint from this dataset.

## 3. What OOP/Orientation Validates

- Synthetic examples: 72
- Clean median angular error: 0.31775328371710465 deg
- Clean max angular error: 0.317754176975825 deg
- OOP monotonicity low > medium > high: True
- Recovered OOP by disorder: {'high': 0.844289756635255, 'low': 0.998193159593016, 'medium': 0.9086115051157587}
- Interpretation: This validates implementation behavior on controlled synthetic striated images, not biological endpoint validity in real tissue.

## 4. What Manual/Expert Validation Showed

- Manual Z-disc masks: Manual Z-disc masks did not confirm automated OOP as real-tissue organisation validation.
- Expert annotation rows: 75
- Expert matched rows: 75
- Visibility OOP medians: {'yes': 0.0865003952286183, 'unclear': 0.0701562998033931, 'no': 0.0582650828861118}
- Organisation vs OOP Spearman: {'computed': True, 'reason': 'computed', 'n': 51, 'rho': 0.01242829969906075, 'p_value': 0.9310222638650707, 'caution': 'Validation-supporting but still pilot/exploratory; no clinical inference.'}
- Confidence-filtered Spearman: {'computed': True, 'reason': 'computed', 'n': 41, 'rho': 0.0015733215847479757, 'p_value': 0.9922106789853568, 'caution': 'Validation-supporting but still pilot/exploratory; no clinical inference.'}
- Dominant orientation column excluded because reviewer reported ambiguity.
- Manual sarcomere length was not completed; spacing was not validated.
- Feature audit: No existing automated patch feature strongly tracked expert organisation score in this small single-reviewer audit.

## 5. Final Interpretation

- oop_orientation_implementation: `validated_on_synthetic_controlled_data`
- real_tissue_oop_as_expert_organisation_endpoint: `not_validated`
- striation_visibility: `weakly_reflected_by_oop`
- sarcomere_spacing: `not_validated_exploratory_low_yield`
- automated_current_pipeline: `useful_as_reproducible_image_texture_orientation_audit_not_yet_validated_biological_organisation_biomarker`

## 6. Recommended Next Directions

- If biological organisation quantification remains the goal, collect a higher-quality confocal subset.
- Repeat the same blinded annotation framework on confocal images.
- Consider larger expert/manual organisation scoring with clearer definitions.
- Consider supervised or semi-supervised models only after adequate labelled data exists.
- If the thesis/report deadline is near, frame the project as method development and negative validation on challenging archival widefield data.
- Emphasise reproducibility, auditability, and honest endpoint triage.
- Do not continue tuning OOP or spacing on the current data without new validation evidence.

## 7. Claims Allowed

- The pipeline processes the full dataset reproducibly.
- Spacing is low-yield in this dataset.
- Synthetic OOP validation passes on controlled striated images.
- Expert validation does not support OOP as a standalone organisation score.
- Widefield archival images are challenging for object-level sarcomere analysis.

## 8. Claims Not Allowed

- OOP is validated as expert-rated sarcomere organisation.
- Disease/healthy differences are biologically meaningful based on OOP.
- Sarcomere length can be robustly measured from this dataset.
- SarcGraph failed because of implementation error.
