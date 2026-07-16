# Expert-Visible Crop Feature Audit

This module recomputes automated image descriptors directly on the blinded PNG crops that Natalia reviewed in the expert annotation pack.

The purpose is to test a region-alignment question: the previous audit compared Natalia's scores for larger expert-visible crops against automated features from the smaller internal production patch. If that mismatch matters, descriptors computed on the exact reviewed PNGs may relate more strongly to manual visibility or organisation scores.

This is exploratory only. It does not change production OOP, spacing, feature tables, analysis tables, thresholds, annotations, or masks.

## Inputs

- `results/expert_annotation_pack/patches/`
- `results/expert_annotation_pack/internal_blinding_key.csv`
- `results/validation/expert_annotation_validation/expert_annotation_validation_matched.csv`
- `configs/default.yaml`

## Descriptors

For each `EXPERT_XXXX.png`, the audit computes:

- crop OOP
- crop mean axial orientation
- crop coherence summaries
- orientation valid pixel count and fraction
- gradient energy
- intensity mean and standard deviation
- percentile contrast
- entropy
- Laplacian variance as a sharpness/blur proxy

The OOP/orientation calculation reuses the existing frozen structure-tensor implementation. The PNG crop is treated as the reviewed display region, not as a replacement for production measurements.

## Analyses

The audit reports:

- median crop features by `striations_visible` group
- yes-minus-no separation for visibility
- Spearman correlations between crop features and `organisation_score`
- confidence-filtered Spearman correlations for `confidence_score >= 3`
- median crop features by collapsed organisation groups:
  - low = scores 1-2
  - medium = score 3
  - high = scores 4-5
- comparison of previous production-patch OOP correlation versus crop-level OOP correlation

Dominant orientation remains excluded as a primary validation endpoint because the reviewer reported ambiguity in how that field was interpreted. Spacing remains not validated.

## Run

```bash
../sarcgraph-env/bin/python scripts/audit_expert_crop_features.py --config configs/default.yaml
```

Outputs are written to:

```text
results/validation/expert_crop_feature_audit/
```

## Interpretation

This audit can show whether the weak expert-validation result was partly caused by scoring one visible region while measuring another. It is not feature selection, threshold tuning, clinical analysis, or validation of a new production endpoint.
