# Confocal Cohort Audit

This audit summarizes an already-generated consolidated confocal pipeline run. It does not rerun image analysis, tune gates, adopt the relaxed gate, or change widefield outputs.

## Command

```bash
../sarcgraph-env/bin/python scripts/audit_confocal_cohort.py \
  --config configs/default.yaml \
  --pipeline-dir results/confocal_larger_pipeline \
  --pilot-dir results/confocal_pipeline \
  --collect-previews
```

## Inputs

- `results/confocal_larger_pipeline/confocal_pipeline_manifest.csv`
- `results/confocal_larger_pipeline/confocal_pipeline_per_image.csv`
- `results/confocal_larger_pipeline/confocal_pipeline_per_patch.csv`
- `results/confocal_larger_pipeline/confocal_pipeline_summary.json`
- Optional 11-image pilot summaries from `results/confocal_pipeline/`

## Outputs

Outputs are written under `results/confocal_larger_audit/`:

- `confocal_larger_image_triage.csv`
- `confocal_larger_cohort_summary.csv`
- `confocal_larger_spacing_distribution.csv`
- `confocal_larger_audit_summary.json`
- `confocal_larger_audit_summary.txt`
- `review_previews/` when `--collect-previews` is used

## Triage Classes

- `spacing_robust`: selected spacing fraction is at least 0.25 and at least 25 selected patches have valid spacing.
- `spacing_moderate`: selected spacing fraction is at least 0.10 and at least 10 selected patches have valid spacing.
- `oop_only_low_spacing`: selected candidate regions exist, but spacing yield is below 0.10.
- `low_candidate_fraction_review`: fewer than 5% of patches pass the selected-region gate.
- `broad_candidate_fraction_review`: more than 70% of patches pass the selected-region gate.
- `failed_or_error`: the pipeline reported an image processing error.

## Interpretation

The 42-image run is an operational cohort audit. It is intended to show whether the confocal selected-region workflow scales and which images are spacing-eligible versus OOP-only or review-needed.

The current larger cohort has much lower selected-region spacing yield than the 11-image pilot. That means confocal spacing remains exploratory and image-dependent, even though selected-region OOP remains stable/promising. The primary gate remains `moderate`; `moderate_relaxed_combined` remains sensitivity/review only.

No biological or clinical claims are made from this audit.
