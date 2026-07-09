# Spacing Failure Triage

## Purpose

This diagnostic module explains why the corrected spacing estimator rejects almost all patches. It does not change the estimator, thresholds, calibration, preprocessing, masks, QC logic, OOP logic, schemas, or biological interpretation.

Run:

```bash
../sarcgraph-env/bin/python scripts/triage_spacing_failures.py --config configs/default.yaml
```

Outputs:

- `results/diagnostics/spacing_failure_by_image.csv`
- `results/diagnostics/spacing_failure_summary.json`
- `results/diagnostics/spacing_failure_summary.txt`

## What It Reads

- `results/tables/per_patch_metrics.csv`
- `results/tables/per_image_metrics.csv`
- `results/diagnostics/spacing_diagnostic_summary.csv` if present

## What It Reports

- Total patches and image counts
- QC-valid spacing patches
- Patches that appear to reach the spacing estimator
- Final valid spacing patches
- Finite spacing values
- Inferred spacing rejection stages
- Top patch QC and spacing invalid reasons
- Failure stratification by tissue fraction, contrast, gradient energy, patch OOP, and spacing confidence when those columns exist
- By-image dominant rejection reason and QC metric summaries
- Whether candidate-level lag/peak diagnostics are available in the main patch table

## Safety Interpretation

The corrected spacing estimator is conservative. Fourteen valid spacing patches across the full batch is not enough for a reliable spacing endpoint. The next action should be evidence-based threshold and algorithm sensitivity analysis, not manual cherry-picking.

Sparse spacing does not automatically invalidate OOP/orientation outputs. Those are separate metrics and should be evaluated independently unless they share the same failing QC gate.

## Current Limitation

The main batch patch table does not contain candidate-level lag and peak diagnostics such as `selected_lag_px`, `autocorr_peak_value`, or `autocorr_baseline_value`. The spacing diagnostic module can produce some of these values, but deeper candidate-level diagnosis should be added as a diagnostic output later rather than by changing the scientific endpoint tables.
