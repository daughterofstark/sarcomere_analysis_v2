# Spacing Candidate Diagnostics

## Purpose

Candidate-level spacing diagnostics expose what the autocorrelation spacing estimator sees inside QC-passing, oriented patches. This is diagnostic-only. It does not change thresholds, final accept/reject logic, calibration, preprocessing, masks, OOP/orientation logic, endpoint schemas, or biological outputs.

Run one image:

```bash
../sarcgraph-env/bin/python scripts/diagnose_spacing_candidates.py --config configs/default.yaml --image-id 2.007-1 --compare-main-table
```

Run a bounded smoke batch:

```bash
../sarcgraph-env/bin/python scripts/diagnose_spacing_candidates.py --config configs/default.yaml --all --max-images 10 --compare-main-table
```

The full batch is intentionally not the default.

## Outputs

- `results/diagnostics/spacing_candidates.csv`
- `results/diagnostics/spacing_candidates_summary.json`
- `results/diagnostics/spacing_candidates_summary.txt`

## Definitions

- `expected_min_lag_px` and `expected_max_lag_px`: the configured sarcomere spacing band converted to pixels.
- `local peak`: an autocorrelation lag whose value is higher than the previous lag and at least as high as the next lag.
- `in-band peak`: a local peak whose lag lies inside the configured expected spacing band.
- `global peak`: the strongest local peak outside lag zero across the autocorrelation curve.
- `selected_lag_px`: the in-band local peak selected by the current estimator.
- `baseline_value`: the configured percentile baseline of the in-band autocorrelation values.
- `peak_prominence` / `peak_confidence`: selected peak value minus baseline, matching the current estimator confidence definition.
- `final_valid_for_spacing`: the current estimator's final accept/reject decision for the patch.

## Why This Comes Before Threshold Tuning

The corrected estimator rejects most patches because no local peak is found in the expected band. Candidate-level diagnostics distinguish several possibilities:

- There are no local peaks at all, suggesting weak or monotonic profiles.
- There are peaks, but they are outside the expected spacing band.
- There are in-band peaks, but confidence/prominence is below the current threshold.
- There are accepted in-band peaks, but yield remains too sparse for a robust spacing endpoint.

This evidence should be reviewed before changing thresholds. Lowering thresholds without knowing which case dominates risks restoring artefactual spacing calls.

## What Would Support Next Actions

- Threshold sensitivity: many patches have in-band peaks with confidence just below threshold.
- Algorithm improvement: many patches have plausible peaks outside the band, multiple ambiguous peaks, or profiles where peak selection is unstable.
- Spacing not currently reliable: most patches have no local peak, no in-band peak, or confidence distributions indistinguishable from weak/noisy profiles.
- OOP as primary endpoint for now: spacing remains sparse while orientation/OOP QC and maps remain interpretable, because OOP does not depend on the spacing peak-picking step.

No biological conclusion follows directly from this diagnostic output.
