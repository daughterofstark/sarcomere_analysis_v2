# Spacing Sensitivity Report

## Purpose

The spacing sensitivity report simulates fixed acceptance variants from candidate-level spacing diagnostics. It is diagnostic-only. It does not change default config thresholds, production spacing acceptance logic, endpoint CSV schemas, calibration, preprocessing, masks, QC, OOP/orientation, or biological outputs.

Run:

```bash
../sarcgraph-env/bin/python scripts/report_spacing_sensitivity.py --config configs/default.yaml
```

Input:

- `results/diagnostics/spacing_candidates.csv`

If this file is missing, first run:

```bash
../sarcgraph-env/bin/python scripts/diagnose_spacing_candidates.py --config configs/default.yaml --all --compare-main-table
```

## Outputs

- `results/diagnostics/spacing_sensitivity_variants.csv`
- `results/diagnostics/spacing_sensitivity_summary.json`
- `results/diagnostics/spacing_sensitivity_summary.txt`

## Variant Dimensions

- Confidence threshold: default grid includes `0.10`, `0.12`, `0.14`, `0.15`, `0.18`, and `0.20`.
- Spacing band padding:
  - `current`
  - `min_minus_1`
  - `max_plus_1`
  - `both_plus_1`
- Peak rule:
  - `in_band_best_only`: accept best in-band candidate if confidence passes.
  - `current_selected_if_available`: use the current selected candidate columns if confidence passes.
  - `global_best_allowed`: diagnostic-only rule that allows global best peaks and is explicitly artefact-risk flagged when outside-band peaks dominate.

The diagnostic band padding is clipped to stay within 10 to 20 px unless the input candidates specify a narrower current band.

## Artefact Risk Heuristics

Variants are flagged as high risk if:

- `global_best_allowed` accepts many peaks outside the current expected band.
- Accepted lags concentrate heavily at one repeated lag.
- Accepted patches increase mainly by admitting candidates below the current confidence threshold.

Variants can also be labeled as:

- `conservative_low_yield`
- `plausible_for_review`
- `high_artefact_risk`
- `uninformative_low_yield`

These labels are triage labels, not biological conclusions and not final threshold recommendations.

## Interpretation

Evidence that would support threshold sensitivity:

- Many in-band peaks sit just below the current confidence threshold.
- Loosening confidence modestly increases image coverage without repeated-lag artefact flags.

Evidence that would support algorithm review:

- Many candidates have strong global peaks outside the expected band.
- Accepted candidates concentrate at suspicious repeated lags.
- Global-best variants greatly increase yield but are high artefact risk.

Evidence that spacing is not currently reliable:

- Most patches remain without in-band peaks.
- Accepted image coverage remains low across cautious variants.
- Confidence distributions remain low even under widened bands.

Evidence that OOP should remain primary for now:

- Spacing remains sparse or high-risk while OOP/orientation outputs remain interpretable and are not dependent on spacing peak acceptance.

No final threshold should be selected from this report alone.
