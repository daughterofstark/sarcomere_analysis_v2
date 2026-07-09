# Spacing Candidate Review Pack

This review pack is diagnostic-only. It exports representative patch panels so a human reviewer can inspect what the spacing estimator sees before any threshold or algorithm changes are considered.

It does not change:

- saved per-patch or per-image metrics
- spacing thresholds
- `valid_for_spacing_final`
- preprocessing, masking, QC, orientation, or OOP logic

## Command

```bash
../sarcgraph-env/bin/python scripts/export_spacing_review_pack.py \
  --config configs/default.yaml \
  --max-per-class 10 \
  --overwrite
```

Outputs are written to:

```text
results/diagnostics/spacing_candidate_review/
```

Key files:

- `review_index.csv`
- `review_summary.json`
- one PNG panel per selected patch

## Review Classes

- `accepted_current`: accepted by the current corrected spacing estimator.
- `no_local_peak`: no local autocorrelation peak was found in the expected spacing band.
- `low_periodicity_confidence`: an in-band peak exists, but confidence is below the current threshold.
- `global_out_of_band`: the strongest global peak is outside the expected spacing band and the in-band peak is absent or weaker.
- `borderline_in_band`: an in-band peak exists with confidence close to the current threshold.

If a class has zero available examples, the pack records count 0 and continues.

## How To Read A Panel

Each panel shows the preprocessed patch, tissue mask overlay, local orientation indicator, directional intensity profile, and autocorrelation curve. The expected spacing band is shaded. Selected, best in-band, and best global candidate peaks are marked when available.

These panels are QC evidence. They are not publication figures and do not validate sarcomere spacing.

## What Evidence Would Support Next Actions

- Keep spacing disabled or secondary if panels show weak, noisy, or non-periodic profiles across most classes.
- Improve the algorithm if panels show plausible periodic signal but the current peak rule misses it systematically.
- Consider threshold sensitivity only after panels show credible in-band peaks that fail narrowly and after external validation is planned.
- Prioritize OOP as the current primary endpoint if orientation maps remain interpretable while spacing evidence stays sparse or artefact-prone.

Spacing remains preliminary until manual/FIJI validation is implemented.
