# Manual Spacing Assist

This helper assists a human annotator in measuring optional sarcomere spacing from an exported annotation crop. It is not a production spacing estimator and does not replace the frozen pipeline outputs.

## Purpose

Use this tool only when sarcomere bands are visibly and confidently identifiable. It samples a user-specified line, shows the intensity profile, marks candidate peaks/troughs, and writes a human-assisted measurement row.

Spacing remains `exploratory_low_yield` in the main project. OOP/orientation remains the primary endpoint.

## Command

```bash
../sarcgraph-env/bin/python scripts/assist_manual_spacing.py \
  --config configs/default.yaml \
  --crop results/annotation_pack/crops/ANN_0001__4.068-5__4.068-5_p00186.png \
  --x0 10 --y0 40 --x1 60 --y1 20 \
  --intervals 3 \
  --confidence-score 2 \
  --accepted-by-user yes \
  --notes "faint bands, low confidence" \
  --write-panel \
  --append
```

Outputs:

- `results/annotation_pack/manual_spacing_assist_results.csv`
- optional diagnostic panels under `results/annotation_pack/manual_spacing_panels/`

## How Spacing Is Estimated

If `--intervals` is supplied, spacing is:

```text
line_length_um / interval_count
```

If `--intervals` is omitted, the helper lightly smooths the line profile and uses candidate peak/trough intervals as a suggestion. The human annotator remains responsible for accepting or rejecting the value.

## Rules For Use

- Do not force faint Z-discs or ambiguous bands.
- Use only when 3 or more intervals are confidently identifiable.
- Record low confidence honestly with `--confidence-score` and `--notes`.
- Use `--accepted-by-user no` or `unsure` when the suggested profile is not credible.
- Manual spacing is optional in the annotation pack.
- Accepted rows are human-assisted manual values, not automated production spacing.

## Boundary

This tool does not change:

- production spacing algorithms
- thresholds
- feature tables
- analysis tables
- OOP/orientation outputs
- clinical/statistical outputs

It also does not compute validation statistics, correlations, Bland-Altman summaries, publication figures, or biological conclusions.
