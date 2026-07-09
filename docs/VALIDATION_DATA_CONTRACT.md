# Validation Data Contract

This module defines and audits the CSV format for future expert/manual FIJI measurements. It does not validate algorithm performance yet.

It does not compute:

- Bland-Altman summaries
- correlations
- regressions
- mixed models
- plots
- benchmark comparisons
- clinical statistics
- biological interpretation

## Template

Generate the template with:

```bash
../sarcgraph-env/bin/python scripts/prepare_validation_template.py --config configs/default.yaml
```

Template path:

```text
templates/manual_validation_template.csv
```

The template contains example rows marked with `EXAMPLE_...`. Replace these rows before real validation. Example rows are allowed in an audit only when `--allow-example-rows` is passed.

## Required Columns

- `measurement_id`
- `image_id`
- `donor_id`
- `measurement_type`
- `manual_value`
- `manual_unit`
- `expert_id`

## Optional Columns

- `region_id`
- `patch_id`
- `x_px`
- `y_px`
- `x0_px`
- `y0_px`
- `x1_px`
- `y1_px`
- `structure_label`
- `notes`
- `measurement_date`

## Measurement Types

Allowed `measurement_type` values:

- `oop_manual`
- `orientation_manual_deg`
- `sarcomere_length_manual_um`
- `zdisc_width_manual_um`
- `other`

Unknown measurement types fail by default. They can be allowed for intake testing with `--allow-unknown-types`, but this does not imply biological validity.

## Identifier Rules

`image_id` and `donor_id` are required for matching. `donor_id` is a string identifier and must never be treated as a floating-point number. Values such as `2.007` and `4.083` are IDs, not measurements.

Manual/FIJI measurements should be traceable to an image and, whenever possible, to a region or patch via `region_id`, `patch_id`, or pixel coordinates.

## Audit

Run an audit against the current analysis image table:

```bash
../sarcgraph-env/bin/python scripts/audit_validation_measurements.py \
  --config configs/default.yaml \
  --validation-csv path/to/manual_validation.csv
```

Template/example audit:

```bash
../sarcgraph-env/bin/python scripts/audit_validation_measurements.py \
  --config configs/default.yaml \
  --validation-csv templates/manual_validation_template.csv \
  --allow-example-rows
```

Outputs:

- `results/validation/manual_validation_audit_summary.json`
- `results/validation/manual_validation_audit_summary.txt`
- `results/validation/manual_validation_matched_rows.csv`
- `results/validation/manual_validation_unmatched_rows.csv`

## Interpretation Boundary

The audit checks schema and image/donor matching only. OOP/manual orientation validation is the current priority. Manual spacing values are supported by the schema, but spacing remains `exploratory_low_yield` unless later evidence supports using it.
