# Expert Annotation Validation

This module imports Natalia's completed blinded expert annotation CSV, audits it, joins it to the internal blinding key, and compares manual striation visibility/organisation/confidence against automated patch OOP.

It does not change production algorithms, thresholds, feature tables, analysis tables, manual masks, or validation status outputs.

## Input File

Place Natalia's returned CSV at:

```text
results/expert_annotation_pack/expert_annotation_template_NG.csv
```

or pass another path with `--annotations`.

The internal blinding key must remain internal:

```text
results/expert_annotation_pack/internal_blinding_key.csv
```

## Command

```bash
../sarcgraph-env/bin/python scripts/validate_expert_annotations.py \
  --config configs/default.yaml \
  --annotations results/expert_annotation_pack/expert_annotation_template_NG.csv
```

Optional:

```bash
../sarcgraph-env/bin/python scripts/validate_expert_annotations.py \
  --config configs/default.yaml \
  --annotations results/expert_annotation_pack/expert_annotation_template_NG.csv \
  --min-n-correlation 10 \
  --min-confidence 3
```

## Robust Import

The importer normalizes common spreadsheet quirks:

- strips whitespace from column names
- lowercases column names
- removes trailing asterisks, e.g. `dominant_orientation_deg*`
- drops empty unnamed columns
- tolerates explanatory note columns

Invalid categorical values are converted to missing and reported in the audit. They are not fatal.

## Primary Validation Fields

Primary manual endpoints:

- `striations_visible`
- `organisation_score`
- `confidence_score`

The module reports:

- automated OOP medians by striation visibility
- automated OOP medians by organisation score
- Spearman association between organisation score and automated OOP when enough rows are available
- the same main association after filtering to `confidence_score >= 3`
- median organisation score by automated OOP bin

No clinical or disease comparisons are performed.

## Dominant Orientation

Natalia reported that she interpreted `dominant_orientation_deg` as the degree/direction of myofibril organisation rather than striation orientation for many rows.

Therefore:

- raw values are preserved as `expert_dominant_orientation_deg_raw`
- `expert_orientation_usable_primary` is `False` for all rows by default
- no primary orientation agreement is computed

This prevents a misleading orientation validation claim.

## Spacing

Manual sarcomere length was not completed in this annotation file. This is acceptable because spacing is already `exploratory_low_yield`.

The module reports `spacing_measurable` counts but does not validate spacing from this file.

## Outputs

```text
results/validation/expert_annotation_validation/
```

Files:

- `expert_annotations_normalized.csv`
- `expert_annotation_validation_matched.csv`
- `expert_annotation_validation_summary.json`
- `expert_annotation_validation_summary.txt`

The matched output includes internal IDs and automated OOP values and should be treated as internal analysis material, not expert-facing material.
