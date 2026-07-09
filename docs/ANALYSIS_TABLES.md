# Analysis Tables

Analysis tables join OOP-first feature tables with enriched image and donor metadata. They are analysis-ready inputs, not statistical results.

## Command

```bash
../sarcgraph-env/bin/python scripts/build_analysis_tables.py --config configs/default.yaml
```

Outputs:

- `results/tables/analysis_per_image.csv`
- `results/tables/analysis_per_donor.csv`
- `results/tables/analysis_table_summary.json`
- `results/tables/analysis_table_summary.txt`

## Table Levels

`analysis_per_image.csv` has one row per `image_id`. Image rows are nested within donor and should not be treated as independent patients.

`analysis_per_donor.csv` has one row per `donor_id`. Donor is the independent biological unit for later statistical planning.

## Sources

Image-level analysis rows join:

- `features_per_image.csv`
- `enriched_manifest.csv`

Donor-level analysis rows join:

- `features_per_donor.csv`
- `donor_metadata.csv`

Each output includes lightweight provenance columns:

- `feature_source`
- `metadata_source`

## Healthy Flag

The `is_healthy` flag comes from configured donor IDs. It is for exploratory grouping and validation planning only. In the current dataset, the healthy donor count is small (`n_healthy = 4`), so no healthy-vs-diseased inference should be made from this module.

## Spacing Status

Spacing remains exploratory low-yield. The analysis tables preserve spacing status/count/fraction columns so downstream code can track them, but spacing should not be treated as a primary endpoint.

## Interpretation Boundary

This module does not compute:

- p-values
- correlations
- group differences
- mixed models
- plots or figures
- clinical interpretation
- FIJI validation

It only prepares joined tables for later validated analysis.
