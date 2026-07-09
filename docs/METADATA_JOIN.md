# Metadata Join

The metadata join module creates clean image-level and donor-level metadata tables for later validation or statistics. It does not join metadata into feature tables, run clinical models, perform hypothesis tests, or make biological claims.

## Command

```bash
../sarcgraph-env/bin/python scripts/enrich_manifest.py --config configs/default.yaml
```

Optional external donor metadata:

```bash
../sarcgraph-env/bin/python scripts/enrich_manifest.py \
  --config configs/default.yaml \
  --metadata path/to/donor_metadata.csv
```

Outputs:

- `results/tables/enriched_manifest.csv`
- `results/tables/donor_metadata.csv`
- `results/tables/metadata_join_summary.json`
- `results/tables/metadata_join_summary.txt`

## Identifier Rules

`donor_id` must always be treated as a string. Values such as `4.083` are identifiers, not numeric measurements. The join code reads and writes donor IDs as strings to avoid float coercion.

`image_id` is also preserved as a string, for example `2.007-1`.

## Healthy Flag

Known healthy donor IDs are configured in `configs/default.yaml`:

- `4.083`
- `5.003`
- `6.052`
- `7.028`

The resulting `is_healthy` flag is for exploratory grouping and later validation planning only. It is not a clinical conclusion.

## Nesting

The independent biological unit is `donor_id`.

Images are nested within donors. Patches are nested within images. Downstream analysis should not treat patches or images as independent patients.

## External Metadata

External metadata is joined on `donor_id`. Unmatched donors in the manifest and metadata-only donors are reported in `metadata_join_summary.json`.

By default, unmatched donors are allowed and reported. With `--strict`, unmatched donors cause the command to fail.

## Interpretation Boundary

This module prepares metadata for later work. Clinical/statistical interpretation happens later, after validation and an explicit analysis plan.
