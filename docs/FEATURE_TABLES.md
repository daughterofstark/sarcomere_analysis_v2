# Feature Tables

Feature assembly converts existing batch outputs into stable downstream tables. It does not rerun image analysis, tune thresholds, validate spacing, perform clinical joins, or make biological/statistical inferences.

## Command

```bash
../sarcgraph-env/bin/python scripts/assemble_features.py --config configs/default.yaml
```

Outputs:

- `results/tables/features_per_patch.csv`
- `results/tables/features_per_image.csv`
- `results/tables/features_per_donor.csv`
- `results/tables/feature_assembly_summary.json`
- `results/tables/feature_assembly_summary.txt`

## Feature Policy

OOP/orientation is the primary feature family for the current pipeline:

- `image_oop`
- `image_oop_heterogeneity`
- `n_orientation_valid_patches`
- `orientation_valid_fraction`
- patch-level OOP summaries and dispersion metrics

Spacing is preserved only as exploratory low-yield metadata after the corrected spacing diagnostics:

- `n_spacing_valid_patches`
- `spacing_valid_fraction`
- `spacing_low_yield_flag`
- `spacing_endpoint_status`

Spacing should not gate OOP/orientation downstream. A low or missing spacing yield does not invalidate orientation/OOP measurements unless a separate QC review shows a shared upstream failure.

## Table Levels

`features_per_patch.csv` contains computational patch rows. These rows are useful for QC and spatial summaries, but they are not independent biological samples.

`features_per_image.csv` is the current highest-resolution analysis unit for downstream validation. It includes OOP endpoints, QC descriptors, and spacing status fields.

`features_per_donor.csv` aggregates image-level features by `donor_id`. This avoids treating patches or images as independent patients. It does not perform clinical analysis or hypothesis testing.

## Spacing Status

If an image has fewer than the configured minimum number of valid spacing patches, `spacing_endpoint_status` is:

```text
insufficient_patch_yield
```

The global summary records `spacing_global_status`. With the current corrected spacing scaffold, spacing is expected to remain exploratory and low-yield until external validation is performed.

## Interpretation Boundary

These feature tables are reproducibility and validation inputs. They are not clinical results, patient-level conclusions, or publication-ready statistical outputs.
