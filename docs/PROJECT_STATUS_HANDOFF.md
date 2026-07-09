# Project Status Handoff

Generated: `2026-07-09T14:40:53.353379+00:00`

Project path: `/Users/medhasharma/sarcomere_tools/sarcomere-analysis`

Test status recorded: `178 passed`

## Current Scientific Decisions

- Primary feature family: `OOP/orientation`
- Spacing status: `exploratory_low_yield`
- Spacing policy: Spacing is preserved but should not be used as a primary endpoint unless future validation changes this.
- Biological claims made: `False`
- Independent biological unit: `donor_id`
- Healthy-vs-diseased status: exploratory_grouping_only; healthy_donor_count=4
- Real FIJI/manual validation data ingested: `False`

## Row Counts

```json
{
  "manifest": 145,
  "per_patch_metrics": 32625,
  "per_image_metrics": 145,
  "features_per_patch": 32625,
  "features_per_image": 145,
  "features_per_donor": 29,
  "enriched_manifest": 145,
  "donor_metadata": 29,
  "analysis_per_image": 145,
  "analysis_per_donor": 29,
  "pipeline_run_summary": null
}
```

## Safety Checks

```json
{
  "missing_required_core_outputs": [],
  "analysis_per_image_matches_manifest_rows": true,
  "analysis_per_donor_matches_donor_metadata_rows": true,
  "donor_id_string_preserved": true,
  "spacing_status": "exploratory_low_yield",
  "spacing_status_present": true,
  "passed": true
}
```

## Completed Modules

- scaffold/config/calibration
- IO/manifest
- preprocessing
- tissue masking/QC/patch grid
- orientation/OOP
- spacing scaffold + diagnostics
- batch metrics
- feature assembly
- metadata/enriched manifest
- analysis-ready tables
- validation intake scaffold
- pipeline orchestration

## Not Yet Implemented

- real FIJI validation statistics
- Bland-Altman/correlation
- benchmark tools
- synthetic degradation
- clinical/mixed-model stats
- publication figures
- JOSS packaging
- cell segmentation/ML

## Reproducible Commands

- `full_tests`: `../sarcgraph-env/bin/python -m pytest`
- `dry_run_pipeline`: `../sarcgraph-env/bin/python scripts/run_classical_pipeline.py --config configs/default.yaml --dry-run`
- `skip_existing_pipeline`: `../sarcgraph-env/bin/python scripts/run_classical_pipeline.py --config configs/default.yaml --skip-existing`
- `full_table_pipeline`: `../sarcgraph-env/bin/python scripts/run_classical_pipeline.py --config configs/default.yaml`
- `optional_previews`: `../sarcgraph-env/bin/python scripts/run_classical_pipeline.py --config configs/default.yaml --with-previews`
- `optional_validation_template`: `../sarcgraph-env/bin/python scripts/run_classical_pipeline.py --config configs/default.yaml --with-validation-template`
- `feature_assembly`: `../sarcgraph-env/bin/python scripts/assemble_features.py --config configs/default.yaml`
- `analysis_table_build`: `../sarcgraph-env/bin/python scripts/build_analysis_tables.py --config configs/default.yaml`

## Boundary

This handoff records project state only. It does not add algorithms, statistics, validation analysis, figures, benchmark tools, clinical models, ML, segmentation, or spacing changes.
