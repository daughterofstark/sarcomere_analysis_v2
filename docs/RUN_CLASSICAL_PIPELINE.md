# Run Classical Pipeline

This is the reproducible runner for the current classical image-analysis pipeline. It orchestrates existing modules in the correct order and records what was generated.

It does not perform:

- validation statistics
- Bland-Altman analysis
- correlations
- clinical models
- publication figures
- benchmark comparisons
- ML
- cell segmentation
- spacing algorithm changes

## Default Table Run

The default command runs the core table pipeline:

```bash
../sarcgraph-env/bin/python scripts/run_classical_pipeline.py --config configs/default.yaml
```

Default order:

1. build manifest
2. run batch image metrics
3. audit batch outputs
4. assemble feature tables
5. enrich manifest
6. build analysis-ready tables
7. write pipeline run summary

Outputs:

- `results/pipeline_run_summary.json`
- `results/pipeline_run_summary.txt`

## Skip Existing Outputs

Use this when the full pipeline has already been run and you want to verify orchestration without recomputing the batch:

```bash
../sarcgraph-env/bin/python scripts/run_classical_pipeline.py \
  --config configs/default.yaml \
  --skip-existing
```

## Dry Run

Dry run prints the planned steps and output paths without running anything:

```bash
../sarcgraph-env/bin/python scripts/run_classical_pipeline.py \
  --config configs/default.yaml \
  --dry-run
```

## Optional QC Previews

Previews and the QC gallery are opt-in because they are slower and generate many PNGs:

```bash
../sarcgraph-env/bin/python scripts/run_classical_pipeline.py \
  --config configs/default.yaml \
  --with-previews
```

## Optional Spacing Diagnostics

Spacing diagnostics remain optional. Spacing is still `exploratory_low_yield` and should not be treated as a primary endpoint:

```bash
../sarcgraph-env/bin/python scripts/run_classical_pipeline.py \
  --config configs/default.yaml \
  --with-spacing-diagnostics
```

## Validation Template

Prepare the manual/FIJI validation template as part of orchestration:

```bash
../sarcgraph-env/bin/python scripts/run_classical_pipeline.py \
  --config configs/default.yaml \
  --with-validation-template
```

## Tests

Tests are not run automatically by the pipeline runner. Run them explicitly:

```bash
../sarcgraph-env/bin/python -m pytest
```

## Safety

Raw TIFFs are external inputs. The runner does not copy or modify raw TIFFs. It writes derived outputs under `results/` and, optionally, the validation template under `templates/`.
