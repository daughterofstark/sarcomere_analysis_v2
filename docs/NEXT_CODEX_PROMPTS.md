# Next Codex Prompts

## Prompt A: QC Preview Gallery/Index Only

Continue in `<repo-root>`.

Task: implement a QC preview gallery/index only. Create a lightweight HTML or CSV index that links existing per-image preview PNGs and key per-image QC metrics. Do not modify algorithms, thresholds, schemas, or generated scientific metrics.

Scope:
- One module/system only: visual QC indexing.
- Add tests for index generation on synthetic/minimal outputs.
- Include CLI command to generate the index.
- Stop after the gallery/index is generated and tested.

Do not implement FIJI validation, stats, benchmarking, ML, segmentation, publication figures, or biological conclusions.

## Prompt B: FIJI Validation Harness Design/Implementation

Continue in `<repo-root>`.

Task: design and implement a FIJI/manual validation harness for sarcomere spacing only. Use existing output tables and external annotation files. Do not tune the spacing scaffold.

Scope:
- One module/system only: validation harness.
- Define annotation schema.
- Add import/summarization scripts for manual/FIJI measurements.
- Add tests with tiny fake annotation files.
- Stop after validation inputs can be loaded, checked, and compared to existing computational outputs.

Do not implement clinical stats, donor-level aggregation, benchmarking, ML, segmentation, or publication figures.

## Prompt C: Per-Image To Per-Donor Feature Aggregation

Continue in `<repo-root>`.

Task: implement per-image to per-donor feature aggregation after validation scaffolding exists. Aggregate only already-computed image-level metrics and QC counts; do not create new biological interpretations.

Scope:
- One module/system only: aggregation.
- Add deterministic aggregation schemas.
- Add CLI and tests using synthetic per-image tables.
- Stop after per-donor tables are generated and schema-tested.

Do not implement stats until validation and aggregation are both present. Do not implement clinical inference, ML, segmentation, benchmarking, or publication figures.
