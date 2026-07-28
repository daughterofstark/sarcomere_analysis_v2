# Confocal Endpoint Review Pack

This module exports a compact visual QC pack for the larger confocal endpoint classification. It does not rerun analysis, change thresholds, adopt the relaxed gate, modify widefield outputs, or create publication figures.

## Command

```bash
../sarcgraph-env/bin/python scripts/export_confocal_endpoint_review_pack.py \
  --config configs/default.yaml \
  --write-zip
```

Optional arguments:

- `--endpoint-dir`
- `--pipeline-dir`
- `--audit-dir`
- `--output-dir`
- `--n-oop-only-examples`
- `--write-zip`

## Inputs

- `results/confocal_endpoint_report/confocal_endpoint_per_image.csv`
- `results/confocal_larger_audit/confocal_larger_image_triage.csv`
- `results/confocal_larger_pipeline/confocal_pipeline_per_image.csv`
- `results/confocal_larger_pipeline/confocal_pipeline_per_patch.csv`
- `results/confocal_larger_pipeline/previews/`
- `results/confocal_larger_audit/review_previews/` when available

## Outputs

Outputs are written under `results/confocal_endpoint_review_pack/`:

- `review_images/`
- `confocal_endpoint_review_index.csv`
- `confocal_endpoint_review_notes.md`
- `confocal_endpoint_review_pack_summary.json`
- `confocal_endpoint_review_pack_summary.txt`
- `confocal_endpoint_review_pack.zip` when `--write-zip` is used

The zip includes only review images, the review index, notes, and the text summary. It excludes raw images, full pipeline tables, internal large tables, and generated analysis directories.

## Review Groups

- `spacing_reportable`: all images where the endpoint report marks spacing as reportable.
- `low_candidate_review`: all images with `endpoint_class == low_candidate_review_needed`.
- `oop_only_examples`: deterministic representative examples from `oop_only_spacing_low_yield` and `spacing_eligible_low_confidence`.

The review is meant to answer whether endpoint classification is visually sensible:

- Do spacing-reportable images show genuine striation/spacing overlays?
- Do low-candidate images reflect missed signal, poor quality, or genuinely sparse striation?
- Do OOP-only examples clarify why spacing failed while OOP/coherence remained available?

No biological or clinical claims are made here.
