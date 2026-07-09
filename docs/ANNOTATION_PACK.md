# OOP/Orientation Annotation Pack

This export creates a small, stratified set of patch crops for expert review in FIJI/ImageJ or another visual annotation workflow.

It is for manual validation planning, not ML training and not statistical validation.

## Purpose

OOP/orientation is the current primary endpoint family. The annotation pack is intended to support later validation of:

- dominant orientation
- visible organisation/striated texture
- manual organisation score

Spacing can be recorded optionally, but spacing remains `exploratory_low_yield` unless later validation supports using it.

## Export Command

```bash
../sarcgraph-env/bin/python scripts/export_annotation_pack.py \
  --config configs/default.yaml \
  --n-patches 80 \
  --seed 123 \
  --overwrite
```

Outputs:

- `results/annotation_pack/annotation_patch_index.csv`
- `results/annotation_pack/annotation_summary.json`
- `results/annotation_pack/annotation_template.csv`
- `results/annotation_pack/crops/*.png`

There is also a standalone template:

- `templates/oop_orientation_annotation_template.csv`

## Sampling

The pack is deterministic for a fixed seed. It samples only from existing feature tables and does not recompute metrics.

Sampling is stratified by automated patch OOP:

- low OOP
- medium OOP
- high OOP

Only patches with `valid_for_orientation == True` are used for the main OOP/orientation review. A small number of invalid or low-quality patches are included as negative controls if available.

The sampler spreads examples across donors and images so one donor or image does not dominate the pack.

## Annotation Rules

Annotate all exported patches. Do not manually select only visually pretty regions, because that would introduce selection bias.

Manual organisation score:

1. disorganised / no coherent striation orientation
2. weakly organised
3. moderately organised
4. strongly organised
5. highly organised

Suggested annotation fields:

- `manual_dominant_orientation_deg`
- `manual_organisation_score`
- `manual_organisation_label`
- `visible_striations_yes_no`
- `manual_sarcomere_length_um_optional`
- `confidence_score`
- `annotator_id`
- `notes`

## FIJI/ImageJ Workflow

Open the exported PNG crops in FIJI/ImageJ as ordinary image files. Use `annotation_patch_index.csv` to map each crop back to `image_id`, `donor_id`, `patch_id`, and patch coordinates.

Fill `annotation_template.csv` row-by-row. Keep the original `annotation_id`, `image_id`, `donor_id`, and `patch_id` unchanged so future validation code can match annotations back to the automated outputs.

## Boundary

This module exports review material only. It does not compute validation statistics, correlations, Bland-Altman summaries, figures, or biological conclusions.
