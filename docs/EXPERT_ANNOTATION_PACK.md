# Expert Annotation Pack

This module exports a blinded patch annotation package for external review of alpha-actinin striation visibility, organisation, dominant orientation, and confidence.

It exists because synthetic OOP validation showed controlled implementation recovery, but the current manual Z-disc mask pilots were not confirmatory for real-tissue OOP. The next validation evidence should be blinded manual organisation/orientation scoring, not more sparse Z-disc masks.

## Command

```bash
../sarcgraph-env/bin/python scripts/export_expert_annotation_pack.py --config configs/default.yaml --write-zip
```

Optional:

```bash
../sarcgraph-env/bin/python scripts/export_expert_annotation_pack.py \
  --config configs/default.yaml \
  --n-total 75 \
  --n-per-bin 25 \
  --seed 123 \
  --max-per-donor 4 \
  --max-per-image 3 \
  --write-zip
```

## Generated Files

Output directory:

```text
results/expert_annotation_pack/
```

Expert-facing files:

- `patches/`
- `expert_annotation_template.csv`
- `annotation_instructions.md`
- `expert_annotation_pack_summary.txt`
- optional `expert_annotation_pack_for_natalia.zip`

Internal-only file:

- `internal_blinding_key.csv`

Do not send `internal_blinding_key.csv` to Natalia or any blinded reviewer. It contains image IDs, donor IDs, patch IDs, automated OOP bins, automated OOP values, and source paths.

## Scoring Fields

The expert template contains:

- `annotation_id`
- `patch_filename`
- `striations_visible`
- `organisation_score`
- `dominant_orientation_deg`
- `confidence_score`
- `spacing_measurable`
- `manual_sarcomere_length_um_optional`
- `notes`

Only `annotation_id` and `patch_filename` are pre-filled.

Allowed values:

- `striations_visible`: `yes`, `unclear`, `no`
- `organisation_score`: 1 to 5
- `dominant_orientation_deg`: axial 0-180 degrees, blank if unclear
- `confidence_score`: 1 to 5
- `spacing_measurable`: `yes`, `unclear`, `no`

Manual sarcomere length is optional and should be left blank unless at least 3 adjacent Z-disc intervals are clearly visible.

## Selection

The default pack selects 75 patches:

- 25 low automated OOP
- 25 medium automated OOP
- 25 high automated OOP

The bins are quantile-based from automated patch OOP. The exporter includes only patches that passed orientation QC and limits dominance by donor/image by default:

- max 4 patches per donor
- max 3 patches per image

If exact balance is impossible under these constraints, the exporter records the shortfall in the summary.

## Later Use

The resulting blinded annotations can later be joined back through the internal key to evaluate whether automated patch OOP and orientation agree with manual organisation/orientation review.

This module does not compute validation statistics, p-values, clinical comparisons, plots, ML training data, or publication figures.
