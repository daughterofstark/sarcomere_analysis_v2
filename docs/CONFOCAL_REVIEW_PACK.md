# Confocal Review Pack

This module exports a small shareable QC pack for Natalia to review confocal selected-region overlays and calibrated spacing previews.

It does not compute new metrics, tune thresholds, change algorithms, alter widefield outputs, or create publication figures.

## Command

```bash
../sarcgraph-env/bin/python scripts/export_confocal_review_pack.py \
  --config configs/default.yaml \
  --write-zip
```

Default images:

- `5138`
- `6052-CLEAR_STRIPES`
- `3112`
- `7028`

## Outputs

Outputs are written under `results/confocal_review_pack/`:

- `review_images/`
- `confocal_review_summary.csv`
- `confocal_review_notes_for_natalia.md`
- `confocal_review_pack_summary.json`
- `confocal_review_pack_summary.txt`
- `confocal_review_pack_for_natalia.zip`, when requested

The zip contains only the review images, summary CSV, Natalia-facing notes, and summary TXT. It excludes raw images and large internal source tables.

## Review Questions

Ask Natalia to check:

- Whether `5138` and `6052-CLEAR_STRIPES` overlays look biologically plausible.
- Whether `3112` should be treated as a negative or complex example.
- Whether `7028` is too broadly selected.
- Whether valid spacing patches appear to measure real adjacent Z-disc intervals.

The selected regions are candidate confident-striation regions, not final segmentation. Calibrated spacing remains exploratory until visually/manual reviewed.
