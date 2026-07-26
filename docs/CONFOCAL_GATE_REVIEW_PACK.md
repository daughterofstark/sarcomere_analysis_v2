# Confocal Gate Review Pack

This module exports a shareable visual review pack comparing the conservative `moderate_reference` confocal gate against the review-guided `moderate_relaxed_combined` gate.

It is review packaging only. It does not tune thresholds, refresh spacing, change algorithms, alter widefield outputs, or create publication figures.

## Command

```bash
../sarcgraph-env/bin/python scripts/export_confocal_gate_review_pack.py \
  --config configs/default.yaml \
  --write-zip
```

Default focus images:

- `5138`
- `6052-CLEAR_STRIPES`
- `3112`
- `7028`

## Outputs

Outputs are written under `results/confocal_gate_review_pack/`:

- `review_images/`
- `confocal_gate_review_summary.csv`
- `confocal_gate_review_notes_for_natalia.md`
- `confocal_gate_review_pack_summary.json`
- `confocal_gate_review_pack_summary.txt`
- `confocal_gate_review_pack_for_natalia.zip`, when requested

The zip contains only the review images, summary CSV, Natalia-facing notes, and summary TXT. It excludes raw images and large/internal source tables.

## What Natalia Should Check

- In the added-vs-moderate overlays, the extra highlighted layer marks regions newly admitted by the relaxed gate.
- Whether newly added relaxed-gate regions are mostly valid visible striations.
- Whether the missed middle region in `5138` is better captured.
- Whether shorter visible Z-disc structures in `3112` are better captured.
- Whether `7028` becomes too broad.

Spacing has not yet been recomputed for newly added relaxed-gate patches. The included spacing overlay is context from the prior moderate-gate audit only. If Natalia approves the relaxed gate, the next step is a refreshed calibrated spacing audit using `moderate_relaxed_combined`.
