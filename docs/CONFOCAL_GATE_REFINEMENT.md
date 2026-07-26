# Confocal Gate Refinement Audit

This module runs a small review-guided sensitivity around the current moderate confocal confident-striation gate.

It is not final tuning, not a new segmentation algorithm, and not a biological claim. The saved moderate outputs remain the conservative baseline and are not overwritten.

## Command

```bash
../sarcgraph-env/bin/python scripts/run_confocal_gate_refinement.py \
  --config configs/default.yaml \
  --write-previews
```

## Variants

- `moderate_reference`
- `moderate_relaxed_coherence`
- `moderate_relaxed_gradient`
- `moderate_relaxed_contrast`
- `moderate_relaxed_combined`

Relaxed variants move selected thresholds partway from the current moderate gate toward the previously audited lenient gate. They use existing patch-level features only.

## Review Focus

The default focus images are:

- `5138`
- `6052-CLEAR_STRIPES`
- `3112`
- `7028`

The preview overlays show selected patches, newly added patches relative to moderate, and calibrated valid spacing patches where available.

## Outputs

Outputs are written under `results/confocal_gate_refinement/`:

- `confocal_gate_refinement_variants.csv`
- `confocal_gate_refinement_per_image.csv`
- `confocal_gate_refinement_summary.json`
- `confocal_gate_refinement_summary.txt`
- optional previews under `previews/`

Spacing summaries use the calibrated spacing values already available from the prior spacing audit. Newly added relaxed-gate patches should be visually reviewed before any refreshed spacing audit is considered.
