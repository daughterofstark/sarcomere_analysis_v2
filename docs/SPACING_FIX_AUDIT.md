# Spacing Fix Audit

## Context

The original spacing scaffold used directional autocorrelation and selected the maximum autocorrelation value inside the configured sarcomere spacing band. On weak or nonperiodic patches, the autocorrelation curve often decayed monotonically, so the lower band edge could become the largest value in the allowed range without being a true periodic peak.

Observed pre-fix diagnostic pattern:

- Accepted spacing patches: 338
- Accepted lag distribution: 329 patches at 12 px / 1.5588 um, 1 patch at 17 px / 2.2083 um, 8 patches at 18 px / 2.3382 um
- Lower-bound accepted patches: 329/338, 97.34%

## Correction

The autocorrelation estimator now selects the strongest local maximum inside the expected lag band. A boundary lag is accepted only if it is a local peak relative to the adjacent lag outside the band. The confidence rule remains peak minus baseline, and the configured confidence threshold was not loosened.

No changes were made to:

- preprocessing
- tissue masks
- patch QC thresholds
- orientation/OOP logic
- calibration
- clinical/statistical code
- FIJI validation

## Verification

Tests:

- `../sarcgraph-env/bin/python -m pytest`
- Result: 91 passed

Synthetic recovery tests now cover known periods of 12, 14, 16, and 18 px across multiple orientations, with noise/blur/weak-contrast controls and no-striation negative controls.

## Refreshed Outputs

The full generated output set was refreshed after the fix.

Commands run:

```bash
../sarcgraph-env/bin/python scripts/run_batch_metrics.py --config configs/default.yaml --write-tables --write-provenance --continue-on-error
../sarcgraph-env/bin/python scripts/audit_batch_outputs.py --config configs/default.yaml
../sarcgraph-env/bin/python scripts/generate_qc_previews.py --config configs/default.yaml --overwrite --continue-on-error
../sarcgraph-env/bin/python scripts/build_qc_gallery.py --config configs/default.yaml --write-index --write-html
../sarcgraph-env/bin/python scripts/diagnose_spacing.py --config configs/default.yaml --write-summary
```

Full batch status:

- Images processed: 145
- OK: 145
- Errors: 0
- Per-image rows: 145
- Per-patch rows: 32,625
- Donors represented: 29

Corrected spacing results:

- Valid spacing patches: 14
- Images with no valid spacing: 132
- Spacing valid fraction: min 0.0, median 0.0, mean 0.000429, max 0.008889
- Accepted lag distribution: 12 px: 13 patches; 17 px: 1 patch
- Accepted spacing distribution: 1.5588 um: 13 patches; 2.2083 um: 1 patch

Top rejection stages after the fix:

- peak_picking: 20,680
- failed_patch_qc: 11,110
- confidence: 627
- missing_orientation: 194
- accepted: 14

Top invalid reasons after the fix:

- no_local_peak: 20,680
- low_contrast;failed_patch_qc: 4,777
- low_tissue_fraction;failed_patch_qc: 3,670
- low_tissue_fraction;low_contrast;failed_patch_qc: 1,525
- low_periodicity_confidence: 627

QC gallery:

- Images indexed: 145
- Images with missing previews: 0

## Output Paths

- `results/tables/per_image_metrics.csv`
- `results/tables/per_patch_metrics.csv`
- `results/tables/batch_run_summary.csv`
- `results/tables/batch_audit_summary.json`
- `results/tables/batch_audit_summary.txt`
- `results/tables/qc_preview_generation_summary.csv`
- `results/tables/qc_gallery_index.csv`
- `results/qc_gallery/index.html`
- `results/diagnostics/spacing_diagnostic_summary.csv`
- `results/diagnostics/spacing_diagnostic_by_image.csv`

## Interpretation Warning

Spacing remains very sparse and preliminary. The corrected estimator removed most boundary-artifact spacing calls, but usable spacing yield is now low. FIJI validation can now be implemented against the corrected scaffold, but validation should explicitly account for low spacing yield and the possibility that many archival myocardium patches do not contain a confidently measurable periodic signal under this method.
