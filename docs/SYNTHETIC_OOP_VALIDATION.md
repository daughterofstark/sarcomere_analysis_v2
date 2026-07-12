# Synthetic OOP Validation

This module validates the frozen structure-tensor orientation/OOP implementation on controlled synthetic striated images with known orientation and controlled disorder.

It is an implementation sanity check only. It does not prove that real tissue OOP is biologically valid, and it does not replace expert real-image validation.

## Command

```bash
../sarcgraph-env/bin/python scripts/validate_synthetic_oop.py --config configs/default.yaml
```

Optional:

```bash
../sarcgraph-env/bin/python scripts/validate_synthetic_oop.py \
  --config configs/default.yaml \
  --seed 123 \
  --n-replicates 1 \
  --size 256 \
  --write-example-images
```

## Outputs

- `results/validation/synthetic_oop_validation_results.csv`
- `results/validation/synthetic_oop_validation_summary.json`
- `results/validation/synthetic_oop_validation_summary.txt`
- optional example PNGs in `results/validation/synthetic_oop_examples/`

## Synthetic Cases

The generator creates sinusoidal striated patches with:

- known axial orientation
- known period in pixels
- controlled disorder level: `low`, `medium`, `high`
- optional Gaussian noise
- optional blur
- optional background gradient

The default grid is intentionally modest: six orientations, three disorder levels, three noise levels, and two blur levels.

## Orientation Convention

`true_orientation_deg` is defined in the same axial convention used by the frozen structure-tensor implementation. It corresponds to the sinusoidal intensity-gradient orientation estimated by the current code path. Angles are axial: 0 and 180 degrees are equivalent.

## Metrics

The validation reports:

- clean low-disorder orientation recovery using axial angular error
- recovered OOP medians by disorder level
- whether `low > medium > high` recovered OOP monotonicity holds
- degradation summaries by noise and blur level

No p-values, clinical comparisons, or biological claims are made.

## Interpretation

Passing synthetic validation supports the claim that the implementation can recover orientation/OOP on controlled images matching its assumptions.

It does not establish that OOP is biologically valid in archival myocardium. The manual Z-disc mask validation remains pilot-only and did not validate automated OOP on the current manually labeled masks. Real expert validation is still needed before publication claims.

Spacing remains `exploratory_low_yield`.
