# Confocal Selective-Region Analysis

This module summarizes confocal patch features specifically inside candidate confident-striation regions selected by a sensitivity variant, defaulting to `moderate`.

It is exploratory. It does not change the widefield pipeline, default config, production algorithms, validation outputs, baseline confocal outputs, striation-mask outputs, or sensitivity outputs.

## Inputs

- `results/confocal_striation_mask/confocal_striation_mask_per_patch.csv`
- `results/confocal_baseline/confocal_baseline_per_patch.csv`
- `results/confocal_striation_sensitivity/confocal_striation_sensitivity_variants.csv`
- `results/confocal_striation_sensitivity/confocal_striation_sensitivity_per_image.csv`

The selected variant is read from the sensitivity variants table. No threshold search is recomputed.

Baseline patch-level OOP/orientation features are joined by `confocal_image_id` and `patch_id` when available. If coordinates are present, the analysis checks that the baseline patch grid and selective-region patch grid align. OOP is summarized only when the join is coordinate-consistent; mismatched-grid OOP is reported in the join audit but not used as same-region OOP.

## Run

```bash
../sarcgraph-env/bin/python scripts/run_confocal_selective_analysis.py \
  --config configs/default.yaml \
  --selected-variant moderate \
  --write-previews
```

Optional arguments:

```bash
--patch-table
--baseline-patch-table
--sensitivity-variants
--sensitivity-per-image
--output-dir
--min-candidate-patches
--write-previews
```

## Outputs

Outputs are written under:

```text
results/confocal_selective_analysis/
```

Files:

- `confocal_selective_per_patch.csv`
- `confocal_selective_per_image.csv`
- `confocal_selective_summary.json`
- `confocal_selective_summary.txt`
- optional previews under `results/confocal_selective_analysis/previews/`

## Metrics

For each image, the audit reports candidate patch counts/fractions and selected-region medians for:

- orientation coherence
- gradient energy
- intensity standard deviation
- OOP if a same-grid `patch_oop` column is available from the optional baseline patch join

The current confocal candidate-mask grid may differ from the baseline OOP grid. If the grids do not align, OOP summaries remain unavailable rather than forced.

## Interpretation

This answers whether selective-region summaries behave more sensibly than whole-image or all-patch summaries. It is not a validated confocal segmentation, not a spacing endpoint, and not a biological claim.

Confocal pixel size is still unknown, so spacing in microns is not computed.
