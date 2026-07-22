# Confocal Same-Grid OOP/Orientation

This module computes orientation and OOP directly on the same patch grid used by the confocal confident-striation candidate mask. It was added because the earlier confocal selective analysis could not reuse baseline patch OOP: the baseline confocal grid used larger 256 px patches, while the moderate candidate mask uses a 128 px grid.

## Purpose

- Measure OOP/orientation on the exact regions selected by the moderate confocal candidate mask.
- Preserve the existing frozen orientation/OOP implementation.
- Avoid joining incompatible patch grids.
- Keep confocal analysis separate from the frozen widefield outputs.

This is exploratory and not manually validated on confocal annotations yet.

## Command

```bash
../sarcgraph-env/bin/python scripts/run_confocal_same_grid_oop.py \
  --config configs/default.yaml \
  --write-previews
```

## Inputs

- `results/confocal_striation_mask/confocal_striation_mask_per_patch.csv`
- `results/confocal_baseline/confocal_manifest.csv`
- confocal source images referenced in the manifest
- the moderate candidate assignments from `results/confocal_selective_analysis/confocal_selective_per_patch.csv`, when present

## Outputs

All outputs are written under `results/confocal_same_grid_oop/`:

- `confocal_same_grid_oop_per_patch.csv`
- `confocal_same_grid_oop_per_image.csv`
- `confocal_same_grid_oop_summary.json`
- `confocal_same_grid_oop_summary.txt`
- optional previews under `previews/`

The per-patch table carries the 128 px patch coordinates, the moderate candidate flag, expected-positive/complex flags, and OOP/orientation values computed directly on that patch.

## Interpretation

This answers a narrow technical question: among the same candidate patches selected by the confocal striation-region mask, what are the frozen orientation/OOP measurements? It does not introduce a new segmentation method, does not compute spacing in microns, and does not make biological claims.

Confocal pixel size is still unknown, so sarcomere spacing in microns is not reported.
