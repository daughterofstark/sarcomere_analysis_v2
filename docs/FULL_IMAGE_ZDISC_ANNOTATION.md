# Full-Image Z-Disc Annotation

This scaffold prepares selected full-image PNG working copies for local manual Z-disc/striation annotation. It is local-only and does not upload images, train ML, run validation statistics, change production OOP/spacing/features/analysis tables, or replace the frozen production pipeline.

## Prepare Full Images

From `<repo-root>`:

```bash
../sarcgraph-env/bin/python scripts/prepare_full_image_zdisc_annotation_set.py \
  --config configs/default.yaml \
  --n-images 12 \
  --seed 123 \
  --overwrite
```

Outputs:

```text
results/full_image_zdisc_annotation/
  images/
  masks/
  overlays/
  full_image_annotation_index.csv
  full_image_annotation_summary.json
  full_image_annotation_summary.txt
```

The `images/` files are preprocessed contrast-normalized PNG working copies for easier drawing. Raw TIFFs remain external local inputs and are not copied into this folder.

## Draw Masks

Check paths first:

```bash
../sarcgraph-env/bin/python scripts/draw_full_image_zdisc_annotations.py --config configs/default.yaml --headless-check
```

Start the drawing UI:

```bash
../sarcgraph-env/bin/python scripts/draw_full_image_zdisc_annotations.py --config configs/default.yaml
```

The UI uses Matplotlib. The built-in controls are:

- Mouse wheel: zoom in/out at the cursor
- `p`: toggle pan mode
- Left-click drag: paint current label when pan mode is off; pan the view when pan mode is on
- `r`: reset to full-image view
- `1`: visible Z-disc/striation
- `2`: ignore/uncertain/autofluorescence/ambiguous
- `0` or `e`: eraser/background
- `[` / `]`: decrease/increase brush size
- `n` or right arrow: save and next image
- `b` or left arrow: save and previous image
- `s`: save current mask
- `c`: clear current mask
- `o`: write overlay PNG
- `q`: save and quit
- `h`: print controls

Masks autosave after each completed mouse stroke.
Changing label, changing brush size, and saving preserve the current zoom. Moving next/back reloads that image at full view. Brush size is always in image pixels, so zooming changes only the view, not the mask radius.

## Label Convention

- `0` = background / unlabeled
- `1` = visible Z-disc / striation
- `2` = ignore / uncertain / autofluorescence / ambiguous

Draw only visible Z-discs/striations. Do not force faint ambiguous bands. Empty masks are valid if no visible Z-discs/striations are present.

## Audit

Run:

```bash
../sarcgraph-env/bin/python scripts/audit_full_image_zdisc_annotations.py --config configs/default.yaml --write-overlays
```

The audit checks that every selected full-image PNG has a mask, masks match image shape, labels are restricted to `0`, `1`, and `2`, and overlays are written when requested. Partially annotated or empty masks are allowed.

## Scope

This is a manual annotation scaffold only. It does not compute validation statistics, clinical statistics, ML training data products, publication figures, or production spacing replacements.
