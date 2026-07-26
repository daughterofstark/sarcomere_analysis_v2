# Local Z-Disc Drawing UI

This tool lets you draw Z-disc/striation masks directly on the local crop images prepared under `results/zdisc_annotation/`. It is local-only: no images are uploaded, no network service is used, and no production OOP/spacing/features/analysis tables are changed.

## Start The UI

From `<repo-root>`:

```bash
../sarcgraph-env/bin/python scripts/draw_zdisc_annotations.py --config configs/default.yaml
```

Before opening the window, check that all images and masks are readable:

```bash
../sarcgraph-env/bin/python scripts/draw_zdisc_annotations.py --config configs/default.yaml --headless-check
```

To start from a specific crop:

```bash
../sarcgraph-env/bin/python scripts/draw_zdisc_annotations.py --config configs/default.yaml --start ANN_0001
```

## Label Convention

- `0` = background / unlabeled
- `1` = visible Z-disc / striation
- `2` = ignore / uncertain / autofluorescence / ambiguous

If an existing mask contains FIJI-style value `255`, the UI interprets it as label `1`. When the mask is saved after editing, labels are written back as `0`, `1`, or `2`.

## Controls

- Mouse wheel: zoom in/out at the cursor.
- `p`: toggle pan mode.
- Left-click drag: paint current label when pan mode is off; pan the view when pan mode is on.
- `r`: reset to full-image view.
- `1`: set current label to visible Z-disc/striation.
- `2`: set current label to ignore/uncertain.
- `0` or `e`: eraser/background.
- `[` / `]`: decrease/increase brush size.
- `n` or right arrow: save and go to next crop.
- `b` or left arrow: save and go to previous crop.
- `s`: save current mask.
- `c`: clear current mask.
- `o`: write overlay PNG for the current crop.
- `q`: save and quit.
- `h`: print controls.

Masks autosave after each completed mouse stroke.
Changing label, changing brush size, and saving do not reset the current zoom. Moving to next/previous crop reloads that crop at full view. Brush size is always measured in image pixels, so zooming does not change the actual mask radius.

## Annotation Rule

Draw only visible Z-discs/striations. Do not force faint ambiguous bands. Use label `2` for uncertain blobs, autofluorescence, or regions you want downstream review to ignore. Leave the crop blank if no Z-discs are visible.

## After Annotation

Run the audit and write overlays:

```bash
../sarcgraph-env/bin/python scripts/audit_zdisc_annotations.py --config configs/default.yaml --write-overlays
```

The audit checks that each mask exists, has the same shape as its crop image, and contains only labels `0`, `1`, and `2`.

## Scope

This UI is a manual annotation helper. It does not train ML, integrate ilastik, replace production spacing, compute validation statistics, or alter the frozen production pipeline.
