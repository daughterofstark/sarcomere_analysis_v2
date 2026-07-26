# Z-Disc / Striation Annotation Protocol

This scaffold prepares local crop images and editable mask PNGs for manual marking of visible Z-discs or striations. It is a manual annotation workflow only. It does not upload images, train a model, run ilastik, change the frozen production pipeline, replace spacing metrics, or compute validation statistics.

## Prepare The Set

From `<repo-root>`:

```bash
../sarcgraph-env/bin/python scripts/prepare_zdisc_annotation_set.py \
  --config configs/default.yaml \
  --n-crops 40 \
  --seed 123 \
  --overwrite
```

Outputs are written under:

```text
results/zdisc_annotation/
  images/
  masks/
  overlays/
  zdisc_annotation_index.csv
  zdisc_annotation_summary.json
  zdisc_annotation_summary.txt
```

The `images/` directory contains copied crop PNGs from the existing OOP annotation pack. The `masks/` directory contains blank mask PNGs with the same shape as each crop.

## Label Convention

Use integer labels only:

- `0` = background / unlabeled
- `1` = visible Z-disc / striation
- `2` = ignore / uncertain / autofluorescence / ambiguous

Empty masks are allowed. If no Z-discs or striations are visible, leave the mask all zero. Do not force labels where striations are indistinct.

## Editing Locally

Use any local image annotation tool that can edit label-mask PNGs, such as FIJI/ImageJ, napari, or ilastik.

In FIJI/ImageJ:

1. Open the crop PNG from `results/zdisc_annotation/images/`.
2. Open the matching blank mask from `results/zdisc_annotation/masks/`.
3. Draw only visible Z-discs/striations as label `1`.
4. Mark uncertain blobs, autofluorescence, or ambiguous regions as label `2`.
5. Save the edited mask PNG back to the same path in `results/zdisc_annotation/masks/`.

In napari or ilastik, load the crop image and corresponding mask as a labels layer. Keep labels restricted to `0`, `1`, and `2`, then save the edited mask PNG.

## Audit Masks

After editing masks, run:

```bash
../sarcgraph-env/bin/python scripts/audit_zdisc_annotations.py --config configs/default.yaml
```

To additionally write quick overlay previews:

```bash
../sarcgraph-env/bin/python scripts/audit_zdisc_annotations.py --config configs/default.yaml --write-overlays
```

The audit checks that each selected image has a mask, mask shapes match crop shapes, labels are restricted to `0`, `1`, and `2`, and counts empty masks versus masks containing visible Z-disc labels.

## Scope

These annotations can later support manual validation or possible ilastik-style classifier experiments. They do not replace the frozen OOP/spacing production pipeline, and they are not biological conclusions by themselves.
