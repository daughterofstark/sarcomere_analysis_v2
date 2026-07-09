# Full-Image Z-Disc Annotation Features

This module extracts summary features from manually drawn full-image Z-disc/striation masks under `results/full_image_zdisc_annotation/`.

It only summarizes manual annotations. It does not validate automated metrics, compute correlations, perform clinical statistics, train ML, create publication figures, run an automatic detector, or change production OOP/spacing/features/analysis outputs.

## Inputs

- `results/full_image_zdisc_annotation/full_image_annotation_index.csv`
- `results/full_image_zdisc_annotation/images/`
- `results/full_image_zdisc_annotation/masks/`

Mask labels:

- `0` = background / unlabeled
- `1` = visible Z-disc / striation
- `2` = ignore / uncertain / autofluorescence / ambiguous

Label `2` is treated as ignore, not as a Z-disc.

## Command

```bash
../sarcgraph-env/bin/python scripts/extract_full_image_zdisc_annotation_features.py --config configs/default.yaml
```

Optional:

```bash
../sarcgraph-env/bin/python scripts/extract_full_image_zdisc_annotation_features.py \
  --config configs/default.yaml \
  --min-zdisc-pixels 10 \
  --min-components 1
```

## Outputs

- `results/full_image_zdisc_annotation/full_image_zdisc_annotation_features.csv`
- `results/full_image_zdisc_annotation/full_image_zdisc_annotation_feature_summary.json`
- `results/full_image_zdisc_annotation/full_image_zdisc_annotation_feature_summary.txt`

The per-image table includes label pixel counts, label fractions, annotation status, connected-component counts, median component size, and a rough manual-mask orientation proxy where estimable.

## Annotation Status

- `empty`: no label 1 and no label 2 pixels
- `zdisc_labeled`: label 1 pixels present, no label 2 pixels
- `ignore_only`: label 2 pixels present, no label 1 pixels
- `mixed`: both label 1 and label 2 pixels present

Empty masks are valid unclear or negative examples.

## Orientation Proxy

Manual mask orientation is estimated from label-1 pixels only using PCA over drawn pixel coordinates. The result is an axial angle in 0-180 degrees.

Orientation is returned as NaN when there are too few label-1 pixels, too few connected components according to the configured threshold, or degenerate mask geometry.

This is a rough mask-geometry proxy. It is not biological inference.

## Sparse Annotation Caution

Full-image masks may mark only visible representative Z-disc/striation regions. They should be treated as sparse manual annotations, not exhaustive segmentations of all Z-discs in the image.

The next step is pilot validation against automated image-level OOP/orientation. That validation is not performed here.
