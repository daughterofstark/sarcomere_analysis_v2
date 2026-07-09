# Z-Disc Annotation Features

This module extracts summary features from manually drawn Z-disc/striation masks under `results/zdisc_annotation/`. It is annotation summarization only. It does not validate automated metrics, compute correlations, perform clinical statistics, train ML, replace production spacing, or change the frozen analysis pipeline.

## Inputs

- `results/zdisc_annotation/zdisc_annotation_index.csv`
- `results/zdisc_annotation/images/`
- `results/zdisc_annotation/masks/`

Mask labels:

- `0` = background / unlabeled
- `1` = visible Z-disc / striation
- `2` = ignore / uncertain / autofluorescence / ambiguous

Label `2` is treated as ignore, not as a Z-disc.

## Command

```bash
../sarcgraph-env/bin/python scripts/extract_zdisc_annotation_features.py --config configs/default.yaml
```

Optional parameters:

```bash
../sarcgraph-env/bin/python scripts/extract_zdisc_annotation_features.py \
  --config configs/default.yaml \
  --min-zdisc-pixels 10 \
  --min-components 1
```

## Outputs

- `results/zdisc_annotation/zdisc_annotation_features.csv`
- `results/zdisc_annotation/zdisc_annotation_feature_summary.json`
- `results/zdisc_annotation/zdisc_annotation_feature_summary.txt`

The per-crop table includes pixel counts, label fractions, annotation status, connected-component counts, a simple component length proxy, and a rough manual-mask orientation proxy where estimable.

## Annotation Status

- `empty`: no label 1 and no label 2 pixels
- `zdisc_labeled`: label 1 pixels present, no label 2 pixels
- `ignore_only`: label 2 pixels present, no label 1 pixels
- `mixed`: both label 1 and label 2 pixels present

Empty masks are valid negative or unclear examples and should not be treated as failures.

## Orientation Proxy

Manual mask orientation is estimated from label-1 pixels only using PCA over drawn pixel coordinates. The result is an axial angle in 0-180 degrees, with 0 degrees horizontal and 90 degrees vertical.

The confidence value is the PCA anisotropy:

```text
(major_variance - minor_variance) / (major_variance + minor_variance)
```

Orientation is returned as NaN when there are too few label-1 pixels, too few connected components according to the configured threshold, or a degenerate mask geometry.

This orientation is a rough proxy from manual strokes, not a biological result by itself.

## Next Step

The next module can compare these manual-mask features to automated OOP/orientation outputs. That comparison is not performed here.
