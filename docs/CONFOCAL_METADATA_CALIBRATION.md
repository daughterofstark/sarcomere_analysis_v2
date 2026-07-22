# Confocal Metadata Calibration

This audit reads per-image pixel-size metadata from the confocal files referenced by `results/confocal_baseline/confocal_manifest.csv`.

It is intentionally separate from the frozen widefield pipeline. The widefield pixel size is never used as a fallback for confocal images.

## Command

```bash
../sarcgraph-env/bin/python scripts/audit_confocal_metadata.py \
  --config configs/default.yaml \
  --confocal-manifest results/confocal_baseline/confocal_manifest.csv \
  --write-manual-template
```

## Outputs

Outputs are written under `results/confocal_metadata/`:

- `confocal_metadata_calibration.csv`
- `confocal_metadata_summary.json`
- `confocal_metadata_summary.txt`
- `confocal_manual_pixel_size_template.csv`, when requested

## Metadata Sources

The parser checks, in order:

1. OME TIFF physical pixel sizes.
2. ImageJ TIFF unit and resolution tags, including ImageJ files with `unit=micron`.
3. TIFF resolution tags with physical units such as inch or centimeter.
4. PIL DPI metadata for non-TIFF images.

If pixel size is missing or unparseable, the image remains uncalibrated. The manual template can be filled from FIJI using `Image > Properties`.

## Spacing Policy

Confocal spacing in microns must use valid per-image confocal calibration. Do not assume all confocal images share one pixel size, and do not use the widefield pixel size as a fallback.

This module does not compute spacing, change thresholds, alter widefield outputs, or make biological claims.
