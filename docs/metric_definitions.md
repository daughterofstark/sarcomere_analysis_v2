# Metric Definitions And Assumptions

This document records scientific assumptions introduced by the pipeline.

## Calibration

- Pixel size is configured in `configs/default.yaml` as `0.1299 um/px`.
- Expected sarcomere spacing is configured in microns and converted to pixels at runtime.
- No module should hard-code pixel size or spacing bands.

## Implemented So Far

Configuration, calibration, TIFF IO, filename parsing, manifest generation, minimal preprocessing, tissue masking, patch grids, patch QC gates, structure tensor orientation/OOP, a conservative spacing scaffold, standardized outputs, and run provenance are implemented.

## Preprocessing Assumptions

- Percentile clipping is used for robust intensity normalization only. It is not interpreted as biological intensity quantification.
- Background subtraction removes low-frequency illumination and staining variation before texture analysis.
- Adaptive local contrast enhancement is excluded from the measurement preprocessing path because it can alter quantitative texture and periodicity.

## Masking And Patch QC Assumptions

- The tissue mask restricts later analysis to tissue-like signal. It is not a cell segmentation or cardiomyocyte segmentation.
- Patch QC is a gate for downstream measurements, not a biological endpoint.
- Invalid patches should return missing downstream metrics rather than forced orientation, periodicity, or spacing values.
- Thresholds are configurable and must be sensitivity-checked later before biological interpretation.
- Patch invalid reasons are semicolon-joined in a stable priority order: empty patch, low tissue fraction, low contrast, then low gradient energy.

## Structure Tensor Orientation And OOP

- Orientation is axial rather than directional, so angles separated by pi are equivalent.
- The orientational order parameter is computed as `|mean(exp(2i theta))|` using configurable energy/coherence-based weights.
- OOP measures local alignment of image texture. It does not measure sarcomere spacing.
- Patch-level OOP heterogeneity is a descriptive spatial heterogeneity metric, not a clinical conclusion.
- Patches that fail orientation QC return missing orientation metrics instead of forced values.

## Periodicity And Spacing Scaffold

- Spacing estimates are preliminary and require later validation against expert/FIJI measurements.
- Spacing is reported only for patches that pass QC, have a finite local orientation, and exceed the configured periodicity confidence threshold.
- Invalid or weak patches return missing spacing values rather than forced estimates.
- Directional autocorrelation is the primary scaffold direction for spacing estimation.
- The autocorrelation scaffold searches only the configured spacing band after converting microns to pixels. It now selects the strongest local maximum in that band rather than the largest boundary value. A boundary lag can be selected only if it is a local peak relative to its neighbor just outside the band, which helps reject monotonically decaying autocorrelation tails.
- Autocorrelation confidence remains peak value minus the configured band baseline percentile; the confidence threshold was not loosened by this peak-picking change.
- FFT-based spacing is included only as a scaffold/cross-check at this stage.

## Current Output Contract

- Per-patch metrics are QC and measurement rows. They are not independent biological samples.
- Per-image metrics are the current highest-level output today.
- Provenance records config, code, and input identity for reproducibility.
- Preview PNGs are QC diagnostics, not publication figures.

## Feature Tables

- Feature tables are assembled from existing metric outputs; they do not introduce new image-analysis measurements.
- OOP/orientation is the primary feature family in the current analysis layer.
- Spacing columns are preserved as exploratory low-yield descriptors with explicit status fields.
- Per-donor feature rows aggregate image-level features by `donor_id` and do not perform clinical or statistical inference.
