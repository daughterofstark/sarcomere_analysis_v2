from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import tifffile

from sarcomere_analysis.spacing import PATCH_SPACING_COLUMNS, compute_spacing_analysis, spacing_band_px
from sarcomere_analysis.spacing.autocorrelation import estimate_spacing_autocorrelation
from sarcomere_analysis.spacing.base import summarize_image_spacing
from test_step_5_orientation import step5_config, stripe_image


def step6_config(tmp_path: Path | None = None) -> dict:
    config = step5_config(tmp_path)
    config["calibration"]["pixel_size_um"] = 0.1
    config["calibration"]["expected_sarcomere_spacing_um"] = {"min": 1.0, "max": 1.6}
    config["spacing"] = {
        "method": "autocorrelation",
        "fallback_to_fft": False,
        "min_spacing_um": 1.0,
        "max_spacing_um": 1.6,
        "min_periodicity_confidence": 0.10,
        "autocorrelation": {
            "profile_bin_px": 1.0,
            "min_profile_length_px": 32,
            "peak_baseline_percentile": 50.0,
        },
        "fft": {
            "min_profile_length_px": 32,
            "peak_prominence_ratio": 4.0,
        },
    }
    return config


def single_patch_metrics(valid_for_spacing: bool = True, orientation: float = np.pi / 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": "synthetic",
                "patch_id": "synthetic_p00000",
                "y0": 0,
                "x0": 0,
                "y1": 64,
                "x1": 64,
                "center_y": 32,
                "center_x": 32,
                "tissue_fraction": 1.0 if valid_for_spacing else 0.0,
                "intensity_mean": 0.5,
                "intensity_std": 0.2,
                "rms_contrast": 0.2,
                "gradient_energy": 0.1,
                "near_zero_fraction": 0.0,
                "saturated_fraction": 0.0,
                "valid_for_orientation": valid_for_spacing,
                "valid_for_periodicity": valid_for_spacing,
                "valid_for_spacing": valid_for_spacing,
                "invalid_reason": "ok" if valid_for_spacing else "low_tissue_fraction",
                "patch_oop": 1.0 if np.isfinite(orientation) else np.nan,
                "patch_mean_orientation_rad": orientation,
                "patch_mean_orientation_deg": float(np.degrees(orientation)) if np.isfinite(orientation) else np.nan,
                "patch_orientation_weight_sum": 1.0 if np.isfinite(orientation) else 0.0,
                "patch_orientation_valid_pixels": 4096 if np.isfinite(orientation) else 0,
            }
        ]
    )


def oriented_stripe_patch(
    period_px: float,
    orientation_rad: float,
    shape: tuple[int, int] = (128, 128),
    noise_sigma: float = 0.0,
    contrast: float = 0.35,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    coord = xx * np.cos(orientation_rad) + yy * np.sin(orientation_rad)
    image = 0.5 + contrast * np.sin(2.0 * np.pi * coord / period_px)
    if blur_sigma > 0:
        from scipy import ndimage as ndi

        image = ndi.gaussian_filter(image, sigma=blur_sigma)
    if noise_sigma > 0:
        rng = np.random.default_rng(123)
        image = image + rng.normal(0.0, noise_sigma, size=shape)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def test_spacing_package_imports_cleanly() -> None:
    import sarcomere_analysis.spacing as spacing

    assert "patch_spacing_um" in spacing.PATCH_SPACING_COLUMNS


def test_spacing_band_px_is_derived_from_config_calibration() -> None:
    config = step6_config()
    assert spacing_band_px(config) == (10.0, 16.0)


def test_synthetic_sinusoidal_patch_returns_known_spacing() -> None:
    config = step6_config()
    image = stripe_image((64, 64), period=12.0)
    result = estimate_spacing_autocorrelation(image, np.pi / 2, config)
    assert result.valid_for_spacing_final
    assert abs(result.patch_spacing_px - 12.0) <= 1.0
    assert abs(result.patch_spacing_um - 1.2) <= 0.11


def test_synthetic_periods_recover_without_lower_bound_collapse() -> None:
    config = step6_config()
    config["spacing"]["min_spacing_um"] = 1.0
    config["spacing"]["max_spacing_um"] = 2.0
    config["spacing"]["min_periodicity_confidence"] = 0.05
    for period in [12.0, 14.0, 16.0, 18.0]:
        image = oriented_stripe_patch(period, np.pi / 2, noise_sigma=0.02)
        result = estimate_spacing_autocorrelation(image, np.pi / 2, config)
        assert result.valid_for_spacing_final
        assert abs(result.patch_spacing_px - period) <= 1.0


def test_synthetic_oriented_periods_recover_across_angles() -> None:
    config = step6_config()
    config["spacing"]["min_spacing_um"] = 1.0
    config["spacing"]["max_spacing_um"] = 2.0
    config["spacing"]["min_periodicity_confidence"] = 0.05
    for period in [12.0, 14.0, 16.0, 18.0]:
        for angle_deg in [0.0, 30.0, 60.0, 90.0]:
            orientation = np.deg2rad(angle_deg)
            image = oriented_stripe_patch(period, orientation, noise_sigma=0.02)
            result = estimate_spacing_autocorrelation(image, orientation, config)
            assert result.valid_for_spacing_final
            assert abs(result.patch_spacing_px - period) <= 1.0


def test_blurred_and_weak_contrast_striations_recover_when_clear() -> None:
    config = step6_config()
    config["spacing"]["min_spacing_um"] = 1.0
    config["spacing"]["max_spacing_um"] = 2.0
    config["spacing"]["min_periodicity_confidence"] = 0.05
    image = oriented_stripe_patch(16.0, np.deg2rad(30.0), noise_sigma=0.015, contrast=0.18, blur_sigma=1.0)
    result = estimate_spacing_autocorrelation(image, np.deg2rad(30.0), config)
    assert result.valid_for_spacing_final
    assert abs(result.patch_spacing_px - 16.0) <= 1.0


def test_no_striation_negative_control_returns_invalid() -> None:
    config = step6_config()
    config["spacing"]["min_spacing_um"] = 1.0
    config["spacing"]["max_spacing_um"] = 2.0
    ramp = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    image = np.repeat(ramp[None, :], 128, axis=0)
    result = estimate_spacing_autocorrelation(image, 0.0, config)
    assert not result.valid_for_spacing_final
    assert result.spacing_invalid_reason in {"no_local_peak", "low_periodicity_confidence"}


def test_period_16_and_18_do_not_return_lower_bound_12_px() -> None:
    config = step6_config()
    config["spacing"]["min_spacing_um"] = 1.0
    config["spacing"]["max_spacing_um"] = 2.0
    config["spacing"]["min_periodicity_confidence"] = 0.05
    for period in [16.0, 18.0]:
        image = oriented_stripe_patch(period, np.pi / 2, noise_sigma=0.02)
        result = estimate_spacing_autocorrelation(image, np.pi / 2, config)
        assert result.valid_for_spacing_final
        assert result.patch_spacing_px != 12.0


def test_striated_patch_confidence_exceeds_random_noise() -> None:
    config = step6_config()
    config["spacing"]["min_spacing_um"] = 1.0
    config["spacing"]["max_spacing_um"] = 2.0
    config["spacing"]["min_periodicity_confidence"] = 0.05
    striated = estimate_spacing_autocorrelation(oriented_stripe_patch(16.0, np.pi / 2, noise_sigma=0.02), np.pi / 2, config)
    rng = np.random.default_rng(456)
    noise = rng.normal(0.5, 0.2, size=(128, 128)).astype(np.float32)
    noisy = estimate_spacing_autocorrelation(noise, np.pi / 2, config)
    assert striated.valid_for_spacing_final
    assert striated.patch_spacing_confidence > noisy.patch_spacing_confidence


def test_noise_patch_returns_nan_not_forced_spacing() -> None:
    config = step6_config()
    config["spacing"]["min_periodicity_confidence"] = 0.95
    rng = np.random.default_rng(12)
    image = rng.normal(0.5, 0.2, size=(64, 64)).astype(np.float32)
    result = estimate_spacing_autocorrelation(image, np.pi / 2, config)
    assert not result.valid_for_spacing_final
    assert np.isnan(result.patch_spacing_um)


def test_blank_patch_returns_nan() -> None:
    result = estimate_spacing_autocorrelation(np.zeros((64, 64), dtype=np.float32), np.pi / 2, step6_config())
    assert not result.valid_for_spacing_final
    assert np.isnan(result.patch_spacing_px)


def test_patch_with_invalid_spacing_preserves_invalid_reason() -> None:
    image = stripe_image((64, 64), period=12.0)
    result = compute_spacing_analysis(image, single_patch_metrics(valid_for_spacing=False), step6_config())
    row = result.patch_metrics.iloc[0]
    assert not bool(row["valid_for_spacing_final"])
    assert np.isnan(row["patch_spacing_um"])
    assert "low_tissue_fraction" in row["spacing_invalid_reason"]


def test_missing_orientation_returns_missing_orientation_reason() -> None:
    image = stripe_image((64, 64), period=12.0)
    result = compute_spacing_analysis(image, single_patch_metrics(orientation=float("nan")), step6_config())
    row = result.patch_metrics.iloc[0]
    assert not bool(row["valid_for_spacing_final"])
    assert np.isnan(row["patch_spacing_um"])
    assert "missing_orientation" in row["spacing_invalid_reason"]


def test_spacing_confidence_threshold_is_respected() -> None:
    config = step6_config()
    config["spacing"]["min_periodicity_confidence"] = 10.0
    image = stripe_image((64, 64), period=12.0)
    result = estimate_spacing_autocorrelation(image, np.pi / 2, config)
    assert not result.valid_for_spacing_final
    assert result.spacing_invalid_reason == "low_periodicity_confidence"


def test_patch_metrics_schema_includes_spacing_columns() -> None:
    image = stripe_image((64, 64), period=12.0)
    result = compute_spacing_analysis(image, single_patch_metrics(), step6_config())
    for column in PATCH_SPACING_COLUMNS:
        assert column in result.patch_metrics.columns


def test_image_level_spacing_summary_handles_zero_valid_patches() -> None:
    table = pd.DataFrame(
        {
            "valid_for_spacing_final": [False, False],
            "patch_spacing_um": [np.nan, np.nan],
        }
    )
    summary = summarize_image_spacing(table)
    assert summary["n_spacing_valid_patches"] == 0
    assert summary["spacing_valid_fraction"] == 0.0
    assert np.isnan(summary["image_spacing_mean_um"])


def test_run_image_metrics_spacing_synthetic_smoke(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    image_path = raw_dir / "2.007-1.tif"
    image = (stripe_image((64, 64), period=12.0) * 1000).astype(np.uint16)
    tifffile.imwrite(image_path, image)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  raw_tiff_dir: {raw_dir}
  output_dir: {tmp_path / "results"}
outputs:
  manifest_csv: {tmp_path / "results" / "manifest.csv"}
calibration:
  pixel_size_um: 0.1
  expected_sarcomere_spacing_um:
    min: 1.0
    max: 1.6
filename_pattern:
  regex: '^(?P<donor_id>\\d+\\.\\d+)-(?P<region_id>\\d+)$'
run:
  include_extensions: [.tif, .tiff]
  recursive: false
preprocessing:
  lower_percentile: 1.0
  upper_percentile: 99.0
  background_method: none
  background_sigma_px: 0
  enable_denoise: false
  denoise_sigma_px: 0.5
  output_dtype: float32
  cache_preprocessed: false
masking:
  tissue_method: percentile
  tissue_percentile: 1.0
  min_object_size_px: 4
  fill_holes: true
patches:
  patch_size_px: 32
  stride_px: 32
  margin_px: 0
qc:
  min_tissue_fraction: 0.25
  min_contrast: 0.0
  min_gradient_energy: 0.0
  near_zero_threshold: 0.02
  saturation_threshold: 0.98
orientation:
  tensor_sigma_px: 1.0
  weight_mode: energy_x_coherence
  min_orientation_weight_sum: 1.0e-8
  min_orientation_valid_pixels: 8
  heterogeneity_method: std
  eps: 1.0e-12
spacing:
  method: autocorrelation
  fallback_to_fft: false
  min_spacing_um: 1.0
  max_spacing_um: 1.6
  min_periodicity_confidence: 0.05
  autocorrelation:
    profile_bin_px: 1.0
    min_profile_length_px: 24
    peak_baseline_percentile: 50.0
  fft:
    min_profile_length_px: 24
    peak_prominence_ratio: 4.0
""",
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts" / "run_image_metrics.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path), "--image", str(image_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Spacing scaffold:" in completed.stdout
    assert "n_spacing_valid_patches:" in completed.stdout
