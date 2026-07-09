from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import tifffile

from sarcomere_analysis.masking import compute_tissue_mask
from sarcomere_analysis.orientation import (
    ORIENTATION_COLUMNS,
    axial_order_parameter,
    compute_orientation_analysis,
    orientation_params,
    structure_tensor_orientation,
)
from sarcomere_analysis.qc import compute_patch_qc
from test_step_4_masks_patches_qc import step4_config


def step5_config(tmp_path: Path | None = None) -> dict:
    config = step4_config(tmp_path)
    config["orientation"] = {
        "tensor_sigma_px": 1.0,
        "weight_mode": "energy_x_coherence",
        "min_orientation_weight_sum": 1e-8,
        "min_orientation_valid_pixels": 8,
        "heterogeneity_method": "std",
        "eps": 1e-12,
    }
    return config


def stripe_image(shape: tuple[int, int] = (64, 64), period: float = 8.0) -> np.ndarray:
    y = np.arange(shape[0], dtype=np.float32)[:, None]
    image = 0.5 + 0.5 * np.sin(2.0 * np.pi * y / period)
    return np.repeat(image, shape[1], axis=1).astype(np.float32)


def test_orientation_maps_match_image_shape() -> None:
    image = stripe_image()
    params = orientation_params(step5_config())
    theta, coherence, energy = structure_tensor_orientation(image, params)
    assert theta.shape == image.shape
    assert coherence.shape == image.shape
    assert energy.shape == image.shape


def test_coherence_and_energy_are_safe_ranges() -> None:
    image = stripe_image()
    params = orientation_params(step5_config())
    _, coherence, energy = structure_tensor_orientation(image, params)
    assert np.isfinite(coherence).all()
    assert float(coherence.min()) >= -1e-6
    assert float(coherence.max()) <= 1.0 + 1e-6
    assert np.isfinite(energy).all()
    assert float(energy.min()) >= -1e-12


def test_image_oop_is_nan_or_unit_interval() -> None:
    image = stripe_image()
    mask = np.ones_like(image, dtype=bool)
    patch_qc = compute_patch_qc(image, mask, "synthetic", step5_config())
    result = compute_orientation_analysis(image, mask, patch_qc, step5_config())
    value = result.image_metrics["image_oop"]
    assert np.isnan(value) or 0.0 <= value <= 1.0


def test_synthetic_stripes_give_high_oop() -> None:
    image = stripe_image()
    mask = np.ones_like(image, dtype=bool)
    config = step5_config()
    patch_qc = compute_patch_qc(image, mask, "synthetic", config)
    result = compute_orientation_analysis(image, mask, patch_qc, config)
    assert result.image_metrics["image_oop"] > 0.90


def test_random_noise_does_not_force_patch_oop_when_qc_fails() -> None:
    rng = np.random.default_rng(3)
    image = rng.normal(0.5, 0.01, size=(64, 64)).astype(np.float32)
    mask = np.zeros_like(image, dtype=bool)
    config = step5_config()
    patch_qc = compute_patch_qc(image, mask, "synthetic", config)
    result = compute_orientation_analysis(image, mask, patch_qc, config)
    assert result.patch_metrics["patch_oop"].isna().all()


def test_patch_metrics_schema_includes_orientation_columns() -> None:
    image = stripe_image()
    mask = np.ones_like(image, dtype=bool)
    config = step5_config()
    patch_qc = compute_patch_qc(image, mask, "synthetic", config)
    result = compute_orientation_analysis(image, mask, patch_qc, config)
    for column in ORIENTATION_COLUMNS:
        assert column in result.patch_metrics.columns


def test_invalid_orientation_patches_get_nan_metrics() -> None:
    patch_qc = pd.DataFrame(
        [
            {
                "image_id": "synthetic",
                "patch_id": "synthetic_p00000",
                "y0": 0,
                "x0": 0,
                "y1": 16,
                "x1": 16,
                "center_y": 8,
                "center_x": 8,
                "tissue_fraction": 0.0,
                "intensity_mean": 0.0,
                "intensity_std": 0.0,
                "rms_contrast": 0.0,
                "gradient_energy": 0.0,
                "near_zero_fraction": 1.0,
                "saturated_fraction": 0.0,
                "valid_for_orientation": False,
                "valid_for_periodicity": False,
                "valid_for_spacing": False,
                "invalid_reason": "low_tissue_fraction",
            }
        ]
    )
    image = np.ones((16, 16), dtype=np.float32)
    mask = np.ones_like(image, dtype=bool)
    result = compute_orientation_analysis(image, mask, patch_qc, step5_config())
    assert np.isnan(result.patch_metrics.loc[0, "patch_oop"])
    assert np.isnan(result.patch_metrics.loc[0, "patch_mean_orientation_rad"])


def test_axial_circular_mean_treats_pi_equivalent_angles_as_same() -> None:
    theta = np.array([0.1, 0.1 + np.pi], dtype=np.float32)
    weights = np.ones_like(theta)
    valid = np.ones_like(theta, dtype=bool)
    oop, mean_rad, weight_sum, valid_pixels = axial_order_parameter(theta, weights, valid, 0.0, 2)
    assert oop > 0.99
    assert abs(mean_rad - 0.1) < 1e-5
    assert weight_sum == 2.0
    assert valid_pixels == 2


def test_run_image_metrics_orientation_synthetic_smoke(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    image_path = raw_dir / "2.007-1.tif"
    image = (stripe_image((64, 64)) * 1000).astype(np.uint16)
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
  pixel_size_um: 0.1299
  expected_sarcomere_spacing_um:
    min: 1.5
    max: 2.4
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
  patch_size_px: 16
  stride_px: 16
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
    assert "Orientation/OOP:" in completed.stdout
    assert "image_oop:" in completed.stdout
