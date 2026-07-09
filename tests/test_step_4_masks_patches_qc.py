from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import tifffile

from sarcomere_analysis.masking import compute_tissue_mask
from sarcomere_analysis.patches import patch_grid_table
from sarcomere_analysis.qc import PATCH_QC_COLUMNS, compute_patch_qc


def step4_config(tmp_path: Path | None = None) -> dict:
    output = str((tmp_path or Path(".")).resolve())
    return {
        "paths": {"raw_tiff_dir": output, "output_dir": output},
        "outputs": {"manifest_csv": str(Path(output) / "manifest.csv")},
        "calibration": {
            "pixel_size_um": 0.1299,
            "expected_sarcomere_spacing_um": {"min": 1.5, "max": 2.4},
        },
        "filename_pattern": {"regex": r"^(?P<donor_id>\d+\.\d+)-(?P<region_id>\d+)$"},
        "run": {"include_extensions": [".tif", ".tiff"], "recursive": False},
        "preprocessing": {
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "background_method": "none",
            "background_sigma_px": 0.0,
            "enable_denoise": False,
            "denoise_sigma_px": 0.5,
            "output_dtype": "float32",
            "cache_preprocessed": False,
        },
        "masking": {
            "tissue_method": "percentile",
            "tissue_percentile": 50.0,
            "min_object_size_px": 4,
            "fill_holes": True,
        },
        "patches": {"patch_size_px": 16, "stride_px": 16, "margin_px": 0},
        "qc": {
            "min_tissue_fraction": 0.25,
            "min_contrast": 0.01,
            "min_gradient_energy": 0.00001,
            "near_zero_threshold": 0.02,
            "saturation_threshold": 0.98,
        },
    }


def test_tissue_mask_shape_and_bool_dtype() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 8:24] = 1.0
    result = compute_tissue_mask(image, step4_config())
    assert result.mask.shape == image.shape
    assert result.mask.dtype == bool


def test_blank_image_safely_returns_mostly_false_mask() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    result = compute_tissue_mask(image, step4_config())
    assert float(result.mask.mean()) == 0.0


def test_synthetic_bright_object_produces_nonzero_tissue_mask() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 8:24] = 1.0
    result = compute_tissue_mask(image, step4_config())
    assert int(result.mask.sum()) > 0


def test_patch_grid_coordinates_stay_within_bounds() -> None:
    table = patch_grid_table((40, 48), "synthetic", step4_config())
    assert len(table) > 0
    assert (table["y0"] >= 0).all()
    assert (table["x0"] >= 0).all()
    assert (table["y1"] <= 40).all()
    assert (table["x1"] <= 48).all()


def test_patch_qc_table_schema_flags_and_reasons() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[:, 16:] = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    mask = image > 0.25
    table = compute_patch_qc(image, mask, "synthetic", step4_config())
    assert list(table.columns) == PATCH_QC_COLUMNS
    assert table["valid_for_orientation"].map(type).eq(bool).all()
    assert table["valid_for_periodicity"].map(type).eq(bool).all()
    assert table["valid_for_spacing"].map(type).eq(bool).all()
    assert table["invalid_reason"].notna().all()
    assert table["invalid_reason"].str.len().gt(0).all()


def test_low_tissue_patches_are_invalid() -> None:
    image = np.ones((32, 32), dtype=np.float32) * 0.5
    mask = np.zeros((32, 32), dtype=bool)
    table = compute_patch_qc(image, mask, "synthetic", step4_config())
    assert not table["valid_for_orientation"].any()
    assert not table["valid_for_periodicity"].any()
    assert table["invalid_reason"].str.contains("low_tissue_fraction").all()


def test_run_image_metrics_synthetic_smoke(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    image_path = raw_dir / "2.007-1.tif"
    image = np.zeros((64, 64), dtype=np.uint16)
    image[16:48, 16:48] = 1000
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
  tissue_percentile: 50.0
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
    assert "Patch QC:" in completed.stdout
    assert "patches:" in completed.stdout
