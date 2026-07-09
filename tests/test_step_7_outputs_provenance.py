from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import tifffile

from sarcomere_analysis.outputs import (
    IMAGE_METRICS_REQUIRED_COLUMNS,
    PATCH_METRICS_REQUIRED_COLUMNS,
    ensure_output_dirs,
    write_heatmap,
    write_image_metrics,
    write_mask_overlay,
    write_patch_metrics,
)
from sarcomere_analysis.provenance import collect_run_provenance, write_run_provenance
from test_step_6_spacing import step6_config
from test_step_5_orientation import stripe_image


def minimal_patch_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": "2.007-1",
                "donor_id": "2.007",
                "patch_id": "2.007-1_p00000",
                "y0": 0,
                "x0": 0,
                "y1": 16,
                "x1": 16,
                "center_y": 8,
                "center_x": 8,
                "tissue_fraction": 1.0,
                "intensity_mean": 0.5,
                "intensity_std": 0.1,
                "gradient_energy": 0.01,
                "valid_for_orientation": True,
                "valid_for_periodicity": True,
                "valid_for_spacing": True,
                "invalid_reason": "ok",
                "patch_oop": 0.8,
                "patch_mean_orientation_rad": 0.1,
                "patch_mean_orientation_deg": 5.729,
                "patch_orientation_weight_sum": 1.0,
                "patch_orientation_valid_pixels": 128,
                "patch_spacing_um": np.nan,
                "patch_spacing_px": np.nan,
                "patch_periodicity_score": np.nan,
                "patch_spacing_confidence": 0.0,
                "patch_spacing_method": "autocorrelation",
                "valid_for_spacing_final": False,
                "spacing_invalid_reason": "low_periodicity_confidence",
            }
        ]
    )


def minimal_image_metrics() -> dict:
    return {
        "image_id": "2.007-1",
        "donor_id": "2.007",
        "tissue_fraction": 0.9,
        "total_patches": 1,
        "valid_orientation_patches": 1,
        "image_oop": 0.8,
        "image_mean_orientation_rad": 0.1,
        "image_mean_orientation_deg": 5.729,
        "image_oop_heterogeneity": 0.0,
        "n_orientation_valid_patches": 1,
        "image_spacing_mean_um": np.nan,
        "image_spacing_median_um": np.nan,
        "image_spacing_std_um": np.nan,
        "image_spacing_cv": np.nan,
        "n_spacing_valid_patches": 0,
        "spacing_valid_fraction": 0.0,
    }


def test_output_dirs_are_created_deterministically(tmp_path: Path) -> None:
    config = step6_config(tmp_path)
    dirs = ensure_output_dirs(config)
    assert dirs["tables"] == tmp_path / "tables"
    assert dirs["previews"] == tmp_path / "previews"
    assert dirs["provenance"] == tmp_path / "provenance"
    assert all(path.exists() for path in dirs.values())


def test_patch_metrics_schema_contains_required_columns(tmp_path: Path) -> None:
    config = step6_config(tmp_path)
    path = write_patch_metrics(minimal_patch_table(), "2.007-1", config)
    reloaded = pd.read_csv(path)
    for column in PATCH_METRICS_REQUIRED_COLUMNS:
        assert column in reloaded.columns


def test_image_metrics_schema_contains_required_columns(tmp_path: Path) -> None:
    config = step6_config(tmp_path)
    path = write_image_metrics(minimal_image_metrics(), "2.007-1", config)
    reloaded = pd.read_csv(path)
    for column in IMAGE_METRICS_REQUIRED_COLUMNS:
        assert column in reloaded.columns


def test_heatmap_writer_handles_nan_patch_values(tmp_path: Path) -> None:
    config = step6_config(tmp_path)
    table = minimal_patch_table()
    out = write_heatmap("patch_spacing_um", table, (32, 32), tmp_path / "heatmap.png", config)
    assert out.exists()
    assert out.suffix == ".png"


def test_mask_overlay_writer_creates_png(tmp_path: Path) -> None:
    image = np.zeros((16, 16), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    out = write_mask_overlay(image, mask, tmp_path / "overlay.png")
    assert out.exists()
    assert out.suffix == ".png"


def test_provenance_json_contains_required_fields(tmp_path: Path) -> None:
    config = step6_config(tmp_path)
    provenance = collect_run_provenance(
        config,
        tmp_path / "raw" / "2.007-1.tif",
        "2.007-1",
        {
            "config_path": tmp_path / "config.yaml",
            "input_image_shape": (64, 64),
            "input_image_dtype": "uint16",
            "counts": {
                "total_patches": 4,
                "valid_orientation_patches": 3,
                "valid_spacing_patches": 1,
            },
        },
    )
    path = write_run_provenance(provenance, "2.007-1", config)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["image_id"] == "2.007-1"
    assert loaded["config_hash"]
    assert loaded["python_version"]
    assert loaded["input_image_shape"] == [64, 64]
    assert loaded["counts"]["total_patches"] == 4


def test_provenance_works_when_git_metadata_is_unavailable(tmp_path: Path) -> None:
    config = step6_config(tmp_path)
    provenance = collect_run_provenance(config, tmp_path / "image.tif", "image", {})
    assert "git_commit" in provenance
    assert "git_dirty_status" in provenance


def test_run_image_metrics_synthetic_write_all(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    image_path = raw_dir / "2.007-1.tif"
    tifffile.imwrite(image_path, (stripe_image((64, 64), period=12.0) * 1000).astype(np.uint16))
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
    subprocess.run(
        [sys.executable, str(script), "--config", str(config_path), "--image", str(image_path), "--write-all"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "results" / "tables" / "2.007-1_per_patch_metrics.csv").exists()
    assert (tmp_path / "results" / "tables" / "2.007-1_per_image_metrics.csv").exists()
    assert (tmp_path / "results" / "provenance" / "2.007-1_run_provenance.json").exists()


def test_generated_output_paths_are_under_results(tmp_path: Path) -> None:
    config = step6_config(tmp_path)
    patch_path = write_patch_metrics(minimal_patch_table(), "2.007-1", config)
    image_path = write_image_metrics(minimal_image_metrics(), "2.007-1", config)
    provenance_path = write_run_provenance({"image_id": "2.007-1"}, "2.007-1", config)
    for path in [patch_path, image_path, provenance_path]:
        assert str(path).startswith(str(tmp_path))
        assert "raw" not in path.parts


def test_gitignore_excludes_generated_outputs() -> None:
    gitignore = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "results/previews/" in gitignore
    assert "results/tables/" in gitignore
    assert "results/provenance/" in gitignore
    assert "*.npz" in gitignore
