from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import tifffile

from sarcomere_analysis.pipeline import WriteOptions, run_single_image
from test_step_5_orientation import stripe_image


def write_step8_config(tmp_path: Path, raw_dir: Path) -> Path:
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
    return config_path


def write_synthetic_tiffs(raw_dir: Path, n: int = 3) -> None:
    raw_dir.mkdir()
    for index in range(1, n + 1):
        image = (stripe_image((64, 64), period=12.0) * (900 + index)).astype(np.uint16)
        tifffile.imwrite(raw_dir / f"2.00{index}-{index}.tif", image)


def script_path(name: str) -> Path:
    return Path(__file__).parents[1] / "scripts" / name


def test_build_manifest_cli_dry_run_synthetic_dir(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=2)
    config_path = write_step8_config(tmp_path, raw_dir)
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path("build_manifest.py")),
            "--config",
            str(config_path),
            "--image-dir",
            str(raw_dir),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Discovered images: 2" in completed.stdout
    assert "Donors: 2" in completed.stdout


def test_build_manifest_empty_dir_allow_empty_and_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config_path = write_step8_config(tmp_path, raw_dir)
    failing = subprocess.run(
        [sys.executable, str(script_path("build_manifest.py")), "--config", str(config_path), "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert failing.returncode != 0
    passing = subprocess.run(
        [
            sys.executable,
            str(script_path("build_manifest.py")),
            "--config",
            str(config_path),
            "--dry-run",
            "--allow-empty",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Discovered images: 0" in passing.stdout


def test_run_image_metrics_cli_writes_tables_and_provenance(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=1)
    config_path = write_step8_config(tmp_path, raw_dir)
    image_path = raw_dir / "2.001-1.tif"
    subprocess.run(
        [
            sys.executable,
            str(script_path("run_image_metrics.py")),
            "--config",
            str(config_path),
            "--image",
            str(image_path),
            "--write-tables",
            "--write-provenance",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "results" / "tables" / "2.001-1_per_patch_metrics.csv").exists()
    assert (tmp_path / "results" / "tables" / "2.001-1_per_image_metrics.csv").exists()
    assert (tmp_path / "results" / "provenance" / "2.001-1_run_provenance.json").exists()


def test_run_batch_metrics_limit_one_writes_combined_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=2)
    config_path = write_step8_config(tmp_path, raw_dir)
    subprocess.run(
        [sys.executable, str(script_path("build_manifest.py")), "--config", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(script_path("run_batch_metrics.py")),
            "--config",
            str(config_path),
            "--limit",
            "1",
            "--write-tables",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    per_image = pd.read_csv(tmp_path / "results" / "tables" / "per_image_metrics.csv")
    per_patch = pd.read_csv(tmp_path / "results" / "tables" / "per_patch_metrics.csv")
    summary = pd.read_csv(tmp_path / "results" / "tables" / "batch_run_summary.csv")
    assert len(per_image) == 1
    assert per_patch["image_id"].nunique() == 1
    assert "status" in summary.columns


def test_batch_manifest_reader_preserves_donor_id_strings(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config_path = write_step8_config(tmp_path, raw_dir)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "image_id": "3.110-1",
                "donor_id": "3.110",
                "region_id": "1",
                "image_path": str(raw_dir / "3.110-1.tif"),
            }
        ]
    ).to_csv(manifest_path, index=False)
    from scripts.run_batch_metrics import load_manifest
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = load_manifest(cfg, str(manifest_path))
    assert manifest.loc[0, "donor_id"] == "3.110"


def test_continue_on_error_records_error_row(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=1)
    config_path = write_step8_config(tmp_path, raw_dir)
    manifest = pd.DataFrame(
        [
            {"image_id": "2.001-1", "donor_id": "2.001", "image_path": str(raw_dir / "2.001-1.tif")},
            {"image_id": "9.999-1", "donor_id": "9.999", "image_path": str(raw_dir / "missing.tif")},
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    subprocess.run(
        [
            sys.executable,
            str(script_path("run_batch_metrics.py")),
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--continue-on-error",
            "--write-tables",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = pd.read_csv(tmp_path / "results" / "tables" / "batch_run_summary.csv")
    assert set(summary["status"]) == {"ok", "error"}


def test_limit_limits_processed_images(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=3)
    config_path = write_step8_config(tmp_path, raw_dir)
    subprocess.run(
        [sys.executable, str(script_path("run_batch_metrics.py")), "--config", str(config_path), "--limit", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = pd.read_csv(tmp_path / "results" / "tables" / "batch_run_summary.csv")
    assert len(summary) == 2


def test_run_single_image_returns_stable_paths_and_counts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=1)
    config_path = write_step8_config(tmp_path, raw_dir)
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result = run_single_image(
        raw_dir / "2.001-1.tif",
        "2.001-1",
        "2.001",
        cfg,
        WriteOptions(tables=True, preview=False, provenance=True),
        config_path=config_path,
    )
    assert result.total_patches == 4
    assert result.valid_orientation_patches == 4
    assert "per_patch_metrics" in result.output_paths
    assert "run_provenance" in result.output_paths
