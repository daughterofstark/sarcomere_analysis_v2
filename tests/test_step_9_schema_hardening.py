from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from sarcomere_analysis.config import get_calibration, load_config
from sarcomere_analysis.io import build_manifest, parse_image_filename
from sarcomere_analysis.outputs import write_image_metrics, write_patch_metrics
from sarcomere_analysis.preprocessing import preprocess_image
from sarcomere_analysis.schemas import (
    BATCH_RUN_SUMMARY_COLUMNS,
    IMAGE_METRICS_COLUMNS,
    MANIFEST_COLUMNS,
    PATCH_METRICS_COLUMNS,
    stabilize_columns,
)
from test_step_7_outputs_provenance import minimal_image_metrics, minimal_patch_table


def test_default_and_mutated_calibration_drive_spacing_px(tmp_path: Path) -> None:
    default_config = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert get_calibration(default_config).pixel_size_um == 0.1299

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  raw_tiff_dir: /tmp/raw
  output_dir: /tmp/results
outputs:
  manifest_csv: /tmp/results/manifest.csv
calibration:
  pixel_size_um: 0.2
  expected_sarcomere_spacing_um:
    min: 1.0
    max: 2.0
filename_pattern:
  regex: '^(?P<donor_id>\\d+\\.\\d+)-(?P<region_id>\\d+)$'
run:
  include_extensions: [.tif, .tiff]
  recursive: false
""",
        encoding="utf-8",
    )
    mutated = get_calibration(load_config(config_path))
    assert mutated.expected_spacing_px_min == 5.0
    assert mutated.expected_spacing_px_max == 10.0


def test_filename_examples_and_invalid_controlled_error() -> None:
    regex = r"^(?P<donor_id>\d+\.\d+)-(?P<region_id>\d+)$"
    assert parse_image_filename("2.007-1.tif", regex)["donor_id"] == "2.007"
    parsed = parse_image_filename("4.083-5.tiff", regex)
    assert parsed["image_id"] == "4.083-5"
    assert parsed["region_id"] == "5"
    with pytest.raises(ValueError):
        parse_image_filename("not-valid.tif", regex)


def test_manifest_schema_and_no_duplicate_image_ids(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    tifffile.imwrite(raw_dir / "2.007-1.tif", np.zeros((4, 4), dtype=np.uint16))
    tifffile.imwrite(raw_dir / "4.083-5.tiff", np.ones((4, 4), dtype=np.uint16))
    config = {
        "paths": {"raw_tiff_dir": str(raw_dir), "output_dir": str(tmp_path / "results")},
        "outputs": {"manifest_csv": str(tmp_path / "results" / "manifest.csv")},
        "calibration": {
            "pixel_size_um": 0.1299,
            "expected_sarcomere_spacing_um": {"min": 1.5, "max": 2.4},
        },
        "filename_pattern": {"regex": r"^(?P<donor_id>\d+\.\d+)-(?P<region_id>\d+)$"},
        "run": {"include_extensions": [".tif", ".tiff"], "recursive": False},
    }
    manifest = build_manifest(config)
    assert list(manifest.columns[: len(MANIFEST_COLUMNS)]) == MANIFEST_COLUMNS
    assert {"image_id", "donor_id"}.issubset(manifest.columns)
    assert not manifest["image_id"].duplicated().any()


def test_saturated_image_preprocessing_safe() -> None:
    result = preprocess_image(np.full((16, 16), 65535, dtype=np.uint16), {"preprocessing": {"background_method": "none"}})
    assert result.image.shape == (16, 16)
    assert np.isfinite(result.image).all()
    assert result.image.dtype == np.float32


def test_patch_and_image_schema_exact_order_on_write(tmp_path: Path) -> None:
    config = {"paths": {"output_dir": str(tmp_path)}}
    patch_path = write_patch_metrics(minimal_patch_table(), "2.007-1", config)
    image_path = write_image_metrics(minimal_image_metrics(), "2.007-1", config)
    patch = pd.read_csv(patch_path)
    image = pd.read_csv(image_path)
    assert list(patch.columns[: len(PATCH_METRICS_COLUMNS)]) == PATCH_METRICS_COLUMNS
    assert list(image.columns[: len(IMAGE_METRICS_COLUMNS)]) == IMAGE_METRICS_COLUMNS


def test_batch_summary_schema_order_and_optional_fill() -> None:
    table = stabilize_columns(pd.DataFrame([{"image_id": "x", "status": "ok"}]), BATCH_RUN_SUMMARY_COLUMNS)
    assert list(table.columns) == BATCH_RUN_SUMMARY_COLUMNS
    assert pd.isna(table.loc[0, "runtime_seconds"])


def test_required_core_schema_columns_raise_when_missing() -> None:
    with pytest.raises(ValueError):
        stabilize_columns(pd.DataFrame([{"patch_id": "p0"}]), PATCH_METRICS_COLUMNS, required_core=["image_id"])
