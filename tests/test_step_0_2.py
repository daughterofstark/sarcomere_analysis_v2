from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from sarcomere_analysis.config import get_calibration, load_config
from sarcomere_analysis.io import (
    build_manifest,
    discover_tiffs,
    load_tiff,
    parse_image_filename,
    write_table,
)


def test_config_loads_and_calibration_converts() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    calibration = get_calibration(config)
    assert calibration.pixel_size_um == 0.1299
    assert round(calibration.pixels_per_um, 4) == round(1 / 0.1299, 4)
    assert calibration.expected_spacing_px_min == calibration.expected_spacing_um_min / calibration.pixel_size_um
    assert calibration.expected_spacing_px_max == calibration.expected_spacing_um_max / calibration.pixel_size_um


def test_filename_parsing() -> None:
    parsed = parse_image_filename("2.007-1.tif", r"^(?P<donor_id>\d+\.\d+)-(?P<region_id>\d+)$")
    assert parsed["image_id"] == "2.007-1"
    assert parsed["donor_id"] == "2.007"
    assert parsed["region_id"] == "1"


def test_discover_manifest_and_load_tiff(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    tifffile.imwrite(raw_dir / "2.007-1.tif", np.arange(16, dtype=np.uint16).reshape(4, 4))
    tifffile.imwrite(raw_dir / "2.007-2.tiff", np.ones((4, 4), dtype=np.uint16))
    (raw_dir / "ignore.txt").write_text("not an image", encoding="utf-8")

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

    files = discover_tiffs(config)
    assert [path.name for path in files] == ["2.007-1.tif", "2.007-2.tiff"]

    manifest = build_manifest(config)
    assert list(manifest["image_id"]) == ["2.007-1", "2.007-2"]
    assert list(manifest["donor_id"].unique()) == ["2.007"]
    assert "expected_spacing_px_min" in manifest.columns
    assert "expected_spacing_px_max" in manifest.columns

    image = load_tiff(raw_dir / "2.007-1.tif")
    assert image.shape == (4, 4)

    out = tmp_path / "tables" / "manifest.csv"
    write_table(manifest, out)
    reloaded = pd.read_csv(out)
    assert len(reloaded) == 2
