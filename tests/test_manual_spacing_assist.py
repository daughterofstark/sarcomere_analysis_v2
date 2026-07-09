from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sarcomere_analysis.manual_spacing_assist import (
    MANUAL_SPACING_COLUMNS,
    analyze_spacing_profile,
    run_manual_spacing_assist,
    write_diagnostic_panel,
    write_manual_spacing_result,
)


def spacing_config(tmp_path: Path) -> dict:
    return {
        "paths": {"raw_tiff_dir": str(tmp_path / "raw"), "output_dir": str(tmp_path / "results")},
        "outputs": {"manifest_csv": str(tmp_path / "results" / "tables" / "manifest.csv")},
        "calibration": {
            "pixel_size_um": 0.1299,
            "expected_sarcomere_spacing_um": {"min": 1.5, "max": 2.4},
        },
        "filename_pattern": {"regex": r"^(?P<donor_id>\d+\.\d+)-(?P<region_id>\d+)$"},
        "run": {"include_extensions": [".tif", ".tiff"], "recursive": False},
    }


def write_stripe_png(path: Path, period_px: float = 12.0, shape: tuple[int, int] = (64, 64)) -> Path:
    yy, xx = np.indices(shape, dtype=np.float32)
    image = 0.5 + 0.4 * np.sin(2 * np.pi * xx / period_px)
    png = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(png, mode="L").save(path)
    return path


def write_blank_png(path: Path, shape: tuple[int, int] = (64, 64)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(shape, 128, dtype=np.uint8), mode="L").save(path)
    return path


def test_line_length_conversion_uses_config_pixel_size(tmp_path: Path) -> None:
    cfg = spacing_config(tmp_path)
    crop = write_stripe_png(tmp_path / "ANN_0001__img__patch.png")
    row, _ = run_manual_spacing_assist(cfg, crop, 0, 0, 10, 0, intervals=1)
    assert abs(row["line_length_um"] - 1.299) < 1e-6


def test_interval_count_spacing_is_correct(tmp_path: Path) -> None:
    cfg = spacing_config(tmp_path)
    crop = write_stripe_png(tmp_path / "ANN_0001__img__patch.png")
    row, _ = run_manual_spacing_assist(cfg, crop, 0, 0, 60, 0, intervals=3)
    assert abs(row["estimated_spacing_px"] - 20.0) < 1e-6
    assert abs(row["estimated_spacing_um"] - (20.0 * 0.1299)) < 1e-6


def test_peak_based_spacing_works_on_synthetic_stripes(tmp_path: Path) -> None:
    cfg = spacing_config(tmp_path)
    crop = write_stripe_png(tmp_path / "ANN_0001__img__patch.png", period_px=12)
    row, _ = run_manual_spacing_assist(cfg, crop, 0, 32, 63, 32, intervals=None, min_peak_prominence=0.05)
    assert np.isfinite(row["estimated_spacing_px"])
    assert abs(row["estimated_spacing_px"] - 12.0) <= 1.0


def test_blank_crop_returns_nan_or_low_confidence_safely(tmp_path: Path) -> None:
    cfg = spacing_config(tmp_path)
    crop = write_blank_png(tmp_path / "ANN_0001__img__patch.png")
    row, _ = run_manual_spacing_assist(cfg, crop, 0, 32, 63, 32)
    assert row["detected_peak_count"] == 0
    assert np.isnan(row["estimated_spacing_px"])
    assert np.isnan(row["estimated_spacing_um"])


def test_output_csv_has_required_columns(tmp_path: Path) -> None:
    cfg = spacing_config(tmp_path)
    crop = write_stripe_png(tmp_path / "ANN_0001__img__patch.png")
    row, _ = run_manual_spacing_assist(cfg, crop, 0, 0, 60, 0, intervals=3)
    path = write_manual_spacing_result(row, tmp_path / "results.csv")
    table = pd.read_csv(path)
    assert list(table.columns) == MANUAL_SPACING_COLUMNS


def test_diagnostic_panel_can_be_saved(tmp_path: Path) -> None:
    cfg = spacing_config(tmp_path)
    crop = write_stripe_png(tmp_path / "ANN_0001__img__patch.png")
    row, result = run_manual_spacing_assist(cfg, crop, 0, 32, 63, 32, min_peak_prominence=0.05)
    panel = write_diagnostic_panel(result, tmp_path / "panel.png", row)
    assert panel.exists()
    assert panel.stat().st_size > 0


def test_no_production_tables_are_modified(tmp_path: Path) -> None:
    cfg = spacing_config(tmp_path)
    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    production = tables / "analysis_per_image.csv"
    production.write_text("image_id,donor_id\n2.007-1,2.007\n", encoding="utf-8")
    before = production.read_bytes()
    crop = write_stripe_png(tmp_path / "ANN_0001__img__patch.png")
    row, _ = run_manual_spacing_assist(cfg, crop, 0, 0, 60, 0, intervals=3)
    write_manual_spacing_result(row, tmp_path / "results" / "annotation_pack" / "manual_spacing_assist_results.csv")
    assert production.read_bytes() == before


def test_analyze_spacing_profile_accepts_direct_array() -> None:
    image = np.tile((0.5 + 0.4 * np.sin(2 * np.pi * np.arange(64) / 8.0)).astype(np.float32), (32, 1))
    result = analyze_spacing_profile(image, 0, 16, 63, 16, pixel_size_um=0.1299, min_peak_prominence=0.05)
    assert np.isfinite(result.estimated_spacing_px)
    assert abs(result.estimated_spacing_px - 8.0) <= 1.0
