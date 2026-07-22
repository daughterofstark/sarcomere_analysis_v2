from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from sarcomere_analysis.confocal_intake import (
    build_confocal_manifest,
    discover_confocal_images,
    run_confocal_baseline_audit,
)


def confocal_config(tmp_path: Path) -> dict:
    return {
        "paths": {"raw_tiff_dir": str(tmp_path / "raw"), "output_dir": str(tmp_path / "results")},
        "outputs": {"manifest_csv": str(tmp_path / "results" / "tables" / "manifest.csv")},
        "calibration": {
            "pixel_size_um": 0.1299,
            "expected_sarcomere_spacing_um": {"min": 1.5, "max": 2.4},
        },
        "filename_pattern": {"regex": r"^(?P<donor_id>\d+\.\d+)-(?P<region_id>\d+)$"},
        "run": {"include_extensions": [".tif", ".tiff"], "recursive": False},
        "patches": {"patch_size_px": 32, "stride_px": 32, "margin_px": 0},
        "orientation": {"min_orientation_valid_pixels": 8, "min_orientation_weight_sum": 0.0, "tensor_sigma_px": 1.0},
        "masking": {"min_object_size_px": 4},
        "qc": {"min_tissue_fraction": 0.05, "min_contrast": 0.001, "min_gradient_energy": 0.0},
    }


def write_synthetic_image(path: Path, angle: float = 0.0) -> None:
    y, x = np.mgrid[0:96, 0:96]
    theta = np.deg2rad(angle)
    coord = x * np.cos(theta) + y * np.sin(theta)
    values = 0.5 + 0.4 * np.sin(2 * np.pi * coord / 12.0)
    image = np.clip(values * 255, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def test_discovers_supported_image_files(tmp_path: Path) -> None:
    write_synthetic_image(tmp_path / "6052_good.png")
    write_synthetic_image(tmp_path / "ignore.bmp")
    write_synthetic_image(tmp_path / "nested" / "5138_good.jpg")

    files = discover_confocal_images(tmp_path)

    assert [path.name for path in files] == ["6052_good.png", "5138_good.jpg"]


def test_manifest_records_image_shape_and_dtype(tmp_path: Path) -> None:
    write_synthetic_image(tmp_path / "6052_good.png")
    manifest = build_confocal_manifest(tmp_path)

    assert manifest.loc[0, "image_shape_y"] == 96
    assert manifest.loc[0, "image_shape_x"] == 96
    assert manifest.loc[0, "dtype"] == "uint8"


def test_detects_expected_positive_examples(tmp_path: Path) -> None:
    write_synthetic_image(tmp_path / "sample_6052_region.png")
    write_synthetic_image(tmp_path / "sample_5138_region.png")
    manifest = build_confocal_manifest(tmp_path)

    assert manifest["expected_positive_example"].tolist() == [True, True]


def test_detects_noted_complex_example(tmp_path: Path) -> None:
    write_synthetic_image(tmp_path / "sample_3112_complex.png")
    manifest = build_confocal_manifest(tmp_path)

    assert bool(manifest.loc[0, "noted_complex_example"])


def test_handles_empty_directory_gracefully(tmp_path: Path) -> None:
    manifest, per_image, per_patch, summary, paths = run_confocal_baseline_audit(confocal_config(tmp_path), tmp_path)

    assert manifest.empty
    assert per_image.empty
    assert per_patch.empty
    assert summary["confocal_image_count"] == 0
    assert paths["summary_json"].exists()


def test_does_not_require_pixel_calibration_for_confocal_baseline(tmp_path: Path) -> None:
    write_synthetic_image(tmp_path / "6052_good.png")
    cfg = confocal_config(tmp_path)
    cfg.pop("calibration")

    _, per_image, _, summary, _ = run_confocal_baseline_audit(cfg, tmp_path)

    assert len(per_image) == 1
    assert summary["spacing_calibration_status"] == "confocal_pixel_size_unknown_spacing_um_not_reported"
    assert per_image.loc[0, "spacing_status"] == "not_computed_missing_confocal_pixel_size"


def test_writes_summary_json(tmp_path: Path) -> None:
    write_synthetic_image(tmp_path / "5138_good.png")
    _, _, _, summary, paths = run_confocal_baseline_audit(confocal_config(tmp_path), tmp_path)

    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded["confocal_image_count"] == summary["confocal_image_count"] == 1


def test_per_image_audit_handles_one_synthetic_image(tmp_path: Path) -> None:
    write_synthetic_image(tmp_path / "6052_good.png")
    _, per_image, per_patch, _, _ = run_confocal_baseline_audit(confocal_config(tmp_path), tmp_path)

    assert len(per_image) == 1
    assert per_image.loc[0, "processing_status"] == "ok"
    assert np.isfinite(per_image.loc[0, "image_oop"])
    assert len(per_patch) > 0


def test_preview_flag_off_by_default(tmp_path: Path) -> None:
    write_synthetic_image(tmp_path / "6052_good.png")
    _, _, _, summary, paths = run_confocal_baseline_audit(confocal_config(tmp_path), tmp_path)

    assert summary["previews_written"] is False
    assert summary["preview_paths"] == []
    assert not paths["previews"].exists()


def test_does_not_modify_widefield_outputs(tmp_path: Path) -> None:
    write_synthetic_image(tmp_path / "6052_good.png")
    widefield_table = tmp_path / "results" / "tables" / "features_per_image.csv"
    widefield_table.parent.mkdir(parents=True, exist_ok=True)
    widefield_table.write_text("image_id,image_oop\n2.007-1,0.1\n", encoding="utf-8")
    before = widefield_table.read_bytes()

    run_confocal_baseline_audit(confocal_config(tmp_path), tmp_path)

    assert widefield_table.read_bytes() == before
