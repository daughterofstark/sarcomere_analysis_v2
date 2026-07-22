from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from sarcomere_analysis.confocal_metadata import (
    CONFOCAL_METADATA_COLUMNS,
    audit_confocal_metadata,
    extract_pixel_size_metadata,
)


def metadata_config(tmp_path: Path) -> dict:
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


def write_imagej_tiff(path: Path, pixels_per_micron_x: float = 10.0, pixels_per_micron_y: float = 10.0) -> None:
    image = np.arange(64, dtype=np.uint16).reshape(8, 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        image,
        imagej=True,
        metadata={"unit": "micron"},
        resolution=(pixels_per_micron_x, pixels_per_micron_y),
    )


def write_plain_tiff(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.zeros((8, 8), dtype=np.uint16))


def write_manifest(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def manifest_row(image_id: str, path: Path, expected: bool = False, complex_flag: bool = False) -> dict:
    return {
        "confocal_image_id": image_id,
        "filename": path.name,
        "source_path": str(path),
        "image_shape_y": 8,
        "image_shape_x": 8,
        "expected_positive_example": expected,
        "noted_complex_example": complex_flag,
    }


def test_handles_missing_metadata_gracefully(tmp_path: Path) -> None:
    image_path = tmp_path / "plain.tif"
    write_plain_tiff(image_path)

    metadata = extract_pixel_size_metadata(image_path)

    assert metadata["pixel_size_available"] is False
    assert metadata["pixel_size_source"] != "widefield_config"


def test_records_no_fallback_to_widefield_calibration(tmp_path: Path) -> None:
    image_path = tmp_path / "plain.tif"
    write_plain_tiff(image_path)
    manifest = write_manifest(tmp_path, [manifest_row("plain", image_path)])

    calibration, summary, _ = audit_confocal_metadata(
        metadata_config(tmp_path),
        confocal_manifest=manifest,
        output_directory=tmp_path / "out",
    )

    assert bool(calibration.loc[0, "pixel_size_available"]) is False
    assert "do_not_use_widefield_fallback" in calibration.loc[0, "calibration_warning"]
    assert calibration.loc[0, "spacing_um_policy"] == "disabled_missing_per_image_pixel_size"
    assert summary["widefield_calibration_used"] is False


def test_parses_imagej_style_pixel_size_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "imagej.tif"
    write_imagej_tiff(image_path, pixels_per_micron_x=10.0, pixels_per_micron_y=10.0)

    metadata = extract_pixel_size_metadata(image_path)

    assert metadata["pixel_size_available"] is True
    assert round(metadata["pixel_size_x_um"], 4) == 0.1
    assert metadata["pixel_size_source"] == "imagej_unit_resolution_tags"


def test_records_per_image_differing_pixel_sizes(tmp_path: Path) -> None:
    image_a = tmp_path / "a.tif"
    image_b = tmp_path / "b.tif"
    write_imagej_tiff(image_a, pixels_per_micron_x=10.0, pixels_per_micron_y=10.0)
    write_imagej_tiff(image_b, pixels_per_micron_x=20.0, pixels_per_micron_y=20.0)
    manifest = write_manifest(tmp_path, [manifest_row("a", image_a), manifest_row("b", image_b)])

    _, summary, _ = audit_confocal_metadata(metadata_config(tmp_path), manifest, tmp_path / "out")

    assert summary["pixel_size_available_count"] == 2
    assert summary["spacing_um_enabled_count"] == 2
    assert summary["pixel_sizes_differ_across_images"] is True
    assert len(summary["unique_pixel_sizes_um"]) == 2


def test_detects_isotropic_vs_anisotropic_pixels(tmp_path: Path) -> None:
    iso = tmp_path / "iso.tif"
    aniso = tmp_path / "aniso.tif"
    write_imagej_tiff(iso, pixels_per_micron_x=10.0, pixels_per_micron_y=10.0)
    write_imagej_tiff(aniso, pixels_per_micron_x=10.0, pixels_per_micron_y=20.0)
    manifest = write_manifest(tmp_path, [manifest_row("iso", iso), manifest_row("aniso", aniso)])

    calibration, summary, _ = audit_confocal_metadata(metadata_config(tmp_path), manifest, tmp_path / "out")

    assert bool(calibration.loc[calibration["confocal_image_id"] == "iso", "isotropic_pixels"].iloc[0]) is True
    assert bool(calibration.loc[calibration["confocal_image_id"] == "aniso", "isotropic_pixels"].iloc[0]) is False
    assert summary["anisotropic_pixel_count"] == 1


def test_writes_manual_template_for_missing_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "plain.tif"
    write_plain_tiff(image_path)
    manifest = write_manifest(tmp_path, [manifest_row("plain", image_path)])

    _, summary, paths = audit_confocal_metadata(
        metadata_config(tmp_path),
        confocal_manifest=manifest,
        output_directory=tmp_path / "out",
        write_manual_template=True,
    )
    template = pd.read_csv(paths["manual_template"], dtype=str)

    assert summary["manual_template_requested"] is True
    assert len(template) == 1
    assert template.loc[0, "confocal_image_id"] == "plain"


def test_summary_json_serializable(tmp_path: Path) -> None:
    image_path = tmp_path / "imagej.tif"
    write_imagej_tiff(image_path)
    manifest = write_manifest(tmp_path, [manifest_row("imagej", image_path)])

    _, summary, paths = audit_confocal_metadata(metadata_config(tmp_path), manifest, tmp_path / "out")

    assert set(CONFOCAL_METADATA_COLUMNS).issubset(pd.read_csv(paths["calibration"]).columns)
    json.dumps(summary)
    assert paths["summary_json"].exists()


def test_existing_widefield_and_confocal_outputs_not_modified(tmp_path: Path) -> None:
    image_path = tmp_path / "imagej.tif"
    write_imagej_tiff(image_path)
    manifest = write_manifest(tmp_path, [manifest_row("imagej", image_path)])
    existing_widefield = tmp_path / "results" / "tables" / "features_per_image.csv"
    existing_confocal = tmp_path / "results" / "confocal_baseline" / "confocal_baseline_per_image.csv"
    existing_widefield.parent.mkdir(parents=True, exist_ok=True)
    existing_confocal.parent.mkdir(parents=True, exist_ok=True)
    existing_widefield.write_text("image_id,image_oop\n2.007-1,0.1\n", encoding="utf-8")
    existing_confocal.write_text("confocal_image_id,image_oop\nimagej,0.2\n", encoding="utf-8")
    before_widefield = existing_widefield.read_bytes()
    before_confocal = existing_confocal.read_bytes()

    audit_confocal_metadata(metadata_config(tmp_path), manifest, tmp_path / "out")

    assert existing_widefield.read_bytes() == before_widefield
    assert existing_confocal.read_bytes() == before_confocal
