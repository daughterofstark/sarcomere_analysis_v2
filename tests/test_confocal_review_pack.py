from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from PIL import Image

from sarcomere_analysis.confocal_review_pack import (
    DEFAULT_REVIEW_IMAGES,
    REVIEW_SUMMARY_COLUMNS,
    export_confocal_review_pack,
)


def review_config(tmp_path: Path) -> dict:
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


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(path)


def write_inputs(tmp_path: Path, include_previews: bool = True) -> None:
    root = tmp_path / "results"
    (root / "confocal_pilot").mkdir(parents=True, exist_ok=True)
    (root / "confocal_pilot" / "confocal_pilot_interpretation.txt").write_text("pilot report\n", encoding="utf-8")
    spacing = []
    oop = []
    metadata = []
    for image_id in DEFAULT_REVIEW_IMAGES:
        expected = image_id in {"5138", "6052-CLEAR_STRIPES"}
        complex_flag = image_id == "3112"
        spacing.append(
            {
                "confocal_image_id": image_id,
                "filename": f"{image_id}.tif",
                "total_patches": 100,
                "candidate_patch_count": 25,
                "spacing_valid_patch_count_selected": 10,
                "spacing_valid_fraction_selected": 0.4,
                "selected_median_spacing_um": 1.8,
                "selected_iqr_spacing_um": 0.2,
                "expected_positive_example": expected,
                "noted_complex_example": complex_flag,
            }
        )
        oop.append(
            {
                "confocal_image_id": image_id,
                "filename": f"{image_id}.tif",
                "selected_region_median_oop_128": 0.7,
                "all_region_median_oop_128": 0.6,
                "selected_vs_all_oop_difference_128": 0.1,
                "expected_positive_example": expected,
                "noted_complex_example": complex_flag,
            }
        )
        metadata.append(
            {
                "confocal_image_id": image_id,
                "filename": f"{image_id}.tif",
                "pixel_size_x_um": 0.06,
                "pixel_size_available": True,
            }
        )
    spacing_dir = root / "confocal_spacing_audit"
    oop_dir = root / "confocal_same_grid_oop"
    metadata_dir = root / "confocal_metadata"
    spacing_dir.mkdir(parents=True, exist_ok=True)
    oop_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(spacing).to_csv(spacing_dir / "confocal_spacing_per_image.csv", index=False)
    pd.DataFrame({"confocal_image_id": [], "patch_id": []}).to_csv(spacing_dir / "confocal_spacing_per_patch.csv", index=False)
    pd.DataFrame(oop).to_csv(oop_dir / "confocal_same_grid_oop_per_image.csv", index=False)
    pd.DataFrame(metadata).to_csv(metadata_dir / "confocal_metadata_calibration.csv", index=False)
    if include_previews:
        for image_id in DEFAULT_REVIEW_IMAGES:
            write_png(root / "confocal_selective_analysis" / "previews" / f"{image_id}_selected_candidate_overlay.png")
            write_png(root / "confocal_spacing_audit" / "previews" / f"{image_id}_confocal_spacing_candidate_overlay.png")
            write_png(root / "confocal_spacing_audit" / "previews" / f"{image_id}_confocal_valid_spacing_overlay.png")
            write_png(root / "confocal_spacing_audit" / "previews" / f"{image_id}_confocal_spacing_um_heatmap.png")
            write_png(root / "confocal_same_grid_oop" / "previews" / f"{image_id}_same_grid_candidate_overlay.png")
            write_png(root / "confocal_same_grid_oop" / "previews" / f"{image_id}_same_grid_oop_heatmap.png")


def test_selects_default_review_images(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    _, summary, _ = export_confocal_review_pack(review_config(tmp_path), output_directory=tmp_path / "out")

    assert summary["images_included"] == DEFAULT_REVIEW_IMAGES


def test_summary_csv_includes_required_columns(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    review_summary, _, paths = export_confocal_review_pack(review_config(tmp_path), output_directory=tmp_path / "out")

    assert list(review_summary.columns) == REVIEW_SUMMARY_COLUMNS
    written = pd.read_csv(paths["summary_csv"])
    assert list(written.columns) == REVIEW_SUMMARY_COLUMNS


def test_handles_missing_preview_files_gracefully(tmp_path: Path) -> None:
    write_inputs(tmp_path, include_previews=False)
    _, summary, _ = export_confocal_review_pack(review_config(tmp_path), output_directory=tmp_path / "out")

    assert summary["review_image_files_copied"] == 0
    assert summary["missing_preview_count"] > 0


def test_review_notes_are_written(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    _, _, paths = export_confocal_review_pack(review_config(tmp_path), output_directory=tmp_path / "out")

    text = paths["notes_md"].read_text(encoding="utf-8")
    assert "5138" in text
    assert "7028" in text


def test_zip_includes_review_files(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    _, _, paths = export_confocal_review_pack(review_config(tmp_path), output_directory=tmp_path / "out", write_zip=True)

    with ZipFile(paths["zip"]) as archive:
        names = set(archive.namelist())
    assert "confocal_review_summary.csv" in names
    assert "confocal_review_notes_for_natalia.md" in names
    assert "confocal_review_pack_summary.txt" in names
    assert any(name.startswith("review_images/") and name.endswith(".png") for name in names)


def test_zip_excludes_internal_raw_large_tables(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    _, _, paths = export_confocal_review_pack(review_config(tmp_path), output_directory=tmp_path / "out", write_zip=True)

    with ZipFile(paths["zip"]) as archive:
        names = archive.namelist()
    assert not any("confocal_spacing_per_patch" in name for name in names)
    assert not any("confocal_metadata_calibration" in name for name in names)


def test_summary_json_serializable(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    _, summary, paths = export_confocal_review_pack(review_config(tmp_path), output_directory=tmp_path / "out")

    json.dumps(summary)
    assert paths["summary_json"].exists()


def test_existing_confocal_and_widefield_outputs_not_modified(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    widefield = tmp_path / "results" / "tables" / "features_per_image.csv"
    source_spacing = tmp_path / "results" / "confocal_spacing_audit" / "confocal_spacing_per_image.csv"
    widefield.parent.mkdir(parents=True, exist_ok=True)
    widefield.write_text("image_id,image_oop\n2.007-1,0.1\n", encoding="utf-8")
    before_widefield = widefield.read_bytes()
    before_spacing = source_spacing.read_bytes()

    export_confocal_review_pack(review_config(tmp_path), output_directory=tmp_path / "out")

    assert widefield.read_bytes() == before_widefield
    assert source_spacing.read_bytes() == before_spacing
