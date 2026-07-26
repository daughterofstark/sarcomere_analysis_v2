from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from PIL import Image

from sarcomere_analysis.confocal_gate_review_pack import (
    DEFAULT_BASELINE_VARIANT,
    DEFAULT_FOCUS_IMAGES,
    DEFAULT_RELAXED_VARIANT,
    GATE_REVIEW_SUMMARY_COLUMNS,
    export_confocal_gate_review_pack,
)


def gate_review_config(tmp_path: Path) -> dict:
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


def write_gate_review_inputs(tmp_path: Path, include_previews: bool = True) -> None:
    root = tmp_path / "results"
    refinement = root / "confocal_gate_refinement"
    previous_review = root / "confocal_review_pack"
    refinement.mkdir(parents=True, exist_ok=True)
    previous_review.mkdir(parents=True, exist_ok=True)

    variants = pd.DataFrame(
        [
            {
                "variant_name": DEFAULT_BASELINE_VARIANT,
                "total_candidate_patches": 8,
                "overall_candidate_fraction": 0.2,
                "added_vs_moderate_count": 0,
                "selected_median_oop": 0.7,
                "selected_median_coherence": 0.6,
                "selected_valid_spacing_count": 4,
                "selected_valid_spacing_fraction": 0.5,
                "selected_median_spacing_um": 1.8,
                "classification": "conservative_reference",
            },
            {
                "variant_name": DEFAULT_RELAXED_VARIANT,
                "total_candidate_patches": 16,
                "overall_candidate_fraction": 0.4,
                "added_vs_moderate_count": 8,
                "selected_median_oop": 0.75,
                "selected_median_coherence": 0.65,
                "selected_valid_spacing_count": 4,
                "selected_valid_spacing_fraction": 0.25,
                "selected_median_spacing_um": 1.8,
                "classification": "plausible_for_review",
            },
        ]
    )
    variants.to_csv(refinement / "confocal_gate_refinement_variants.csv", index=False)

    per_image_rows = []
    previous_rows = []
    for image_id in DEFAULT_FOCUS_IMAGES:
        for variant, count, fraction, added, flag in [
            (DEFAULT_BASELINE_VARIANT, 2, 0.2, 0, "keep_conservative_reference"),
            (
                DEFAULT_RELAXED_VARIANT,
                4,
                0.4 if image_id != "7028" else 0.8,
                2,
                "candidate_recovered_more_regions"
                if image_id != "7028"
                else "candidate_recovered_more_regions;broad_selection_risk",
            ),
        ]:
            per_image_rows.append(
                {
                    "confocal_image_id": image_id,
                    "filename": f"{image_id}.tif",
                    "variant_name": variant,
                    "candidate_patch_count": count,
                    "candidate_fraction": fraction,
                    "added_vs_moderate_count": added,
                    "removed_vs_moderate_count": 0,
                    "selected_median_oop": 0.7,
                    "selected_median_coherence": 0.6,
                    "selected_valid_spacing_count": 1,
                    "selected_spacing_valid_fraction": 0.25,
                    "selected_spacing_median_um": 1.8,
                    "expected_positive_example": image_id in {"5138", "6052-CLEAR_STRIPES"},
                    "noted_complex_example": image_id == "3112",
                    "review_flag": flag,
                }
            )
        previous_rows.append(
            {
                "filename": f"{image_id}.tif",
                "valid_selected_spacing_count": 1,
                "selected_spacing_median_um": 1.8,
            }
        )
    pd.DataFrame(per_image_rows).to_csv(refinement / "confocal_gate_refinement_per_image.csv", index=False)
    pd.DataFrame(previous_rows).to_csv(previous_review / "confocal_review_summary.csv", index=False)

    if include_previews:
        for image_id in DEFAULT_FOCUS_IMAGES:
            write_png(refinement / "previews" / f"{image_id}_{DEFAULT_BASELINE_VARIANT}_gate_refinement_overlay.png")
            write_png(refinement / "previews" / f"{image_id}_{DEFAULT_RELAXED_VARIANT}_gate_refinement_overlay.png")
            write_png(previous_review / "review_images" / f"{image_id}_valid_spacing_patch_overlay.png")


def test_default_focus_images_selected(tmp_path: Path) -> None:
    write_gate_review_inputs(tmp_path)
    _, summary, _ = export_confocal_gate_review_pack(gate_review_config(tmp_path), output_directory=tmp_path / "out")

    assert summary["images_included"] == DEFAULT_FOCUS_IMAGES


def test_summary_csv_includes_required_comparison_columns(tmp_path: Path) -> None:
    write_gate_review_inputs(tmp_path)
    table, _, paths = export_confocal_gate_review_pack(gate_review_config(tmp_path), output_directory=tmp_path / "out")

    assert list(table.columns) == GATE_REVIEW_SUMMARY_COLUMNS
    written = pd.read_csv(paths["summary_csv"])
    assert list(written.columns) == GATE_REVIEW_SUMMARY_COLUMNS
    assert set(written["spacing_caveat"]) == {"spacing_from_moderate_gate_not_refreshed_for_relaxed_patches"}


def test_handles_missing_preview_images_gracefully(tmp_path: Path) -> None:
    write_gate_review_inputs(tmp_path, include_previews=False)
    _, summary, _ = export_confocal_gate_review_pack(gate_review_config(tmp_path), output_directory=tmp_path / "out")

    assert summary["review_image_files_copied"] == 0
    assert summary["missing_preview_count"] > 0


def test_notes_file_includes_spacing_not_refreshed_caveat(tmp_path: Path) -> None:
    write_gate_review_inputs(tmp_path)
    _, _, paths = export_confocal_gate_review_pack(gate_review_config(tmp_path), output_directory=tmp_path / "out")

    text = paths["notes_md"].read_text(encoding="utf-8")
    assert "Spacing has not yet been recomputed" in text
    assert "5138" in text
    assert "7028" in text


def test_zip_includes_review_files(tmp_path: Path) -> None:
    write_gate_review_inputs(tmp_path)
    _, _, paths = export_confocal_gate_review_pack(
        gate_review_config(tmp_path), output_directory=tmp_path / "out", write_zip=True
    )

    with ZipFile(paths["zip"]) as archive:
        names = set(archive.namelist())
    assert "confocal_gate_review_summary.csv" in names
    assert "confocal_gate_review_notes_for_natalia.md" in names
    assert "confocal_gate_review_pack_summary.txt" in names
    assert any(name.startswith("review_images/") and name.endswith(".png") for name in names)


def test_zip_excludes_raw_internal_large_tables(tmp_path: Path) -> None:
    write_gate_review_inputs(tmp_path)
    _, _, paths = export_confocal_gate_review_pack(
        gate_review_config(tmp_path), output_directory=tmp_path / "out", write_zip=True
    )

    with ZipFile(paths["zip"]) as archive:
        names = archive.namelist()
    assert not any("confocal_gate_refinement_per_image" in name for name in names)
    assert not any("confocal_gate_refinement_variants" in name for name in names)
    assert not any("internal" in name.lower() for name in names)


def test_summary_json_serializable(tmp_path: Path) -> None:
    write_gate_review_inputs(tmp_path)
    _, summary, paths = export_confocal_gate_review_pack(gate_review_config(tmp_path), output_directory=tmp_path / "out")

    json.dumps(summary)
    assert paths["summary_json"].exists()


def test_existing_confocal_and_widefield_outputs_are_not_modified(tmp_path: Path) -> None:
    write_gate_review_inputs(tmp_path)
    widefield = tmp_path / "results" / "tables" / "features_per_image.csv"
    source = tmp_path / "results" / "confocal_gate_refinement" / "confocal_gate_refinement_per_image.csv"
    widefield.parent.mkdir(parents=True, exist_ok=True)
    widefield.write_text("image_id,image_oop\n2.007-1,0.1\n", encoding="utf-8")
    before_widefield = widefield.read_bytes()
    before_source = source.read_bytes()

    export_confocal_gate_review_pack(gate_review_config(tmp_path), output_directory=tmp_path / "out")

    assert widefield.read_bytes() == before_widefield
    assert source.read_bytes() == before_source
