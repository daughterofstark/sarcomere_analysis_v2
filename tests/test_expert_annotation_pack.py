from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
import tifffile

from sarcomere_analysis.expert_annotation_pack import (
    EXPERT_TEMPLATE_COLUMNS,
    expert_crop_bounds,
    export_expert_annotation_pack,
    prepare_expert_candidates,
    select_expert_patches,
    target_counts,
)


def pack_config(tmp_path: Path) -> dict:
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


def synthetic_tables(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    patch_rows = []
    analysis_rows = []
    oop_values = np.linspace(0.01, 0.99, 18)
    idx = 0
    for donor_idx in range(6):
        donor_id = f"{donor_idx + 1}.001"
        image_id = f"{donor_id}-1"
        image_path = raw_dir / f"{image_id}.tif"
        image = np.arange(64 * 96, dtype=np.uint16).reshape(64, 96) + donor_idx
        tifffile.imwrite(image_path, image)
        analysis_rows.append(
            {
                "image_id": image_id,
                "donor_id": donor_id,
                "image_path": str(image_path),
                "is_healthy": donor_idx % 2 == 0,
            }
        )
        for patch_idx, x0 in enumerate([0, 32, 64]):
            patch_rows.append(
                {
                    "image_id": image_id,
                    "donor_id": donor_id,
                    "patch_id": f"{image_id}_p{patch_idx:05d}",
                    "y0": 0,
                    "x0": x0,
                    "y1": 32,
                    "x1": x0 + 32,
                    "patch_oop": float(oop_values[idx]),
                    "patch_mean_orientation_deg": float((idx * 10) % 180),
                    "valid_for_orientation": True,
                }
            )
            idx += 1
    patches = pd.DataFrame(patch_rows)
    analysis = pd.DataFrame(analysis_rows)
    manifest = analysis.copy()
    return patches, analysis, manifest


def write_input_tables(tmp_path: Path) -> tuple[Path, Path, Path]:
    patches, analysis, manifest = synthetic_tables(tmp_path)
    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    patch_path = tables / "features_per_patch.csv"
    analysis_path = tables / "analysis_per_image.csv"
    manifest_path = tables / "enriched_manifest.csv"
    image_path = tables / "features_per_image.csv"
    patches.to_csv(patch_path, index=False)
    analysis.to_csv(analysis_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    analysis[["image_id", "donor_id"]].to_csv(image_path, index=False)
    return patch_path, analysis_path, manifest_path


def test_selected_pack_has_expected_total_count_when_enough_patches_exist(tmp_path: Path) -> None:
    patches, analysis, manifest = synthetic_tables(tmp_path)
    candidates = prepare_expert_candidates(patches, analysis, manifest)
    selected, _ = select_expert_patches(candidates, target_counts(n_total=12), seed=1, max_per_donor=2, max_per_image=2)

    assert len(selected) == 12


def test_bins_are_approximately_balanced(tmp_path: Path) -> None:
    patches, analysis, manifest = synthetic_tables(tmp_path)
    candidates = prepare_expert_candidates(patches, analysis, manifest)
    selected, _ = select_expert_patches(candidates, target_counts(n_total=12), seed=1, max_per_donor=2, max_per_image=2)

    assert selected["oop_bin"].value_counts().to_dict() == {"low": 4, "medium": 4, "high": 4}


def test_max_per_donor_constraint_is_respected(tmp_path: Path) -> None:
    patches, analysis, manifest = synthetic_tables(tmp_path)
    candidates = prepare_expert_candidates(patches, analysis, manifest)
    selected, _ = select_expert_patches(candidates, target_counts(n_total=12), seed=2, max_per_donor=2, max_per_image=3)

    assert int(selected["donor_id"].value_counts().max()) <= 2


def test_max_per_image_constraint_is_respected(tmp_path: Path) -> None:
    patches, analysis, manifest = synthetic_tables(tmp_path)
    candidates = prepare_expert_candidates(patches, analysis, manifest)
    selected, _ = select_expert_patches(candidates, target_counts(n_total=12), seed=3, max_per_donor=3, max_per_image=2)

    assert int(selected["image_id"].value_counts().max()) <= 2


def test_expert_template_excludes_oop_donor_image_health_labels(tmp_path: Path) -> None:
    write_input_tables(tmp_path)
    _, template, _, _, _ = export_expert_annotation_pack(pack_config(tmp_path), n_total=12, seed=4, max_per_donor=2, max_per_image=2)

    assert list(template.columns) == EXPERT_TEMPLATE_COLUMNS
    forbidden = {"donor_id", "image_id", "patch_id", "patch_oop", "oop_bin", "health_status"}
    assert forbidden.isdisjoint(template.columns)


def test_internal_blinding_key_contains_mapping_metadata(tmp_path: Path) -> None:
    write_input_tables(tmp_path)
    _, _, internal_key, _, _ = export_expert_annotation_pack(pack_config(tmp_path), n_total=12, seed=4, max_per_donor=2, max_per_image=2)

    for column in [
        "image_id",
        "donor_id",
        "patch_id",
        "oop_bin",
        "automated_patch_oop",
        "health_status",
        "production_patch_size_px",
        "requested_expert_crop_size_px",
        "expert_crop_size_px",
        "production_patch_x",
        "production_patch_y",
        "expert_crop_x0",
        "expert_crop_y0",
        "expert_crop_x1",
        "expert_crop_y1",
    ]:
        assert column in internal_key.columns


def test_expert_crop_size_can_be_larger_than_production_patch(tmp_path: Path) -> None:
    write_input_tables(tmp_path)
    _, template, internal_key, _, paths = export_expert_annotation_pack(
        pack_config(tmp_path),
        n_total=12,
        seed=4,
        max_per_donor=2,
        max_per_image=2,
        expert_crop_size=64,
    )
    first_png = paths["patch_dir"] / str(template.loc[0, "patch_filename"])
    image = Image.open(first_png)

    assert int(internal_key["production_patch_size_px"].max()) == 32
    assert int(internal_key["expert_crop_size_px"].min()) == 64
    assert max(image.size) >= 64


def test_crop_clips_safely_at_image_boundaries() -> None:
    row = pd.Series({"production_patch_x": 2, "production_patch_y": 2, "expert_crop_size_px": 64, "x0": 0, "x1": 32, "y0": 0, "y1": 32})

    y0, x0, y1, x1 = expert_crop_bounds(row, (40, 50))

    assert (y0, x0) == (0, 0)
    assert y1 <= 40
    assert x1 <= 50
    assert y1 > y0
    assert x1 > x0


def test_existing_patch_id_internal_mapping_is_preserved(tmp_path: Path) -> None:
    write_input_tables(tmp_path)
    _, _, first_key, _, _ = export_expert_annotation_pack(pack_config(tmp_path), n_total=12, seed=4, max_per_donor=2, max_per_image=2)

    _, _, second_key, _, _ = export_expert_annotation_pack(pack_config(tmp_path), n_total=12, seed=99, max_per_donor=2, max_per_image=2)

    pd.testing.assert_series_equal(first_key["annotation_id"], second_key["annotation_id"])
    pd.testing.assert_series_equal(first_key["patch_id"], second_key["patch_id"])


def test_exported_patch_filenames_are_anonymous(tmp_path: Path) -> None:
    write_input_tables(tmp_path)
    _, template, _, _, paths = export_expert_annotation_pack(pack_config(tmp_path), n_total=12, seed=4, max_per_donor=2, max_per_image=2)

    filenames = template["patch_filename"].tolist()
    assert all(name.startswith("EXPERT_") and name.endswith(".png") for name in filenames)
    assert all("-" not in name for name in filenames)
    assert len(list(paths["patch_dir"].glob("*.png"))) == 12


def test_instructions_file_contains_scoring_rubric(tmp_path: Path) -> None:
    write_input_tables(tmp_path)
    _, _, _, _, paths = export_expert_annotation_pack(pack_config(tmp_path), n_total=12, seed=4, max_per_donor=2, max_per_image=2)

    text = paths["instructions_md"].read_text(encoding="utf-8")
    assert "organisation_score" in text
    assert "1 = disorganised" in text
    assert "5 = highly organised" in text


def test_zip_excludes_internal_blinding_key(tmp_path: Path) -> None:
    write_input_tables(tmp_path)
    _, _, _, _, paths = export_expert_annotation_pack(
        pack_config(tmp_path),
        n_total=12,
        seed=4,
        max_per_donor=2,
        max_per_image=2,
        write_zip=True,
    )

    with ZipFile(paths["zip"]) as archive:
        names = set(archive.namelist())
    assert "internal_blinding_key.csv" not in names
    assert "expert_annotation_template.csv" in names
    assert "annotation_instructions.md" in names
    assert "expert_annotation_contact_sheet.png" in names
    assert any(name.startswith("patches/EXPERT_") for name in names)


def test_contact_sheet_is_written(tmp_path: Path) -> None:
    write_input_tables(tmp_path)
    _, _, _, summary, paths = export_expert_annotation_pack(pack_config(tmp_path), n_total=12, seed=4, max_per_donor=2, max_per_image=2)

    assert paths["contact_sheet_png"].exists()
    assert summary["contact_sheet_written"] is True


def test_summary_json_is_serializable(tmp_path: Path) -> None:
    write_input_tables(tmp_path)
    _, _, _, summary, paths = export_expert_annotation_pack(pack_config(tmp_path), n_total=12, seed=4, max_per_donor=2, max_per_image=2)

    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded["selected_patches"] == summary["selected_patches"]


def test_existing_production_tables_are_not_modified(tmp_path: Path) -> None:
    patch_path, analysis_path, manifest_path = write_input_tables(tmp_path)
    before = {path: path.read_bytes() for path in [patch_path, analysis_path, manifest_path]}

    export_expert_annotation_pack(pack_config(tmp_path), n_total=12, seed=4, max_per_donor=2, max_per_image=2)

    for path, data in before.items():
        assert path.read_bytes() == data
