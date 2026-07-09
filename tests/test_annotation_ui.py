from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sarcomere_analysis.annotation_ui import (
    ANNOTATION_UI_COLUMNS,
    apply_key_to_annotation,
    autosave_annotations,
    default_annotation_paths,
    headless_check,
    initialize_annotation_table,
    load_or_initialize_annotations,
    read_annotation_index,
    save_annotations,
    update_annotation_row,
)


def ui_config(tmp_path: Path) -> dict:
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


def make_fake_pack(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    cfg = ui_config(tmp_path)
    pack_dir = tmp_path / "results" / "annotation_pack"
    crops_dir = pack_dir / "crops"
    crops_dir.mkdir(parents=True)
    rows = []
    template_rows = []
    for idx, donor in enumerate(["2.007", "4.083"], start=1):
        annotation_id = f"ANN_{idx:04d}"
        image_id = f"{donor}-{idx}"
        patch_id = f"{image_id}_p{idx:05d}"
        crop_path = crops_dir / f"{annotation_id}__{image_id}__{patch_id}.png"
        Image.fromarray(np.full((8, 8), idx * 50, dtype=np.uint8)).save(crop_path)
        rows.append(
            {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "donor_id": donor,
                "patch_id": patch_id,
                "patch_oop": 0.25 * idx,
                "patch_mean_orientation_deg": 30.0 * idx,
                "crop_path": str(crop_path),
            }
        )
        template_rows.append(
            {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "donor_id": donor,
                "patch_id": patch_id,
            }
        )
    index_path = pack_dir / "annotation_patch_index.csv"
    template_path = pack_dir / "annotation_template.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    pd.DataFrame(template_rows, columns=ANNOTATION_UI_COLUMNS).to_csv(template_path, index=False)
    return cfg, index_path, template_path, pack_dir


def test_initializes_output_csv_with_required_columns(tmp_path: Path) -> None:
    _, index_path, template_path, _ = make_fake_pack(tmp_path)
    index = read_annotation_index(index_path)
    template = pd.read_csv(template_path, dtype=str)
    annotations = initialize_annotation_table(index, template)

    assert list(annotations.columns) == ANNOTATION_UI_COLUMNS
    assert len(annotations) == 2


def test_resume_preserves_existing_annotations(tmp_path: Path) -> None:
    _, index_path, template_path, pack_dir = make_fake_pack(tmp_path)
    index = read_annotation_index(index_path)
    output_csv = pack_dir / "annotation_filled.csv"
    existing = initialize_annotation_table(index)
    existing = update_annotation_row(
        existing,
        "ANN_0001",
        {"manual_organisation_score": 4, "visible_striations_yes_no": "yes", "notes": "already done"},
    )
    save_annotations(existing, output_csv)

    resumed = load_or_initialize_annotations(index, template_path, output_csv, overwrite=False)

    row = resumed.loc[resumed["annotation_id"] == "ANN_0001"].iloc[0]
    assert float(row["manual_organisation_score"]) == 4.0
    assert row["visible_striations_yes_no"] == "yes"
    assert row["notes"] == "already done"


def test_autosave_writes_after_update(tmp_path: Path) -> None:
    _, index_path, _, pack_dir = make_fake_pack(tmp_path)
    index = read_annotation_index(index_path)
    annotations = initialize_annotation_table(index)
    annotations = update_annotation_row(annotations, "ANN_0002", {"confidence_score": 5})
    output_csv = pack_dir / "annotation_filled.csv"

    autosave_path = autosave_annotations(annotations, output_csv)

    assert autosave_path.exists()
    loaded = pd.read_csv(autosave_path, dtype={"annotation_id": str, "image_id": str, "donor_id": str})
    assert float(loaded.loc[loaded["annotation_id"] == "ANN_0002", "confidence_score"].iloc[0]) == 5.0


def test_donor_id_and_image_id_remain_strings(tmp_path: Path) -> None:
    _, index_path, _, _ = make_fake_pack(tmp_path)
    index = read_annotation_index(index_path)
    annotations = initialize_annotation_table(index)

    assert annotations["donor_id"].map(type).eq(str).all()
    assert annotations["image_id"].map(type).eq(str).all()
    assert annotations.loc[0, "donor_id"] == "2.007"


def test_keyboard_mapping_sets_score_and_visibility_labels() -> None:
    row = {column: "" for column in ANNOTATION_UI_COLUMNS}

    row = apply_key_to_annotation(row, "5")
    assert row["manual_organisation_score"] == 5
    assert row["manual_organisation_label"] == "highly organised"

    row = apply_key_to_annotation(row, "u")
    assert row["visible_striations_yes_no"] == "yes_unclear"

    row = apply_key_to_annotation(row, "r")
    assert pd.isna(row["manual_dominant_orientation_deg"])


def test_headless_check_validates_paths_without_opening_ui(tmp_path: Path) -> None:
    cfg, _, _, _ = make_fake_pack(tmp_path)

    summary = headless_check(cfg)

    assert summary["index_rows"] == 2
    assert summary["annotation_rows"] == 2
    assert summary["crop_count"] == 2
    assert summary["missing_crop_count"] == 0


def test_does_not_modify_annotation_patch_index(tmp_path: Path) -> None:
    cfg, index_path, _, _ = make_fake_pack(tmp_path)
    before = index_path.read_bytes()

    headless_check(cfg)

    assert index_path.read_bytes() == before


def test_default_paths_point_to_annotation_pack(tmp_path: Path) -> None:
    cfg, _, _, pack_dir = make_fake_pack(tmp_path)
    paths = default_annotation_paths(cfg)

    assert paths.pack_dir == pack_dir
    assert paths.output_csv.name == "annotation_filled.csv"
    assert paths.autosave_csv.name == "annotation_filled.autosave.csv"
