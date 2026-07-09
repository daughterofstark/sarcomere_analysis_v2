from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sarcomere_analysis.zdisc_annotation import (
    audit_zdisc_annotations,
    prepare_zdisc_annotation_set,
    select_zdisc_annotation_crops,
)


def zdisc_config(tmp_path: Path) -> dict:
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


def make_annotation_pack(tmp_path: Path, n: int = 12) -> tuple[dict, Path]:
    cfg = zdisc_config(tmp_path)
    pack = tmp_path / "results" / "annotation_pack"
    crops = pack / "crops"
    crops.mkdir(parents=True)
    rows = []
    oop_values = [0.1, 0.2, 0.45, 0.55, 0.75, 0.9]
    for idx in range(n):
        donor = f"{2 + (idx % 4)}.{7 + idx:03d}"
        image_id = f"{donor}-{(idx % 3) + 1}"
        annotation_id = f"ANN_{idx + 1:04d}"
        patch_id = f"{image_id}_p{idx:05d}"
        crop_path = crops / f"{annotation_id}__{image_id}__{patch_id}.png"
        Image.fromarray(np.full((8, 10), idx + 1, dtype=np.uint8)).save(crop_path)
        valid = idx >= 2
        rows.append(
            {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "donor_id": donor,
                "patch_id": patch_id,
                "valid_for_orientation": valid,
                "patch_oop": np.nan if not valid else oop_values[idx % len(oop_values)],
                "patch_mean_orientation_deg": np.nan if not valid else float(idx),
                "suggested_annotation_task": "manual_orientation_oop_review" if valid else "negative_control_quality_review",
                "oop_bin": "invalid_control" if not valid else ["low_oop", "medium_oop", "high_oop"][idx % 3],
                "crop_path": str(crop_path),
            }
        )
    index_path = pack / "annotation_patch_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return cfg, index_path


def test_annotation_set_selection_is_deterministic(tmp_path: Path) -> None:
    _, index_path = make_annotation_pack(tmp_path)
    source = pd.read_csv(index_path, dtype={"annotation_id": str, "image_id": str, "donor_id": str, "patch_id": str})

    first = select_zdisc_annotation_crops(source, n_crops=8, seed=5)
    second = select_zdisc_annotation_crops(source, n_crops=8, seed=5)

    assert first["annotation_id"].tolist() == second["annotation_id"].tolist()


def test_image_and_mask_files_are_created(tmp_path: Path) -> None:
    cfg, index_path = make_annotation_pack(tmp_path)

    index, summary, _ = prepare_zdisc_annotation_set(cfg, n_crops=6, seed=1, annotation_index_path=index_path, overwrite=True)

    assert summary["selected_crops"] == 6
    assert summary["invalid_or_low_quality_controls"] == int((index["valid_for_orientation"].astype(str) == "False").sum())
    assert all(Path(path).exists() for path in index["annotation_image_path"])
    assert all(Path(path).exists() for path in index["mask_path"])


def test_blank_masks_match_crop_shape(tmp_path: Path) -> None:
    cfg, index_path = make_annotation_pack(tmp_path)
    index, _, _ = prepare_zdisc_annotation_set(cfg, n_crops=4, seed=1, annotation_index_path=index_path, overwrite=True)

    for _, row in index.iterrows():
        image = np.asarray(Image.open(row["annotation_image_path"]))
        mask = np.asarray(Image.open(row["mask_path"]))
        assert mask.shape == image.shape[:2]
        assert mask.dtype == np.uint8


def test_mask_labels_are_restricted_to_0_1_2(tmp_path: Path) -> None:
    cfg, index_path = make_annotation_pack(tmp_path)
    index, _, _ = prepare_zdisc_annotation_set(cfg, n_crops=3, seed=1, annotation_index_path=index_path, overwrite=True)
    mask_path = Path(index.loc[0, "mask_path"])
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[4:5, 4:5] = 2
    Image.fromarray(mask).save(mask_path)

    audit, summary, _ = audit_zdisc_annotations(cfg)

    assert summary["invalid_label_masks"] == 0
    assert audit["allowed_labels_only"].all()
    assert summary["masks_with_zdisc_labels"] == 1


def test_audit_detects_missing_masks(tmp_path: Path) -> None:
    cfg, index_path = make_annotation_pack(tmp_path)
    index, _, _ = prepare_zdisc_annotation_set(cfg, n_crops=3, seed=1, annotation_index_path=index_path, overwrite=True)
    Path(index.loc[0, "mask_path"]).unlink()

    _, summary, _ = audit_zdisc_annotations(cfg)

    assert summary["missing_masks"] == 1


def test_audit_detects_shape_mismatch(tmp_path: Path) -> None:
    cfg, index_path = make_annotation_pack(tmp_path)
    index, _, _ = prepare_zdisc_annotation_set(cfg, n_crops=3, seed=1, annotation_index_path=index_path, overwrite=True)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(index.loc[0, "mask_path"])

    _, summary, _ = audit_zdisc_annotations(cfg)

    assert summary["shape_mismatch_masks"] == 1


def test_audit_counts_empty_and_non_empty_masks(tmp_path: Path) -> None:
    cfg, index_path = make_annotation_pack(tmp_path)
    index, _, _ = prepare_zdisc_annotation_set(cfg, n_crops=4, seed=1, annotation_index_path=index_path, overwrite=True)
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    Image.fromarray(mask).save(index.loc[1, "mask_path"])

    _, summary, _ = audit_zdisc_annotations(cfg)

    assert summary["selected_crops"] == 4
    assert summary["empty_masks"] == 3
    assert summary["masks_with_zdisc_labels"] == 1


def test_summary_json_is_serializable(tmp_path: Path) -> None:
    cfg, index_path = make_annotation_pack(tmp_path)
    _, _, paths = prepare_zdisc_annotation_set(cfg, n_crops=3, seed=1, annotation_index_path=index_path, overwrite=True)

    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    assert loaded["selected_crops"] == 3


def test_production_feature_and_analysis_tables_are_not_modified(tmp_path: Path) -> None:
    cfg, index_path = make_annotation_pack(tmp_path)
    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    feature_path = tables / "features_per_image.csv"
    analysis_path = tables / "analysis_per_image.csv"
    feature_path.write_text("image_id,donor_id,image_oop\n2.007-1,2.007,0.5\n", encoding="utf-8")
    analysis_path.write_text("image_id,donor_id,is_healthy\n2.007-1,2.007,False\n", encoding="utf-8")
    before_feature = feature_path.read_bytes()
    before_analysis = analysis_path.read_bytes()

    prepare_zdisc_annotation_set(cfg, n_crops=3, seed=1, annotation_index_path=index_path, overwrite=True)
    audit_zdisc_annotations(cfg)

    assert feature_path.read_bytes() == before_feature
    assert analysis_path.read_bytes() == before_analysis
