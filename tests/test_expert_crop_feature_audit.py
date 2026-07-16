from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sarcomere_analysis.expert_crop_feature_audit import (
    CROP_FEATURE_COLUMNS,
    audit_expert_crop_features,
    build_expert_crop_feature_table,
    compute_crop_features,
    organisation_summary,
    visibility_summary,
)


def crop_config(tmp_path: Path) -> dict:
    return {
        "paths": {"raw_tiff_dir": str(tmp_path / "raw"), "output_dir": str(tmp_path / "results")},
        "outputs": {"manifest_csv": str(tmp_path / "results" / "tables" / "manifest.csv")},
        "calibration": {
            "pixel_size_um": 0.1299,
            "expected_sarcomere_spacing_um": {"min": 1.5, "max": 2.4},
        },
        "filename_pattern": {"regex": r"^(?P<donor_id>\d+\.\d+)-(?P<region_id>\d+)$"},
        "run": {"include_extensions": [".tif", ".tiff"], "recursive": False},
        "orientation": {"min_orientation_valid_pixels": 8, "min_orientation_weight_sum": 0.0, "tensor_sigma_px": 1.0},
    }


def write_png(path: Path, period: float = 8.0, angle: float = 0.0) -> None:
    y, x = np.mgrid[0:64, 0:64]
    theta = np.deg2rad(angle)
    coord = x * np.cos(theta) + y * np.sin(theta)
    image = 0.5 + 0.4 * np.sin(2 * np.pi * coord / period)
    array = np.clip(image * 255, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def synthetic_tables(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    crop_dir = tmp_path / "crops"
    for idx, angle in enumerate([0, 30, 60, 90], start=1):
        write_png(crop_dir / f"EXPERT_{idx:04d}.png", angle=angle)
    matched = pd.DataFrame(
        {
            "annotation_id": [f"EXPERT_{idx:04d}" for idx in range(1, 5)],
            "patch_filename": [f"EXPERT_{idx:04d}.png" for idx in range(1, 5)],
            "image_id": ["2.007-1", "2.007-1", "3.001-1", "3.001-1"],
            "donor_id": ["2.007", "2.007", "3.001", "3.001"],
            "patch_id": ["p1", "p2", "p3", "p4"],
            "oop_bin": ["low", "medium", "high", "high"],
            "automated_patch_oop": [0.01, 0.05, 0.2, 0.3],
            "automated_patch_orientation_deg": [0.0, 30.0, 60.0, 90.0],
            "striations_visible": ["no", "unclear", "yes", "yes"],
            "organisation_score": [1, 3, 4, 5],
            "confidence_score": [2, 3, 4, 5],
            "spacing_measurable": ["no", "unclear", "yes", "yes"],
            "expert_orientation_usable_primary": [False, False, False, False],
        }
    )
    key = matched[["annotation_id", "patch_filename", "image_id", "donor_id", "patch_id"]].copy()
    return matched, key, crop_dir


def test_loads_expert_visible_pngs_by_annotation_id_and_filename(tmp_path: Path) -> None:
    matched, key, crop_dir = synthetic_tables(tmp_path)
    table = build_expert_crop_feature_table(matched, key, crop_dir, crop_config(tmp_path))

    assert len(table) == 4
    assert table["crop_found"].all()
    assert table["annotation_id"].tolist()[0] == "EXPERT_0001"


def test_computes_finite_crop_features(tmp_path: Path) -> None:
    matched, key, crop_dir = synthetic_tables(tmp_path)
    table = build_expert_crop_feature_table(matched, key, crop_dir, crop_config(tmp_path))

    assert np.isfinite(table["crop_oop"]).all()
    assert np.isfinite(table["crop_gradient_energy"]).all()
    assert table["crop_oop"].between(0, 1).all()


def test_handles_missing_png_gracefully(tmp_path: Path) -> None:
    matched, key, crop_dir = synthetic_tables(tmp_path)
    (crop_dir / "EXPERT_0004.png").unlink()
    table = build_expert_crop_feature_table(matched, key, crop_dir, crop_config(tmp_path))

    missing = table.loc[table["annotation_id"] == "EXPERT_0004"].iloc[0]
    assert not bool(missing["crop_found"])
    assert pd.isna(missing["crop_oop"])


def test_visibility_medians_computed(tmp_path: Path) -> None:
    matched, key, crop_dir = synthetic_tables(tmp_path)
    table = build_expert_crop_feature_table(matched, key, crop_dir, crop_config(tmp_path))
    visibility = visibility_summary(table, ["crop_oop"])

    assert visibility.loc[0, "feature"] == "crop_oop"
    assert pd.notna(visibility.loc[0, "median_yes"])
    assert pd.notna(visibility.loc[0, "median_no"])


def test_organisation_spearman_computed_or_skipped_safely(tmp_path: Path) -> None:
    matched, key, crop_dir = synthetic_tables(tmp_path)
    table = build_expert_crop_feature_table(matched, key, crop_dir, crop_config(tmp_path))
    organisation = organisation_summary(table, ["crop_oop"], min_n=3, min_confidence=3)

    assert organisation.loc[0, "feature"] == "crop_oop"
    assert organisation.loc[0, "n"] >= 3
    assert "spearman_rho" in organisation.columns


def test_collapsed_organisation_groups_computed(tmp_path: Path) -> None:
    matched, key, crop_dir = synthetic_tables(tmp_path)
    table = build_expert_crop_feature_table(matched, key, crop_dir, crop_config(tmp_path))
    organisation = organisation_summary(table, ["crop_oop"], min_n=3, min_confidence=3)

    assert "median_organisation_low" in organisation.columns
    assert "median_organisation_medium" in organisation.columns
    assert "median_organisation_high" in organisation.columns
    assert pd.notna(organisation.loc[0, "median_organisation_high"])


def test_summary_json_serializable(tmp_path: Path) -> None:
    matched, key, crop_dir = synthetic_tables(tmp_path)
    matched_path = tmp_path / "matched.csv"
    key_path = tmp_path / "key.csv"
    matched.to_csv(matched_path, index=False)
    key.to_csv(key_path, index=False)

    _, _, _, _, summary, _ = audit_expert_crop_features(
        crop_config(tmp_path),
        crop_dir=crop_dir,
        internal_key=key_path,
        matched_annotations=matched_path,
        output_directory=tmp_path / "out",
        min_n=3,
    )

    json.dumps(summary)


def test_production_tables_not_modified(tmp_path: Path) -> None:
    matched, key, crop_dir = synthetic_tables(tmp_path)
    matched_path = tmp_path / "matched.csv"
    key_path = tmp_path / "key.csv"
    production = tmp_path / "results" / "tables" / "features_per_patch.csv"
    production.parent.mkdir(parents=True, exist_ok=True)
    production.write_text("image_id,patch_id,patch_oop\n2.007-1,p1,0.1\n", encoding="utf-8")
    before = production.read_bytes()
    matched.to_csv(matched_path, index=False)
    key.to_csv(key_path, index=False)

    audit_expert_crop_features(
        crop_config(tmp_path),
        crop_dir=crop_dir,
        internal_key=key_path,
        matched_annotations=matched_path,
        output_directory=tmp_path / "out",
        min_n=3,
    )

    assert production.read_bytes() == before


def test_compute_crop_features_returns_required_columns(tmp_path: Path) -> None:
    image = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))
    features = compute_crop_features(image, crop_config(tmp_path))

    for column in CROP_FEATURE_COLUMNS:
        assert column in features
