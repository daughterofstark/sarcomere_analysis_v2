from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sarcomere_analysis.validation_full_image_zdisc_masks import (
    build_full_image_zdisc_mask_validation,
    validate_full_image_zdisc_masks,
)
from sarcomere_analysis.validation_zdisc_masks import axial_angular_error_deg


def validation_config(tmp_path: Path) -> dict:
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


def synthetic_annotations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "annotation_id": "FULL_0001",
                "image_id": "2.007-1",
                "donor_id": "2.007",
                "annotation_status": "zdisc_labeled",
                "zdisc_pixel_fraction": 0.10,
                "has_zdisc_labels": True,
                "manual_mask_orientation_deg": 175.0,
                "orientation_estimable": True,
            },
            {
                "annotation_id": "FULL_0002",
                "image_id": "2.007-2",
                "donor_id": "2.007",
                "annotation_status": "empty",
                "zdisc_pixel_fraction": 0.0,
                "has_zdisc_labels": False,
                "manual_mask_orientation_deg": np.nan,
                "orientation_estimable": False,
            },
            {
                "annotation_id": "FULL_0003",
                "image_id": "3.032-1",
                "donor_id": "3.032",
                "annotation_status": "zdisc_labeled",
                "zdisc_pixel_fraction": 0.2,
                "has_zdisc_labels": True,
                "manual_mask_orientation_deg": 45.0,
                "orientation_estimable": True,
            },
        ]
    )


def synthetic_images(include_missing: bool = False, donor_mismatch: bool = False) -> pd.DataFrame:
    rows = [
        {
            "image_id": "2.007-1",
            "donor_id": "2.007",
            "image_oop": 0.8,
            "image_mean_orientation_deg": 5.0,
            "image_oop_heterogeneity": 0.1,
            "n_orientation_valid_patches": 100,
            "orientation_valid_fraction": 0.9,
            "status": "ok",
            "spacing_endpoint_status": "insufficient_patch_yield",
        },
        {
            "image_id": "2.007-2",
            "donor_id": "2.007",
            "image_oop": 0.2,
            "image_mean_orientation_deg": np.nan,
            "image_oop_heterogeneity": 0.2,
            "n_orientation_valid_patches": 20,
            "orientation_valid_fraction": 0.4,
            "status": "ok",
            "spacing_endpoint_status": "insufficient_patch_yield",
        },
    ]
    if not include_missing:
        rows.append(
            {
                "image_id": "3.032-1",
                "donor_id": "9.999" if donor_mismatch else "3.032",
                "image_oop": 0.4,
                "image_mean_orientation_deg": 60.0,
                "image_oop_heterogeneity": 0.3,
                "n_orientation_valid_patches": 50,
                "orientation_valid_fraction": 0.7,
                "status": "ok",
                "spacing_endpoint_status": "insufficient_patch_yield",
            }
        )
    return pd.DataFrame(rows)


def test_join_preserves_annotation_rows() -> None:
    matched, summary = build_full_image_zdisc_mask_validation(synthetic_annotations(), synthetic_images())

    assert len(matched) == 3
    assert summary["total_full_image_annotations"] == 3


def test_unmatched_images_are_reported() -> None:
    matched, summary = build_full_image_zdisc_mask_validation(synthetic_annotations(), synthetic_images(include_missing=True))

    assert summary["unmatched_rows"] == 1
    assert "unmatched_image" in set(matched["validation_match_status"])


def test_donor_id_mismatch_is_detected() -> None:
    matched, summary = build_full_image_zdisc_mask_validation(synthetic_annotations(), synthetic_images(donor_mismatch=True))

    assert summary["donor_id_mismatches"] == 1
    assert "donor_id_mismatch" in set(matched["validation_match_status"])


def test_axial_angular_error_handles_wraparound() -> None:
    assert axial_angular_error_deg(175, 5) == 10.0


def test_orientation_metrics_skip_nan_manual_orientations() -> None:
    _, summary = build_full_image_zdisc_mask_validation(synthetic_annotations(), synthetic_images())

    assert summary["n_orientation_pairs"] == 2
    assert summary["median_axial_error_deg"] == 12.5


def test_group_medians_compute_for_zdisc_and_empty_groups() -> None:
    _, summary = build_full_image_zdisc_mask_validation(synthetic_annotations(), synthetic_images())
    medians = summary["oop_medians_by_annotation_status"]

    assert medians["zdisc_labeled"] == 0.6000000000000001
    assert medians["empty"] == 0.2


def test_spearman_is_skipped_safely_when_n_too_small() -> None:
    _, summary = build_full_image_zdisc_mask_validation(synthetic_annotations(), synthetic_images(), min_n_for_correlation=10)

    spearman = summary["spearman_zdisc_fraction_vs_image_oop"]
    assert spearman["computed"] is False
    assert spearman["reason"] == "too_few_rows"


def test_summary_json_is_serializable(tmp_path: Path) -> None:
    cfg = validation_config(tmp_path)
    tables = tmp_path / "results" / "tables"
    ann_dir = tmp_path / "results" / "full_image_zdisc_annotation"
    tables.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    annotation_path = ann_dir / "full_image_zdisc_annotation_features.csv"
    image_path = tables / "features_per_image.csv"
    synthetic_annotations().to_csv(annotation_path, index=False)
    synthetic_images().to_csv(image_path, index=False)

    _, _, paths = validate_full_image_zdisc_masks(cfg, annotation_features=annotation_path, image_features=image_path)
    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    assert loaded["total_full_image_annotations"] == 3


def test_input_tables_are_not_modified() -> None:
    annotations = synthetic_annotations()
    images = synthetic_images()
    annotations_before = annotations.copy(deep=True)
    images_before = images.copy(deep=True)

    build_full_image_zdisc_mask_validation(annotations, images)

    pd.testing.assert_frame_equal(annotations, annotations_before)
    pd.testing.assert_frame_equal(images, images_before)


def test_donor_id_preserved_as_string() -> None:
    matched, _ = build_full_image_zdisc_mask_validation(synthetic_annotations(), synthetic_images())

    assert matched["donor_id"].map(type).eq(str).all()
    assert matched.loc[0, "donor_id"] == "2.007"
