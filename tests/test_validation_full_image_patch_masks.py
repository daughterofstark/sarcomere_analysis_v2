from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sarcomere_analysis.validation_full_image_patch_masks import (
    build_full_image_patch_mask_validation,
    extract_one_manual_patch_feature,
    validate_full_image_patch_masks,
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


def write_png(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)
    return path


def synthetic_files(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    out = tmp_path / "results" / "full_image_zdisc_annotation"
    image_path = write_png(out / "images" / "FULL_0001__2.007-1.png", np.full((20, 20), 100, dtype=np.uint8))
    mask = np.zeros((20, 20), dtype=np.uint8)
    np.fill_diagonal(mask[:10, :10], 1)
    mask[0:10, 10:20] = 2
    mask_path = write_png(out / "masks" / "FULL_0001__2.007-1_mask.png", mask)
    index = pd.DataFrame(
        [
            {
                "annotation_id": "FULL_0001",
                "image_id": "2.007-1",
                "donor_id": "2.007",
                "patch_id": "2.007-1",
                "annotation_image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        ]
    )
    patches = synthetic_patches()
    return index, patches, mask_path


def synthetic_patches(donor_mismatch: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": "2.007-1",
                "donor_id": "9.999" if donor_mismatch else "2.007",
                "patch_id": "2.007-1_p00000",
                "y0": 0,
                "x0": 0,
                "y1": 10,
                "x1": 10,
                "patch_oop": 0.8,
                "patch_mean_orientation_deg": 45.0,
                "valid_for_orientation": True,
                "valid_for_periodicity": True,
                "valid_for_spacing": True,
                "invalid_reason": "ok",
            },
            {
                "image_id": "2.007-1",
                "donor_id": "2.007",
                "patch_id": "2.007-1_p00001",
                "y0": 0,
                "x0": 10,
                "y1": 10,
                "x1": 20,
                "patch_oop": 0.2,
                "patch_mean_orientation_deg": 90.0,
                "valid_for_orientation": True,
                "valid_for_periodicity": True,
                "valid_for_spacing": False,
                "invalid_reason": "ok",
            },
            {
                "image_id": "2.007-1",
                "donor_id": "2.007",
                "patch_id": "2.007-1_p00002",
                "y0": 10,
                "x0": 0,
                "y1": 20,
                "x1": 10,
                "patch_oop": 0.1,
                "patch_mean_orientation_deg": np.nan,
                "valid_for_orientation": False,
                "valid_for_periodicity": False,
                "valid_for_spacing": False,
                "invalid_reason": "low_tissue_fraction",
            },
        ]
    )


def test_patch_level_mask_crop_matches_patch_coordinates() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:5, 0:5] = 1
    row = pd.Series({"annotation_id": "A", "image_id": "I", "donor_id": "D"})
    patch = pd.Series({"patch_id": "P", "y0": 0, "x0": 0, "y1": 10, "x1": 10})

    result = extract_one_manual_patch_feature(row, patch, mask, min_zdisc_pixels=1)

    assert result["manual_zdisc_pixel_count"] == 25
    assert result["manual_zdisc_pixel_fraction"] == 0.25


def test_empty_patch_returns_empty_status(tmp_path: Path) -> None:
    index, patches, _ = synthetic_files(tmp_path)
    matched, _ = build_full_image_patch_mask_validation(index, patches)

    empty = matched.loc[matched["patch_id"] == "2.007-1_p00002"].iloc[0]
    assert empty["manual_patch_annotation_status"] == "empty"


def test_zdisc_labeled_patch_returns_zdisc_labeled(tmp_path: Path) -> None:
    index, patches, _ = synthetic_files(tmp_path)
    matched, _ = build_full_image_patch_mask_validation(index, patches, min_zdisc_pixels=5)

    labeled = matched.loc[matched["patch_id"] == "2.007-1_p00000"].iloc[0]
    assert labeled["manual_patch_annotation_status"] == "zdisc_labeled"
    assert bool(labeled["manual_has_zdisc_labels"]) is True


def test_orientation_finite_for_diagonal_synthetic_label_pixels(tmp_path: Path) -> None:
    index, patches, _ = synthetic_files(tmp_path)
    matched, summary = build_full_image_patch_mask_validation(index, patches, min_zdisc_pixels=5)

    labeled = matched.loc[matched["patch_id"] == "2.007-1_p00000"].iloc[0]
    assert np.isfinite(labeled["manual_patch_orientation_deg"])
    assert summary["patches_manual_orientation_estimable"] == 1


def test_axial_angular_error_handles_wraparound() -> None:
    assert axial_angular_error_deg(175, 5) == 10.0


def test_join_preserves_patch_rows_for_selected_images(tmp_path: Path) -> None:
    index, patches, _ = synthetic_files(tmp_path)
    matched, summary = build_full_image_patch_mask_validation(index, patches)

    assert len(matched) == 3
    assert summary["matched_patch_rows"] == 3


def test_missing_automated_patch_features_are_reported(tmp_path: Path) -> None:
    index, patches, _ = synthetic_files(tmp_path)
    missing_index = pd.concat(
        [
            index,
            pd.DataFrame(
                [
                    {
                        "annotation_id": "FULL_0002",
                        "image_id": "3.032-1",
                        "donor_id": "3.032",
                        "patch_id": "3.032-1",
                        "annotation_image_path": index.loc[0, "annotation_image_path"],
                        "mask_path": index.loc[0, "mask_path"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    _, summary = build_full_image_patch_mask_validation(missing_index, patches)

    assert summary["full_images_without_patch_features"] == ["3.032-1"]


def test_donor_id_mismatch_is_reported(tmp_path: Path) -> None:
    index, _, _ = synthetic_files(tmp_path)
    matched, summary = build_full_image_patch_mask_validation(index, synthetic_patches(donor_mismatch=True))

    assert summary["donor_id_mismatches"] == 1
    assert "donor_id_mismatch" in set(matched["validation_match_status"])


def test_group_medians_compute_safely(tmp_path: Path) -> None:
    index, patches, _ = synthetic_files(tmp_path)
    _, summary = build_full_image_patch_mask_validation(index, patches)
    medians = summary["oop_medians_by_manual_patch_status"]

    assert medians["zdisc_labeled"] == 0.8
    assert medians["ignore_only"] == 0.2
    assert medians["empty"] == 0.1


def test_spearman_skips_safely_when_n_too_small(tmp_path: Path) -> None:
    index, patches, _ = synthetic_files(tmp_path)
    _, summary = build_full_image_patch_mask_validation(index, patches, min_n_for_correlation=10)

    spearman = summary["spearman_zdisc_fraction_vs_patch_oop"]
    assert spearman["computed"] is False
    assert spearman["reason"] == "too_few_rows"


def test_summary_json_serializable(tmp_path: Path) -> None:
    cfg = validation_config(tmp_path)
    index, patches, _ = synthetic_files(tmp_path)
    ann_dir = tmp_path / "results" / "full_image_zdisc_annotation"
    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    index.to_csv(ann_dir / "full_image_annotation_index.csv", index=False)
    patches.to_csv(tables / "features_per_patch.csv", index=False)

    _, _, paths = validate_full_image_patch_masks(cfg)
    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    assert loaded["total_automated_patches_in_annotated_images"] == 3


def test_input_masks_and_production_tables_are_not_modified(tmp_path: Path) -> None:
    index, patches, mask_path = synthetic_files(tmp_path)
    mask_bytes = mask_path.read_bytes()
    index_before = index.copy(deep=True)
    patches_before = patches.copy(deep=True)

    build_full_image_patch_mask_validation(index, patches)

    assert mask_path.read_bytes() == mask_bytes
    pd.testing.assert_frame_equal(index, index_before)
    pd.testing.assert_frame_equal(patches, patches_before)
