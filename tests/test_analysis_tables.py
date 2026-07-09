from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sarcomere_analysis.analysis_tables import (
    ANALYSIS_DONOR_COLUMNS,
    ANALYSIS_IMAGE_COLUMNS,
    build_analysis_tables,
    write_analysis_outputs,
)


def image_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": "4.083-1",
                "donor_id": "4.083",
                "status": "ok",
                "image_oop": 0.2,
                "image_oop_heterogeneity": 0.03,
                "n_orientation_valid_patches": 10,
                "valid_orientation_patches": 10,
                "orientation_valid_fraction": 0.9,
                "n_spacing_valid_patches": 0,
                "spacing_valid_fraction": 0.0,
                "spacing_low_yield_flag": True,
                "spacing_endpoint_status": "insufficient_patch_yield",
                "tissue_fraction": 0.8,
                "total_patches": 12,
            },
            {
                "image_id": "2.007-1",
                "donor_id": "2.007",
                "status": "ok",
                "image_oop": 0.5,
                "image_oop_heterogeneity": 0.04,
                "n_orientation_valid_patches": 9,
                "valid_orientation_patches": 9,
                "orientation_valid_fraction": 0.75,
                "n_spacing_valid_patches": 1,
                "spacing_valid_fraction": 0.08,
                "spacing_low_yield_flag": True,
                "spacing_endpoint_status": "insufficient_patch_yield",
                "tissue_fraction": 0.7,
                "total_patches": 12,
            },
        ]
    )


def donor_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "donor_id": "4.083",
                "n_images": 1,
                "n_ok_images": 1,
                "median_image_oop": 0.2,
                "mean_image_oop": 0.2,
                "sd_image_oop": 0.0,
                "median_oop_heterogeneity": 0.03,
                "mean_oop_heterogeneity": 0.03,
                "median_orientation_valid_fraction": 0.9,
                "mean_orientation_valid_fraction": 0.9,
                "total_valid_orientation_patches": 10,
                "total_patch_rows": 12,
                "total_valid_spacing_patches": 0,
                "n_images_insufficient_spacing": 1,
                "spacing_endpoint_status": "insufficient_patch_yield",
                "spacing_global_status": "exploratory_low_yield",
            },
            {
                "donor_id": "2.007",
                "n_images": 1,
                "n_ok_images": 1,
                "median_image_oop": 0.5,
                "mean_image_oop": 0.5,
                "sd_image_oop": 0.0,
                "median_oop_heterogeneity": 0.04,
                "mean_oop_heterogeneity": 0.04,
                "median_orientation_valid_fraction": 0.75,
                "mean_orientation_valid_fraction": 0.75,
                "total_valid_orientation_patches": 9,
                "total_patch_rows": 12,
                "total_valid_spacing_patches": 1,
                "n_images_insufficient_spacing": 1,
                "spacing_endpoint_status": "insufficient_patch_yield",
                "spacing_global_status": "exploratory_low_yield",
            },
        ]
    )


def enriched_manifest(include_second: bool = True) -> pd.DataFrame:
    rows = [
        {
            "image_id": "4.083-1",
            "donor_id": "4.083",
            "is_healthy": True,
            "image_path": "/raw/4.083-1.tif",
            "region_id": "1",
            "donor_group": "control",
        }
    ]
    if include_second:
        rows.append(
            {
                "image_id": "2.007-1",
                "donor_id": "2.007",
                "is_healthy": False,
                "image_path": "/raw/2.007-1.tif",
                "region_id": "1",
                "donor_group": "case",
            }
        )
    return pd.DataFrame(rows)


def donor_metadata(include_second: bool = True) -> pd.DataFrame:
    rows = [
        {"donor_id": "4.083", "is_healthy": True, "n_images": 1, "donor_group": "control"},
    ]
    if include_second:
        rows.append({"donor_id": "2.007", "is_healthy": False, "n_images": 1, "donor_group": "case"})
    return pd.DataFrame(rows)


def test_image_join_preserves_one_row_per_image() -> None:
    per_image, _, _ = build_analysis_tables(image_features(), donor_features(), enriched_manifest(), donor_metadata())
    assert len(per_image) == 2
    assert per_image["image_id"].is_unique


def test_donor_join_preserves_one_row_per_donor() -> None:
    _, per_donor, _ = build_analysis_tables(image_features(), donor_features(), enriched_manifest(), donor_metadata())
    assert len(per_donor) == 2
    assert per_donor["donor_id"].is_unique


def test_duplicate_image_id_fails_clearly() -> None:
    images = pd.concat([image_features(), image_features().head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate image_id"):
        build_analysis_tables(images, donor_features(), enriched_manifest(), donor_metadata())


def test_duplicate_donor_id_fails_clearly() -> None:
    donors = pd.concat([donor_features(), donor_features().head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate donor_id"):
        build_analysis_tables(image_features(), donors, enriched_manifest(), donor_metadata())


def test_donor_id_remains_string() -> None:
    _, per_donor, _ = build_analysis_tables(image_features(), donor_features(), enriched_manifest(), donor_metadata())
    assert per_donor["donor_id"].map(type).eq(str).all()
    assert "4.083" in set(per_donor["donor_id"])


def test_missing_metadata_is_reported() -> None:
    _, _, summary = build_analysis_tables(image_features(), donor_features(), enriched_manifest(include_second=False), donor_metadata(include_second=False))
    assert summary["missing_image_metadata_count"] == 1
    assert summary["missing_donor_metadata_count"] == 1
    assert summary["missing_image_metadata_ids"] == ["2.007-1"]
    assert summary["missing_donor_metadata_ids"] == ["2.007"]


def test_strict_mode_fails_on_missing_metadata() -> None:
    with pytest.raises(ValueError, match="Missing image metadata"):
        build_analysis_tables(
            image_features(),
            donor_features(),
            enriched_manifest(include_second=False),
            donor_metadata(include_second=False),
            strict=True,
        )


def test_healthy_counts_are_correct() -> None:
    per_image, per_donor, summary = build_analysis_tables(
        image_features(),
        donor_features(),
        enriched_manifest(),
        donor_metadata(),
        expected_healthy_donor_count=1,
    )
    assert int(per_image["is_healthy"].sum()) == 1
    assert int(per_donor["is_healthy"].sum()) == 1
    assert summary["healthy_donor_count"] == 1


def test_spacing_exploratory_status_is_preserved() -> None:
    per_image, per_donor, summary = build_analysis_tables(image_features(), donor_features(), enriched_manifest(), donor_metadata())
    assert set(per_image["spacing_endpoint_status"]) == {"insufficient_patch_yield"}
    assert set(per_donor["spacing_global_status"]) == {"exploratory_low_yield"}
    assert summary["spacing_is_exploratory_low_yield"]


def test_summary_json_is_serializable(tmp_path: Path) -> None:
    per_image, per_donor, summary = build_analysis_tables(image_features(), donor_features(), enriched_manifest(), donor_metadata())
    paths = write_analysis_outputs(per_image, per_donor, summary, tmp_path)
    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded["image_rows"] == 2
    assert paths["summary_txt"].exists()


def test_input_tables_are_not_modified_in_place() -> None:
    images = image_features()
    donors = donor_features()
    manifest = enriched_manifest()
    metadata = donor_metadata()
    before_images = images.copy(deep=True)
    before_donors = donors.copy(deep=True)
    before_manifest = manifest.copy(deep=True)
    before_metadata = metadata.copy(deep=True)

    build_analysis_tables(images, donors, manifest, metadata)

    pd.testing.assert_frame_equal(images, before_images)
    pd.testing.assert_frame_equal(donors, before_donors)
    pd.testing.assert_frame_equal(manifest, before_manifest)
    pd.testing.assert_frame_equal(metadata, before_metadata)


def test_output_schema_contains_required_columns() -> None:
    per_image, per_donor, _ = build_analysis_tables(image_features(), donor_features(), enriched_manifest(), donor_metadata())
    assert list(per_image.columns[: len(ANALYSIS_IMAGE_COLUMNS)]) == ANALYSIS_IMAGE_COLUMNS
    assert list(per_donor.columns[: len(ANALYSIS_DONOR_COLUMNS)]) == ANALYSIS_DONOR_COLUMNS
