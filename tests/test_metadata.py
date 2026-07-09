from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sarcomere_analysis.metadata import (
    DONOR_METADATA_REQUIRED_COLUMNS,
    ENRICHED_MANIFEST_REQUIRED_COLUMNS,
    enrich_manifest,
    load_external_metadata,
    load_manifest_table,
    write_metadata_outputs,
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
        "metadata": {"healthy_donor_ids": ["4.083", "5.003", "6.052", "7.028"]},
    }


def manifest_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"image_id": "4.083-1", "donor_id": "4.083", "region_id": "1", "filename": "4.083-1.tif", "image_path": "/raw/4.083-1.tif"},
            {"image_id": "4.083-2", "donor_id": "4.083", "region_id": "2", "filename": "4.083-2.tif", "image_path": "/raw/4.083-2.tif"},
            {"image_id": "2.007-1", "donor_id": "2.007", "region_id": "1", "filename": "2.007-1.tif", "image_path": "/raw/2.007-1.tif"},
        ]
    )


def external_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"donor_id": "4.083", "age": 55, "group": "control"},
            {"donor_id": "9.999", "age": 61, "group": "metadata_only"},
        ]
    )


def test_donor_ids_remain_strings_and_not_float_coerced(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    manifest_table().to_csv(path, index=False)
    cfg = metadata_config(tmp_path)
    loaded = load_manifest_table(cfg, path)
    assert loaded["donor_id"].map(type).eq(str).all()
    assert "4.083" in set(loaded["donor_id"])


def test_known_healthy_donors_are_flagged_correctly(tmp_path: Path) -> None:
    enriched, donor, _ = enrich_manifest(manifest_table(), metadata_config(tmp_path))
    assert enriched.loc[enriched["donor_id"] == "4.083", "is_healthy"].all()
    assert not enriched.loc[enriched["donor_id"] == "2.007", "is_healthy"].any()
    assert donor.loc[donor["donor_id"] == "4.083", "is_healthy"].iloc[0]


def test_enriched_manifest_preserves_one_row_per_image(tmp_path: Path) -> None:
    enriched, _, _ = enrich_manifest(manifest_table(), metadata_config(tmp_path))
    assert len(enriched) == 3
    assert enriched["image_id"].is_unique


def test_donor_metadata_preserves_one_row_per_donor(tmp_path: Path) -> None:
    _, donor, _ = enrich_manifest(manifest_table(), metadata_config(tmp_path))
    assert len(donor) == 2
    assert donor["donor_id"].is_unique


def test_metadata_join_preserves_external_columns(tmp_path: Path) -> None:
    enriched, donor, _ = enrich_manifest(manifest_table(), metadata_config(tmp_path), metadata=external_metadata())
    assert "age" in enriched.columns
    assert "group" in donor.columns
    assert donor.loc[donor["donor_id"] == "4.083", "age"].iloc[0] == 55


def test_unmatched_manifest_donors_are_reported(tmp_path: Path) -> None:
    _, _, summary = enrich_manifest(manifest_table(), metadata_config(tmp_path), metadata=external_metadata())
    assert summary["unmatched_manifest_donors"] == ["2.007"]


def test_unmatched_metadata_donors_are_reported(tmp_path: Path) -> None:
    _, _, summary = enrich_manifest(manifest_table(), metadata_config(tmp_path), metadata=external_metadata())
    assert summary["unmatched_metadata_donors"] == ["9.999"]


def test_strict_mode_fails_on_unmatched_donors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unmatched donors"):
        enrich_manifest(manifest_table(), metadata_config(tmp_path), metadata=external_metadata(), strict=True)


def test_missing_metadata_still_produces_valid_outputs(tmp_path: Path) -> None:
    enriched, donor, summary = enrich_manifest(manifest_table(), metadata_config(tmp_path), metadata=None)
    assert list(enriched.columns[: len(ENRICHED_MANIFEST_REQUIRED_COLUMNS)]) == ENRICHED_MANIFEST_REQUIRED_COLUMNS
    assert list(donor.columns[: len(DONOR_METADATA_REQUIRED_COLUMNS)]) == DONOR_METADATA_REQUIRED_COLUMNS
    assert not summary["metadata_provided"]


def test_summary_json_is_serializable(tmp_path: Path) -> None:
    enriched, donor, summary = enrich_manifest(manifest_table(), metadata_config(tmp_path), metadata=external_metadata())
    paths = write_metadata_outputs(enriched, donor, summary, tmp_path)
    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded["enriched_manifest_rows"] == 3
    assert paths["summary_txt"].exists()


def test_load_external_metadata_csv_preserves_donor_id_strings(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    external_metadata().to_csv(path, index=False)
    loaded = load_external_metadata(path)
    assert loaded["donor_id"].map(type).eq(str).all()
    assert "4.083" in set(loaded["donor_id"])
