from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_batch_outputs import audit_batch_outputs
from sarcomere_analysis.schemas import (
    BATCH_RUN_SUMMARY_COLUMNS,
    IMAGE_METRICS_COLUMNS,
    PATCH_METRICS_COLUMNS,
)


def audit_config(tmp_path: Path) -> dict:
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


def write_minimal_outputs(tmp_path: Path, all_nan_spacing: bool = False, missing_patch_column: bool = False) -> None:
    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        [
            {"image_id": "2.007-1", "donor_id": "2.007", "image_path": "/tmp/2.007-1.tif"},
            {"image_id": "2.007-2", "donor_id": "2.007", "image_path": "/tmp/2.007-2.tif"},
        ]
    ).to_csv(tables / "manifest.csv", index=False)

    pd.DataFrame(
        [
            {
                **{column: np.nan for column in IMAGE_METRICS_COLUMNS},
                "image_id": "2.007-1",
                "donor_id": "2.007",
                "total_patches": 1,
                "valid_orientation_patches": 1,
                "n_spacing_valid_patches": 0,
                "spacing_valid_fraction": 0.0,
                "image_oop": 0.5,
                "image_oop_heterogeneity": 0.1,
                "image_spacing_median_um": np.nan if all_nan_spacing else 1.8,
                "image_spacing_mean_um": np.nan if all_nan_spacing else 1.8,
                "tissue_fraction": 0.8,
            }
        ]
    ).to_csv(tables / "per_image_metrics.csv", index=False)

    patch_row = {
        **{column: np.nan for column in PATCH_METRICS_COLUMNS},
        "image_id": "2.007-1",
        "donor_id": "2.007",
        "patch_id": "p0",
        "invalid_reason": "ok",
        "spacing_invalid_reason": "low_periodicity_confidence",
    }
    patch = pd.DataFrame([patch_row])
    if missing_patch_column:
        patch = patch.drop(columns=["patch_spacing_um"])
    patch.to_csv(tables / "per_patch_metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                **{column: "" for column in BATCH_RUN_SUMMARY_COLUMNS},
                "image_id": "2.007-1",
                "donor_id": "2.007",
                "status": "ok",
            },
            {
                **{column: "" for column in BATCH_RUN_SUMMARY_COLUMNS},
                "image_id": "2.007-2",
                "donor_id": "2.007",
                "status": "error",
                "error_message": "boom",
            },
        ]
    ).to_csv(tables / "batch_run_summary.csv", index=False)


def test_audit_works_on_synthetic_minimal_outputs(tmp_path: Path) -> None:
    write_minimal_outputs(tmp_path)
    audit = audit_batch_outputs(audit_config(tmp_path))
    assert audit["total_images_expected_from_manifest"] == 2
    assert audit["total_images_processed"] == 2
    assert audit["number_ok"] == 1
    assert audit["total_patch_rows"] == 1
    assert audit["images_per_donor_distribution"] == {"2.007": 1}


def test_audit_preserves_string_like_donor_ids(tmp_path: Path) -> None:
    write_minimal_outputs(tmp_path)
    tables = tmp_path / "results" / "tables"
    per_image = pd.read_csv(tables / "per_image_metrics.csv")
    per_image["donor_id"] = ["3.110"]
    per_image.to_csv(tables / "per_image_metrics.csv", index=False)
    audit = audit_batch_outputs(audit_config(tmp_path))
    assert "3.110" in audit["images_per_donor_distribution"]


def test_audit_detects_missing_required_columns(tmp_path: Path) -> None:
    write_minimal_outputs(tmp_path, missing_patch_column=True)
    audit = audit_batch_outputs(audit_config(tmp_path))
    assert "patch_spacing_um" in audit["missing_required_columns"]["per_patch_metrics"]


def test_audit_reports_error_rows(tmp_path: Path) -> None:
    write_minimal_outputs(tmp_path)
    audit = audit_batch_outputs(audit_config(tmp_path))
    assert audit["number_error"] == 1
    assert audit["errors"] == [{"image_id": "2.007-2", "error_message": "boom"}]


def test_audit_handles_all_nan_spacing_safely(tmp_path: Path) -> None:
    write_minimal_outputs(tmp_path, all_nan_spacing=True)
    audit = audit_batch_outputs(audit_config(tmp_path))
    assert audit["nan_rates"]["image_spacing_median_um"] == 1.0
    assert audit["summary_distribution"]["image_spacing_median_um"]["median"] is None
