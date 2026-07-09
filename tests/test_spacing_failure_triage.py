from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sarcomere_analysis.diagnostics.spacing_failure_triage import (
    infer_spacing_rejection_stage,
    triage_spacing_failures,
    write_spacing_failure_outputs,
)


def synthetic_patch_table(include_optional: bool = False) -> pd.DataFrame:
    rows = [
        {
            "image_id": "3.110-1",
            "donor_id": "3.110",
            "patch_id": "p0",
            "tissue_fraction": 0.9,
            "intensity_std": 0.08,
            "gradient_energy": 0.002,
            "valid_for_orientation": True,
            "valid_for_periodicity": True,
            "valid_for_spacing": True,
            "invalid_reason": "ok",
            "patch_oop": 0.7,
            "patch_mean_orientation_rad": 0.1,
            "patch_spacing_px": 12.0,
            "patch_spacing_um": 1.56,
            "patch_spacing_confidence": 0.2,
            "valid_for_spacing_final": True,
            "spacing_invalid_reason": "ok",
        },
        {
            "image_id": "3.110-1",
            "donor_id": "3.110",
            "patch_id": "p1",
            "tissue_fraction": 0.8,
            "intensity_std": 0.07,
            "gradient_energy": 0.001,
            "valid_for_orientation": True,
            "valid_for_periodicity": True,
            "valid_for_spacing": True,
            "invalid_reason": "ok",
            "patch_oop": 0.6,
            "patch_mean_orientation_rad": 0.2,
            "patch_spacing_px": np.nan,
            "patch_spacing_um": np.nan,
            "patch_spacing_confidence": 0.0,
            "valid_for_spacing_final": False,
            "spacing_invalid_reason": "no_local_peak",
        },
        {
            "image_id": "3.110-2",
            "donor_id": "3.110",
            "patch_id": "p2",
            "tissue_fraction": 0.2,
            "intensity_std": 0.01,
            "gradient_energy": 0.00001,
            "valid_for_orientation": False,
            "valid_for_periodicity": False,
            "valid_for_spacing": False,
            "invalid_reason": "low_tissue_fraction",
            "patch_oop": np.nan,
            "patch_mean_orientation_rad": np.nan,
            "patch_spacing_px": np.nan,
            "patch_spacing_um": np.nan,
            "patch_spacing_confidence": 0.0,
            "valid_for_spacing_final": False,
            "spacing_invalid_reason": "low_tissue_fraction;failed_patch_qc",
        },
        {
            "image_id": "3.110-2",
            "donor_id": "3.110",
            "patch_id": "p3",
            "tissue_fraction": 0.85,
            "intensity_std": 0.06,
            "gradient_energy": 0.001,
            "valid_for_orientation": True,
            "valid_for_periodicity": True,
            "valid_for_spacing": True,
            "invalid_reason": "ok",
            "patch_oop": np.nan,
            "patch_mean_orientation_rad": np.nan,
            "patch_spacing_px": np.nan,
            "patch_spacing_um": np.nan,
            "patch_spacing_confidence": 0.0,
            "valid_for_spacing_final": False,
            "spacing_invalid_reason": "missing_orientation",
        },
    ]
    table = pd.DataFrame(rows)
    if include_optional:
        table["selected_lag_px"] = [12.0, 12.0, np.nan, np.nan]
        table["peak_score"] = [0.5, 0.1, np.nan, np.nan]
        table["spacing_rejection_stage"] = ["accepted", "peak_picking", "failed_patch_qc", "missing_orientation"]
    return table


def synthetic_image_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"image_id": "3.110-1", "donor_id": "3.110", "n_spacing_valid_patches": 1},
            {"image_id": "3.110-2", "donor_id": "3.110", "n_spacing_valid_patches": 0},
        ]
    )


def test_triage_handles_missing_optional_diagnostic_columns() -> None:
    summary, _ = triage_spacing_failures(synthetic_patch_table(False), synthetic_image_table())
    assert not summary["candidate_level_detail_available"]
    assert "lacks candidate-level" in summary["candidate_lag_confidence_summary"]["message"]


def test_counts_total_and_final_valid_spacing_correctly() -> None:
    summary, _ = triage_spacing_failures(synthetic_patch_table(), synthetic_image_table())
    assert summary["total_patches"] == 4
    assert summary["qc_valid_spacing_patches"] == 3
    assert summary["final_valid_spacing_patches"] == 1
    assert summary["finite_spacing_px_patches"] == 1


def test_by_image_aggregation_preserves_image_id() -> None:
    _, by_image = triage_spacing_failures(synthetic_patch_table(), synthetic_image_table())
    assert list(by_image["image_id"]) == ["3.110-1", "3.110-2"]
    assert by_image.loc[0, "donor_id"] == "3.110"


def test_dominant_rejection_reason_is_populated() -> None:
    _, by_image = triage_spacing_failures(synthetic_patch_table(), synthetic_image_table())
    assert by_image.loc[0, "dominant_rejection_reason"] == "no_local_peak"
    assert infer_spacing_rejection_stage(False, "low_periodicity_confidence") == "confidence"


def test_summary_json_is_serializable(tmp_path) -> None:
    summary, by_image = triage_spacing_failures(synthetic_patch_table(include_optional=True), synthetic_image_table())
    paths = write_spacing_failure_outputs(summary, by_image, tmp_path)
    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded["candidate_level_detail_available"]
    assert paths["by_image"].exists()


def test_triage_does_not_modify_input_validity_columns_in_place() -> None:
    patch = synthetic_patch_table()
    before = patch[["valid_for_spacing", "valid_for_spacing_final", "spacing_invalid_reason"]].copy(deep=True)
    triage_spacing_failures(patch, synthetic_image_table())
    pd.testing.assert_frame_equal(before, patch[["valid_for_spacing", "valid_for_spacing_final", "spacing_invalid_reason"]])
