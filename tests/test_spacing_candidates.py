from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sarcomere_analysis.diagnostics.spacing_candidates import (
    SPACING_CANDIDATE_COLUMNS,
    compare_to_main_table,
    diagnose_patch_candidates,
    stabilize_candidate_columns,
    summarize_spacing_candidates,
    write_spacing_candidate_outputs,
)
from sarcomere_analysis.spacing.autocorrelation import estimate_spacing_autocorrelation
from test_step_6_spacing import oriented_stripe_patch, single_patch_metrics, step6_config


def candidate_row(**overrides) -> dict[str, object]:
    row = {
        "image_id": "synthetic",
        "donor_id": "2.001",
        "patch_id": "p0",
        "y0": 0,
        "x0": 0,
        "y1": 64,
        "x1": 64,
        "valid_for_spacing_qc": True,
        "final_valid_for_spacing": False,
        "final_invalid_reason": "no_local_peak",
        "expected_min_lag_px": 10.0,
        "expected_max_lag_px": 16.0,
        "selected_lag_px": np.nan,
        "selected_lag_um": np.nan,
        "selected_peak_value": np.nan,
        "baseline_value": np.nan,
        "peak_prominence": np.nan,
        "peak_confidence": 0.0,
        "n_local_peaks_total": 0,
        "n_local_peaks_in_band": 0,
        "best_in_band_lag_px": np.nan,
        "best_in_band_peak_value": np.nan,
        "best_global_lag_px": np.nan,
        "best_global_peak_value": np.nan,
        "rejected_reason_diagnostic": "no_local_peak",
    }
    row.update(overrides)
    return row


def test_candidate_summary_handles_no_peaks() -> None:
    table = pd.DataFrame([candidate_row()])
    summary = summarize_spacing_candidates(table)
    assert summary["patches_with_any_local_peak"] == 0
    assert summary["patches_with_local_peak_inside_expected_band"] == 0
    assert summary["final_accepted_patch_count"] == 0


def test_candidate_summary_detects_peaks_inside_expected_band() -> None:
    table = pd.DataFrame(
        [
            candidate_row(
                final_valid_for_spacing=True,
                final_invalid_reason="ok",
                n_local_peaks_total=2,
                n_local_peaks_in_band=1,
                best_in_band_lag_px=12.0,
                best_global_lag_px=12.0,
                peak_confidence=0.3,
                peak_prominence=0.3,
            )
        ]
    )
    summary = summarize_spacing_candidates(table)
    assert summary["patches_with_local_peak_inside_expected_band"] == 1
    assert summary["best_in_band_lag_px_distribution"] == {"12.0": 1}


def test_candidate_summary_detects_best_global_peak_outside_band() -> None:
    table = pd.DataFrame(
        [
            candidate_row(
                n_local_peaks_total=1,
                n_local_peaks_in_band=0,
                best_global_lag_px=5.0,
                best_global_peak_value=0.9,
            )
        ]
    )
    summary = summarize_spacing_candidates(table)
    assert summary["patches_where_best_global_peak_outside_expected_band"] == 1


def test_diagnostic_output_has_required_columns() -> None:
    table = stabilize_candidate_columns(pd.DataFrame([{"image_id": "synthetic", "patch_id": "p0"}]))
    assert list(table.columns) == SPACING_CANDIDATE_COLUMNS


def test_json_summary_is_serializable(tmp_path: Path) -> None:
    table = pd.DataFrame([candidate_row(n_local_peaks_total=1, best_global_lag_px=12.0)])
    summary = summarize_spacing_candidates(table)
    paths = write_spacing_candidate_outputs(table, summary, tmp_path)
    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded["total_analyzed_patches"] == 1
    assert paths["candidates"].exists()


def test_comparison_logic_detects_mismatched_final_valid_flags() -> None:
    candidates = pd.DataFrame([candidate_row(final_valid_for_spacing=True)])
    main = pd.DataFrame(
        [
            {
                "image_id": "synthetic",
                "patch_id": "p0",
                "valid_for_spacing_final": False,
            }
        ]
    )
    comparison = compare_to_main_table(candidates, main)
    assert comparison["available"]
    assert comparison["mismatch_count"] == 1


def test_candidate_hook_does_not_change_final_estimator_output() -> None:
    config = step6_config()
    config["spacing"]["min_spacing_um"] = 1.0
    config["spacing"]["max_spacing_um"] = 2.0
    config["spacing"]["min_periodicity_confidence"] = 0.05
    image = oriented_stripe_patch(16.0, np.pi / 2, shape=(128, 128), noise_sigma=0.02)
    patch_row = single_patch_metrics(valid_for_spacing=True, orientation=np.pi / 2).iloc[0].copy()
    patch_row["image_id"] = "synthetic"
    patch_row["patch_id"] = "synthetic_p00000"
    patch_row["y1"] = 128
    patch_row["x1"] = 128

    estimator = estimate_spacing_autocorrelation(image, np.pi / 2, config)
    diagnostic = diagnose_patch_candidates(image, patch_row, config)

    assert estimator.valid_for_spacing_final == diagnostic["final_valid_for_spacing"]
    assert abs(estimator.patch_spacing_px - diagnostic["selected_lag_px"]) <= 1e-6
    assert diagnostic["n_local_peaks_in_band"] >= 1
