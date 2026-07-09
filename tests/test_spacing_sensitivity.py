from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sarcomere_analysis.diagnostics.spacing_sensitivity import (
    SENSITIVITY_COLUMNS,
    build_spacing_sensitivity_report,
    read_candidate_table,
    stabilize_sensitivity_columns,
    write_sensitivity_outputs,
)


def synthetic_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": "img1",
                "donor_id": "d1",
                "patch_id": "p1",
                "expected_min_lag_px": 12.0,
                "expected_max_lag_px": 18.0,
                "selected_lag_px": 12.0,
                "selected_lag_um": 1.56,
                "selected_peak_value": 0.5,
                "peak_confidence": 0.16,
                "peak_prominence": 0.16,
                "best_in_band_lag_px": 12.0,
                "best_in_band_peak_value": 0.5,
                "best_global_lag_px": 12.0,
                "best_global_peak_value": 0.5,
                "final_valid_for_spacing": True,
            },
            {
                "image_id": "img1",
                "donor_id": "d1",
                "patch_id": "p2",
                "expected_min_lag_px": 12.0,
                "expected_max_lag_px": 18.0,
                "selected_lag_px": 16.0,
                "selected_lag_um": 2.08,
                "selected_peak_value": 0.4,
                "peak_confidence": 0.11,
                "peak_prominence": 0.11,
                "best_in_band_lag_px": 16.0,
                "best_in_band_peak_value": 0.4,
                "best_global_lag_px": 16.0,
                "best_global_peak_value": 0.4,
                "final_valid_for_spacing": False,
            },
            {
                "image_id": "img2",
                "donor_id": "d2",
                "patch_id": "p3",
                "expected_min_lag_px": 12.0,
                "expected_max_lag_px": 18.0,
                "selected_lag_px": pd.NA,
                "selected_lag_um": pd.NA,
                "selected_peak_value": pd.NA,
                "peak_confidence": 0.3,
                "peak_prominence": pd.NA,
                "best_in_band_lag_px": pd.NA,
                "best_in_band_peak_value": pd.NA,
                "best_global_lag_px": 9.0,
                "best_global_peak_value": 0.9,
                "final_valid_for_spacing": False,
            },
            {
                "image_id": "img3",
                "donor_id": "d3",
                "patch_id": "p4",
                "expected_min_lag_px": 12.0,
                "expected_max_lag_px": 18.0,
                "selected_lag_px": pd.NA,
                "selected_lag_um": pd.NA,
                "selected_peak_value": pd.NA,
                "peak_confidence": 0.13,
                "peak_prominence": 0.13,
                "best_in_band_lag_px": 19.0,
                "best_in_band_peak_value": 0.45,
                "best_global_lag_px": 19.0,
                "best_global_peak_value": 0.45,
                "final_valid_for_spacing": False,
            },
        ]
    )


def variant_by_threshold(variants: pd.DataFrame, confidence: float, rule: str = "current_selected_if_available") -> pd.Series:
    match = variants.loc[
        (variants["min_confidence"] == confidence)
        & (variants["peak_rule"] == rule)
        & (variants["band_min_px"] == 12.0)
        & (variants["band_max_px"] == 18.0)
    ]
    assert len(match) == 1
    return match.iloc[0]


def test_current_threshold_variant_reproduces_expected_accepted_count() -> None:
    variants, _ = build_spacing_sensitivity_report(
        synthetic_candidates(),
        confidence_grid=[0.15],
        band_padding_grid=["current"],
        peak_rules=["current_selected_if_available"],
    )
    row = variants.iloc[0]
    assert row["accepted_patch_count"] == 1
    assert row["accepted_image_count"] == 1


def test_lowering_confidence_increases_or_preserves_accepted_count() -> None:
    variants, _ = build_spacing_sensitivity_report(
        synthetic_candidates(),
        confidence_grid=[0.10, 0.15],
        band_padding_grid=["current"],
        peak_rules=["current_selected_if_available"],
    )
    low = variant_by_threshold(variants, 0.10)
    current = variant_by_threshold(variants, 0.15)
    assert low["accepted_patch_count"] >= current["accepted_patch_count"]


def test_widening_band_increases_or_preserves_in_band_candidate_availability() -> None:
    variants, _ = build_spacing_sensitivity_report(
        synthetic_candidates(),
        confidence_grid=[0.12],
        band_padding_grid=["current", "max_plus_1"],
        peak_rules=["in_band_best_only"],
    )
    current = variants.loc[variants["band_max_px"] == 18.0].iloc[0]
    wider = variants.loc[variants["band_max_px"] == 19.0].iloc[0]
    assert wider["accepted_patch_count"] >= current["accepted_patch_count"]


def test_global_best_allowed_is_flagged_high_risk_when_outside_band_dominates() -> None:
    variants, _ = build_spacing_sensitivity_report(
        synthetic_candidates(),
        confidence_grid=[0.10],
        band_padding_grid=["current"],
        peak_rules=["global_best_allowed"],
    )
    assert variants.iloc[0]["artefact_risk_flag"] == "high_artefact_risk"
    assert variants.iloc[0]["interpretation_class"] == "high_artefact_risk"


def test_summary_json_is_serializable(tmp_path: Path) -> None:
    variants, summary = build_spacing_sensitivity_report(
        synthetic_candidates(),
        confidence_grid=[0.15],
        band_padding_grid=["current"],
        peak_rules=["current_selected_if_available"],
    )
    paths = write_sensitivity_outputs(variants, summary, tmp_path)
    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded["variant_count"] == 1


def test_missing_candidate_table_fails_with_clear_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="diagnose_spacing_candidates.py"):
        read_candidate_table(tmp_path / "missing.csv")


def test_input_candidate_table_is_not_modified() -> None:
    table = synthetic_candidates()
    before = table.copy(deep=True)
    build_spacing_sensitivity_report(table, confidence_grid=[0.15], band_padding_grid=["current"], peak_rules=["current_selected_if_available"])
    pd.testing.assert_frame_equal(table, before)


def test_variant_csv_contains_required_columns() -> None:
    variants, _ = build_spacing_sensitivity_report(
        synthetic_candidates(),
        confidence_grid=[0.15],
        band_padding_grid=["current"],
        peak_rules=["current_selected_if_available"],
    )
    assert list(stabilize_sensitivity_columns(variants).columns) == SENSITIVITY_COLUMNS
