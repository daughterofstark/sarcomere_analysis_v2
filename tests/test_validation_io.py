from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sarcomere_analysis.validation_io import (
    REQUIRED_VALIDATION_COLUMNS,
    VALIDATION_COLUMNS,
    audit_validation_measurements,
    load_validation_csv,
    template_dataframe,
    validate_manual_measurements,
    write_validation_audit_outputs,
    write_validation_template,
)


def manual_rows(**overrides) -> pd.DataFrame:
    row = {
        "measurement_id": "m1",
        "image_id": "2.007-1",
        "donor_id": "2.007",
        "measurement_type": "oop_manual",
        "manual_value": 0.7,
        "manual_unit": "unitless",
        "expert_id": "expert_a",
        "region_id": "1",
        "patch_id": "",
        "x_px": "",
        "y_px": "",
        "x0_px": "",
        "y0_px": "",
        "x1_px": "",
        "y1_px": "",
        "structure_label": "",
        "notes": "",
        "measurement_date": "",
    }
    row.update(overrides)
    return pd.DataFrame([row], columns=VALIDATION_COLUMNS)


def analysis_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"image_id": "2.007-1", "donor_id": "2.007", "image_oop": 0.2},
            {"image_id": "4.083-1", "donor_id": "4.083", "image_oop": 0.3},
        ]
    )


def test_template_has_required_columns(tmp_path: Path) -> None:
    path = write_validation_template(tmp_path / "manual_validation_template.csv")
    table = pd.read_csv(path, dtype=str)
    assert list(table.columns) == VALIDATION_COLUMNS
    assert set(REQUIRED_VALIDATION_COLUMNS).issubset(table.columns)
    assert table["measurement_id"].str.startswith("EXAMPLE").all()


def test_valid_manual_csv_loads_successfully(tmp_path: Path) -> None:
    path = tmp_path / "manual.csv"
    manual_rows().to_csv(path, index=False)
    loaded = load_validation_csv(path)
    assert len(loaded) == 1
    assert loaded.loc[0, "measurement_type"] == "oop_manual"


def test_donor_id_remains_string(tmp_path: Path) -> None:
    path = tmp_path / "manual.csv"
    manual_rows(donor_id="4.083").to_csv(path, index=False)
    loaded = load_validation_csv(path)
    assert isinstance(loaded.loc[0, "donor_id"], str)
    assert loaded.loc[0, "donor_id"] == "4.083"


def test_missing_required_column_fails_clearly() -> None:
    table = manual_rows().drop(columns=["expert_id"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_manual_measurements(table)


def test_duplicate_measurement_id_fails_clearly() -> None:
    table = pd.concat([manual_rows(), manual_rows()], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate measurement_id"):
        validate_manual_measurements(table)


def test_unknown_measurement_type_fails_by_default() -> None:
    with pytest.raises(ValueError, match="Unknown measurement_type"):
        validate_manual_measurements(manual_rows(measurement_type="novel_manual_metric"))


def test_unknown_measurement_type_passes_with_allow_flag() -> None:
    loaded = validate_manual_measurements(manual_rows(measurement_type="novel_manual_metric"), allow_unknown_types=True)
    assert loaded.loc[0, "measurement_type"] == "novel_manual_metric"


def test_audit_detects_matched_image_rows() -> None:
    measurements = validate_manual_measurements(manual_rows())
    matched, unmatched, summary = audit_validation_measurements(measurements, analysis_rows())
    assert len(matched) == 1
    assert unmatched.empty
    assert summary["rows_matched_to_analysis_per_image"] == 1


def test_audit_detects_unmatched_image_rows() -> None:
    measurements = validate_manual_measurements(manual_rows(image_id="9.999-1", donor_id="9.999"))
    matched, unmatched, summary = audit_validation_measurements(measurements, analysis_rows())
    assert matched.empty
    assert len(unmatched) == 1
    assert summary["unmatched_image_id_rows"] == 1


def test_audit_detects_donor_id_mismatch_for_same_image_id() -> None:
    measurements = validate_manual_measurements(manual_rows(image_id="2.007-1", donor_id="4.083"))
    matched, unmatched, summary = audit_validation_measurements(measurements, analysis_rows())
    assert matched.empty
    assert unmatched.loc[0, "validation_match_status"] == "donor_id_mismatch"
    assert summary["donor_id_mismatch_rows"] == 1


def test_summary_json_is_serializable(tmp_path: Path) -> None:
    measurements = validate_manual_measurements(manual_rows())
    matched, unmatched, summary = audit_validation_measurements(measurements, analysis_rows())
    paths = write_validation_audit_outputs(matched, unmatched, summary, tmp_path)
    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded["total_manual_rows"] == 1
    assert paths["matched_rows"].exists()


def test_no_validation_statistics_are_computed() -> None:
    measurements = validate_manual_measurements(manual_rows())
    _, _, summary = audit_validation_measurements(measurements, analysis_rows())
    assert summary["statistics_computed"] == []
    forbidden = {"correlation", "bland_altman", "regression", "p_value"}
    assert forbidden.isdisjoint(summary.keys())


def test_template_example_rows_require_allow_flag() -> None:
    with pytest.raises(ValueError, match="Example/template rows"):
        validate_manual_measurements(template_dataframe())
    loaded = validate_manual_measurements(template_dataframe(), allow_example_rows=True)
    assert len(loaded) == 3
