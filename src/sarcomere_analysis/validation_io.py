from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_dir


REQUIRED_VALIDATION_COLUMNS = [
    "measurement_id",
    "image_id",
    "donor_id",
    "measurement_type",
    "manual_value",
    "manual_unit",
    "expert_id",
]

OPTIONAL_VALIDATION_COLUMNS = [
    "region_id",
    "patch_id",
    "x_px",
    "y_px",
    "x0_px",
    "y0_px",
    "x1_px",
    "y1_px",
    "structure_label",
    "notes",
    "measurement_date",
]

VALIDATION_COLUMNS = REQUIRED_VALIDATION_COLUMNS + OPTIONAL_VALIDATION_COLUMNS

ALLOWED_MEASUREMENT_TYPES = [
    "oop_manual",
    "orientation_manual_deg",
    "sarcomere_length_manual_um",
    "zdisc_width_manual_um",
    "other",
]


def template_dataframe() -> pd.DataFrame:
    rows = [
        {
            "measurement_id": "EXAMPLE_001",
            "image_id": "2.007-1",
            "donor_id": "2.007",
            "measurement_type": "oop_manual",
            "manual_value": 0.72,
            "manual_unit": "unitless",
            "expert_id": "EXAMPLE_EXPERT",
            "region_id": "1",
            "patch_id": "",
            "x_px": "",
            "y_px": "",
            "x0_px": "",
            "y0_px": "",
            "x1_px": "",
            "y1_px": "",
            "structure_label": "example_orientation_region",
            "notes": "EXAMPLE ROW - replace before real validation",
            "measurement_date": "YYYY-MM-DD",
        },
        {
            "measurement_id": "EXAMPLE_002",
            "image_id": "4.083-1",
            "donor_id": "4.083",
            "measurement_type": "orientation_manual_deg",
            "manual_value": 35.0,
            "manual_unit": "degree",
            "expert_id": "EXAMPLE_EXPERT",
            "region_id": "1",
            "patch_id": "",
            "x_px": "",
            "y_px": "",
            "x0_px": "",
            "y0_px": "",
            "x1_px": "",
            "y1_px": "",
            "structure_label": "example_orientation_axis",
            "notes": "EXAMPLE ROW - replace before real validation",
            "measurement_date": "YYYY-MM-DD",
        },
        {
            "measurement_id": "EXAMPLE_003",
            "image_id": "2.007-1",
            "donor_id": "2.007",
            "measurement_type": "sarcomere_length_manual_um",
            "manual_value": 1.8,
            "manual_unit": "um",
            "expert_id": "EXAMPLE_EXPERT",
            "region_id": "1",
            "patch_id": "",
            "x_px": "",
            "y_px": "",
            "x0_px": 100,
            "y0_px": 100,
            "x1_px": 200,
            "y1_px": 200,
            "structure_label": "example_spacing_region",
            "notes": "EXAMPLE ROW - spacing remains exploratory_low_yield",
            "measurement_date": "YYYY-MM-DD",
        },
    ]
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def write_validation_template(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    template_dataframe().to_csv(out, index=False)
    return out


def load_validation_csv(
    path: str | Path,
    allow_unknown_types: bool = False,
    allow_example_rows: bool = False,
) -> pd.DataFrame:
    validation_path = Path(path)
    if not validation_path.exists():
        raise FileNotFoundError(f"Manual validation CSV not found: {validation_path}")
    table = pd.read_csv(
        validation_path,
        dtype={
            "measurement_id": str,
            "image_id": str,
            "donor_id": str,
            "region_id": str,
            "patch_id": str,
            "expert_id": str,
        },
        keep_default_na=True,
    )
    return validate_manual_measurements(table, allow_unknown_types=allow_unknown_types, allow_example_rows=allow_example_rows)


def validate_manual_measurements(
    measurements: pd.DataFrame,
    allow_unknown_types: bool = False,
    allow_example_rows: bool = False,
) -> pd.DataFrame:
    result = measurements.copy(deep=True)
    missing_columns = [column for column in REQUIRED_VALIDATION_COLUMNS if column not in result.columns]
    if missing_columns:
        raise ValueError(f"Manual validation CSV missing required columns: {missing_columns}")
    for column in OPTIONAL_VALIDATION_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    result["measurement_id"] = result["measurement_id"].map(standardize_id)
    result["image_id"] = result["image_id"].map(standardize_id)
    result["donor_id"] = result["donor_id"].map(standardize_id)
    result["measurement_type"] = result["measurement_type"].fillna("").astype(str).str.strip()
    result["manual_unit"] = result["manual_unit"].fillna("").astype(str).str.strip()
    result["expert_id"] = result["expert_id"].fillna("").astype(str).str.strip()

    example_mask = is_example_row(result)
    if example_mask.any() and not allow_example_rows:
        example_ids = result.loc[example_mask, "measurement_id"].astype(str).tolist()
        raise ValueError(f"Example/template rows are present and require --allow-example-rows: {example_ids}")

    duplicated = duplicate_measurement_ids(result)
    if duplicated:
        raise ValueError(f"Duplicate measurement_id values found: {duplicated}")

    unknown_types = sorted(set(result.loc[~result["measurement_type"].isin(ALLOWED_MEASUREMENT_TYPES), "measurement_type"].astype(str)) - {""})
    if unknown_types and not allow_unknown_types:
        raise ValueError(f"Unknown measurement_type values found: {unknown_types}")

    values = pd.to_numeric(result["manual_value"], errors="coerce")
    nonempty_manual = result["manual_value"].notna() & (result["manual_value"].astype(str).str.strip() != "")
    invalid_value_ids = result.loc[nonempty_manual & values.isna(), "measurement_id"].astype(str).tolist()
    if invalid_value_ids:
        raise ValueError(f"manual_value must be numeric where present; invalid measurement_id values: {invalid_value_ids}")
    result["manual_value"] = values
    return result[VALIDATION_COLUMNS + [column for column in result.columns if column not in VALIDATION_COLUMNS]]


def load_analysis_per_image(cfg: dict[str, Any], path: str | Path | None = None) -> pd.DataFrame:
    analysis_path = Path(path) if path else output_dir(cfg) / "tables" / "analysis_per_image.csv"
    if not analysis_path.exists():
        raise FileNotFoundError(f"analysis_per_image table not found: {analysis_path}")
    table = pd.read_csv(analysis_path, dtype={"image_id": str, "donor_id": str})
    required = ["image_id", "donor_id"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"analysis_per_image table missing required columns: {missing}")
    table["image_id"] = table["image_id"].map(standardize_id)
    table["donor_id"] = table["donor_id"].map(standardize_id)
    return table


def audit_validation_measurements(
    measurements: pd.DataFrame,
    analysis_per_image: pd.DataFrame,
    is_example_audit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manual = measurements.copy(deep=True)
    analysis = analysis_per_image.copy(deep=True)
    manual["image_id"] = manual["image_id"].map(standardize_id)
    manual["donor_id"] = manual["donor_id"].map(standardize_id)
    analysis["image_id"] = analysis["image_id"].map(standardize_id)
    analysis["donor_id"] = analysis["donor_id"].map(standardize_id)

    joined = manual.merge(
        analysis[["image_id", "donor_id"]].rename(columns={"donor_id": "analysis_donor_id"}),
        on="image_id",
        how="left",
        indicator="_analysis_join_status",
    )
    image_missing = joined["_analysis_join_status"] == "left_only"
    donor_mismatch = (~image_missing) & (joined["donor_id"].astype(str) != joined["analysis_donor_id"].astype(str))
    matched = (~image_missing) & (~donor_mismatch)
    joined["validation_match_status"] = np.select(
        [matched, image_missing, donor_mismatch],
        ["matched", "unmatched_image_id", "donor_id_mismatch"],
        default="unmatched",
    )
    matched_rows = joined.loc[matched].drop(columns=["_analysis_join_status"]).copy()
    unmatched_rows = joined.loc[~matched].drop(columns=["_analysis_join_status"]).copy()
    summary = validation_audit_summary(manual, matched_rows, unmatched_rows, is_example_audit)
    return matched_rows, unmatched_rows, summary


def write_validation_audit_outputs(
    matched_rows: pd.DataFrame,
    unmatched_rows: pd.DataFrame,
    summary: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, Path]:
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": out_dir / "manual_validation_audit_summary.json",
        "summary_txt": out_dir / "manual_validation_audit_summary.txt",
        "matched_rows": out_dir / "manual_validation_matched_rows.csv",
        "unmatched_rows": out_dir / "manual_validation_unmatched_rows.csv",
    }
    with paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
    paths["summary_txt"].write_text(validation_audit_summary_text(summary), encoding="utf-8")
    matched_rows.to_csv(paths["matched_rows"], index=False)
    unmatched_rows.to_csv(paths["unmatched_rows"], index=False)
    return paths


def validation_audit_summary(
    manual: pd.DataFrame,
    matched_rows: pd.DataFrame,
    unmatched_rows: pd.DataFrame,
    is_example_audit: bool,
) -> dict[str, Any]:
    unmatched_image_rows = unmatched_rows.loc[unmatched_rows["validation_match_status"] == "unmatched_image_id"]
    donor_mismatch_rows = unmatched_rows.loc[unmatched_rows["validation_match_status"] == "donor_id_mismatch"]
    missing_fields = missing_required_field_summary(manual)
    return json_safe(
        {
            "is_example_audit": bool(is_example_audit),
            "total_manual_rows": int(len(manual)),
            "unique_images_referenced": int(manual["image_id"].nunique()) if "image_id" in manual.columns else 0,
            "unique_donors_referenced": int(manual["donor_id"].nunique()) if "donor_id" in manual.columns else 0,
            "measurement_type_counts": value_counts(manual.get("measurement_type", pd.Series(dtype=object))),
            "rows_matched_to_analysis_per_image": int(len(matched_rows)),
            "unmatched_image_id_rows": int(len(unmatched_image_rows)),
            "unmatched_image_ids": sorted(unmatched_image_rows.get("image_id", pd.Series(dtype=object)).astype(str).unique().tolist()),
            "donor_id_mismatch_rows": int(len(donor_mismatch_rows)),
            "donor_id_mismatch_measurement_ids": donor_mismatch_rows.get("measurement_id", pd.Series(dtype=object)).astype(str).tolist(),
            "duplicate_measurement_ids": duplicate_measurement_ids(manual),
            "missing_required_fields": missing_fields,
            "missing_required_field_rows": int(sum(len(rows) for rows in missing_fields.values())),
            "statistics_computed": [],
            "caution": (
                "This audit only checks validation data schema and image/donor matching. "
                "It computes no Bland-Altman analysis, correlations, regressions, plots, or biological interpretation."
            ),
        }
    )


def missing_required_field_summary(table: pd.DataFrame) -> dict[str, list[str]]:
    summary: dict[str, list[str]] = {}
    for column in REQUIRED_VALIDATION_COLUMNS:
        if column not in table.columns:
            summary[column] = ["<column_missing>"]
            continue
        values = table[column]
        missing = values.isna() | (values.astype(str).str.strip() == "")
        if missing.any():
            summary[column] = table.loc[missing, "measurement_id"].astype(str).tolist()
    return summary


def duplicate_measurement_ids(table: pd.DataFrame) -> list[str]:
    if "measurement_id" not in table.columns:
        return []
    values = table["measurement_id"].fillna("").astype(str)
    duplicates = sorted(values.loc[values.ne("") & values.duplicated()].unique().tolist())
    return duplicates


def is_example_row(table: pd.DataFrame) -> pd.Series:
    measurement_id = table.get("measurement_id", pd.Series("", index=table.index)).fillna("").astype(str)
    notes = table.get("notes", pd.Series("", index=table.index)).fillna("").astype(str)
    expert_id = table.get("expert_id", pd.Series("", index=table.index)).fillna("").astype(str)
    return (
        measurement_id.str.upper().str.startswith("EXAMPLE")
        | notes.str.upper().str.contains("EXAMPLE", regex=False)
        | expert_id.str.upper().str.contains("EXAMPLE", regex=False)
    )


def value_counts(values: pd.Series) -> dict[str, int]:
    counts = values.fillna("missing").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def standardize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        return text[:-2]
    return text


def default_validation_output_dir(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "validation"


def validation_audit_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Manual Validation Audit Summary",
        f"is_example_audit: {summary.get('is_example_audit')}",
        f"total_manual_rows: {summary.get('total_manual_rows')}",
        f"unique_images_referenced: {summary.get('unique_images_referenced')}",
        f"unique_donors_referenced: {summary.get('unique_donors_referenced')}",
        f"measurement_type_counts: {summary.get('measurement_type_counts')}",
        f"rows_matched_to_analysis_per_image: {summary.get('rows_matched_to_analysis_per_image')}",
        f"unmatched_image_id_rows: {summary.get('unmatched_image_id_rows')}",
        f"donor_id_mismatch_rows: {summary.get('donor_id_mismatch_rows')}",
        f"duplicate_measurement_ids: {summary.get('duplicate_measurement_ids')}",
        f"missing_required_fields: {summary.get('missing_required_fields')}",
        str(summary.get("caution", "")),
    ]
    return "\n".join(lines) + "\n"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
