from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_dir
from .zdisc_annotation import json_safe


ANALYSIS_COLUMNS = [
    "confocal_image_id",
    "filename",
    "calibrated",
    "pixel_size_x_um",
    "pixel_size_y_um",
    "endpoint_class",
    "oop_reportable",
    "spacing_reportable",
    "review_needed",
    "selected_candidate_fraction",
    "selected_median_oop",
    "all_region_median_oop",
    "selected_vs_all_oop_difference",
    "selected_median_coherence",
    "valid_selected_spacing_patches",
    "selected_spacing_valid_fraction",
    "selected_spacing_median_um",
    "selected_spacing_iqr_um",
    "selected_spacing_range_um",
    "spacing_value_allowed_for_downstream",
    "oop_value_allowed_for_downstream",
    "spacing_downstream_warning",
    "manual_review_status",
    "manual_review_verdict",
    "notes",
]

REVIEW_TEMPLATE_COLUMNS = [
    "filename",
    "review_group",
    "endpoint_class",
    "spacing_reportable",
    "oop_reportable",
    "selected_candidate_fraction",
    "valid_selected_spacing_patches",
    "selected_spacing_valid_fraction",
    "selected_spacing_median_um",
    "automated_reason",
    "reviewer_name",
    "selected_regions_valid",
    "valid_spacing_patches_valid",
    "image_suitable_for_spacing",
    "image_suitable_for_oop",
    "reviewer_confidence",
    "reviewer_notes",
]

ALLOWED_VALUE_GUIDANCE = {
    "selected_regions_valid": "yes / partial / no / unclear",
    "valid_spacing_patches_valid": "yes / partial / no / unclear / not_applicable",
    "image_suitable_for_spacing": "yes / no / unclear",
    "image_suitable_for_oop": "yes / no / unclear",
    "reviewer_confidence": "1 / 2 / 3 / 4 / 5",
}


def default_confocal_analysis_dataset_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_analysis_dataset"
    docs_root = Path(docs_directory) if docs_directory else Path("docs")
    return {
        "root": root,
        "per_image": root / "confocal_analysis_per_image.csv",
        "review_template": root / "confocal_manual_review_template.csv",
        "summary_json": root / "confocal_analysis_dataset_summary.json",
        "summary_txt": root / "confocal_analysis_dataset_summary.txt",
        "markdown": docs_root / "CONFOCAL_ANALYSIS_DATASET.md",
    }


def build_confocal_analysis_dataset(
    cfg: dict[str, Any],
    endpoint_dir: str | Path,
    pipeline_dir: str | Path,
    audit_dir: str | Path,
    freeze_dir: str | Path,
    review_pack_dir: str | Path,
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    paths = default_confocal_analysis_dataset_paths(cfg, output_directory, docs_directory)
    endpoint = read_required_csv(Path(endpoint_dir) / "confocal_endpoint_per_image.csv")
    pipeline = read_required_csv(Path(pipeline_dir) / "confocal_pipeline_per_image.csv")
    triage = read_required_csv(Path(audit_dir) / "confocal_larger_image_triage.csv")
    freeze = read_required_json(Path(freeze_dir) / "confocal_freeze_summary.json")
    review_index = read_csv_if_exists(Path(review_pack_dir) / "confocal_endpoint_review_index.csv")

    per_image = build_analysis_per_image(endpoint, pipeline, triage, freeze, review_index)
    review_template = build_manual_review_template(per_image, review_index)
    summary = build_analysis_dataset_summary(per_image, review_template, freeze, review_index)
    write_analysis_dataset_outputs(per_image, review_template, summary, paths)
    return per_image, review_template, summary, paths


def build_analysis_per_image(
    endpoint: pd.DataFrame,
    pipeline: pd.DataFrame,
    triage: pd.DataFrame,
    freeze: dict[str, Any],
    review_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    endpoint = normalize_ids(endpoint)
    pipeline = normalize_ids(pipeline)
    triage = normalize_ids(triage)
    review_index = normalize_ids(review_index) if review_index is not None else pd.DataFrame()
    verdicts = manual_verdicts_from_freeze(freeze)
    review_groups = review_groups_by_filename(review_index)
    rows: list[dict[str, Any]] = []
    for _, row in endpoint.iterrows():
        image_id = str(row.get("confocal_image_id", ""))
        filename = str(row.get("filename", ""))
        pipe = first_image_row(pipeline, image_id, filename)
        triage_row = first_image_row(triage, image_id, filename)
        verdict = verdicts.get(filename, {})
        spacing_reportable = bool_value(row.get("spacing_reportable", False))
        oop_reportable = bool_value(row.get("oop_reportable", False))
        review_status = verdict.get("review_status") or (
            "in_endpoint_review_pack_unreviewed" if filename in review_groups else "not_in_endpoint_review_pack"
        )
        notes = combined_notes(row, pipe, triage_row, verdict, review_groups.get(filename))
        rows.append(
            {
                "confocal_image_id": image_id,
                "filename": filename,
                "calibrated": bool_value(row.get("calibrated", pipe.get("pixel_size_available", False))),
                "pixel_size_x_um": safe_float(pipe.get("pixel_size_x_um", triage_row.get("pixel_size_x_um"))),
                "pixel_size_y_um": safe_float(pipe.get("pixel_size_y_um", triage_row.get("pixel_size_y_um"))),
                "endpoint_class": str(row.get("endpoint_class", "")),
                "oop_reportable": oop_reportable,
                "spacing_reportable": spacing_reportable,
                "review_needed": bool_value(row.get("review_needed", False)),
                "selected_candidate_fraction": safe_float(row.get("selected_candidate_fraction")),
                "selected_median_oop": safe_float(row.get("selected_median_oop")),
                "all_region_median_oop": safe_float(pipe.get("all_region_median_oop", triage_row.get("all_region_median_oop"))),
                "selected_vs_all_oop_difference": safe_float(row.get("selected_vs_all_oop_difference")),
                "selected_median_coherence": safe_float(
                    pipe.get("selected_median_coherence", triage_row.get("selected_median_coherence"))
                ),
                "valid_selected_spacing_patches": safe_int(row.get("valid_selected_spacing_patches")),
                "selected_spacing_valid_fraction": safe_float(row.get("selected_spacing_valid_fraction")),
                "selected_spacing_median_um": safe_float(row.get("selected_spacing_median_um")),
                "selected_spacing_iqr_um": safe_float(row.get("selected_spacing_iqr_um")),
                "selected_spacing_range_um": normalize_string(
                    pipe.get("selected_spacing_range_um", triage_row.get("selected_spacing_range_um"))
                ),
                "spacing_value_allowed_for_downstream": spacing_reportable,
                "oop_value_allowed_for_downstream": oop_reportable,
                "spacing_downstream_warning": (
                    "selected_region_spacing_only" if spacing_reportable else "not_reportable_endpoint_low_yield"
                ),
                "manual_review_status": review_status,
                "manual_review_verdict": verdict.get("verdict", ""),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=ANALYSIS_COLUMNS)


def build_manual_review_template(per_image: pd.DataFrame, review_index: pd.DataFrame | None = None) -> pd.DataFrame:
    review_index = normalize_ids(review_index) if review_index is not None else pd.DataFrame()
    if not review_index.empty and "filename" in review_index.columns:
        selected = review_index[["filename", "review_group"]].copy(deep=True)
        selected["filename"] = selected["filename"].astype(str)
        selected = selected.drop_duplicates("filename", keep="first")
        source = per_image.merge(selected, on="filename", how="inner")
    else:
        source = per_image.loc[
            bool_series(per_image["spacing_reportable"]) | bool_series(per_image["review_needed"])
        ].copy(deep=True)
        source["review_group"] = source["endpoint_class"].map(default_review_group_for_endpoint)
    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        rows.append(
            {
                "filename": str(row.get("filename", "")),
                "review_group": str(row.get("review_group", "")),
                "endpoint_class": str(row.get("endpoint_class", "")),
                "spacing_reportable": bool_value(row.get("spacing_reportable", False)),
                "oop_reportable": bool_value(row.get("oop_reportable", False)),
                "selected_candidate_fraction": safe_float(row.get("selected_candidate_fraction")),
                "valid_selected_spacing_patches": safe_int(row.get("valid_selected_spacing_patches")),
                "selected_spacing_valid_fraction": safe_float(row.get("selected_spacing_valid_fraction")),
                "selected_spacing_median_um": safe_float(row.get("selected_spacing_median_um")),
                "automated_reason": str(row.get("reason", row.get("notes", ""))),
                "reviewer_name": "",
                "selected_regions_valid": "",
                "valid_spacing_patches_valid": "",
                "image_suitable_for_spacing": "",
                "image_suitable_for_oop": "",
                "reviewer_confidence": "",
                "reviewer_notes": "",
            }
        )
    return pd.DataFrame(rows, columns=REVIEW_TEMPLATE_COLUMNS)


def build_analysis_dataset_summary(
    per_image: pd.DataFrame,
    review_template: pd.DataFrame,
    freeze: dict[str, Any],
    review_index: pd.DataFrame | None = None,
) -> dict[str, Any]:
    spacing_allowed = per_image.loc[bool_series(per_image["spacing_value_allowed_for_downstream"])]
    oop_allowed = per_image.loc[bool_series(per_image["oop_value_allowed_for_downstream"])]
    review_needed = per_image.loc[bool_series(per_image["review_needed"])]
    freeze_decisions = freeze.get("final_frozen_interpretation", [])
    return json_safe(
        {
            "mode": "confocal_analysis_ready_dataset",
            "images_total": int(len(per_image)),
            "oop_allowed_count": int(len(oop_allowed)),
            "spacing_allowed_count": int(len(spacing_allowed)),
            "review_template_rows": int(len(review_template)),
            "spacing_reportable_image_list": spacing_allowed["filename"].astype(str).tolist(),
            "images_requiring_review": review_needed["filename"].astype(str).tolist(),
            "freeze_report_decision_string": "; ".join(str(item) for item in freeze_decisions),
            "manual_review_template_allowed_values": ALLOWED_VALUE_GUIDANCE,
            "review_index_present": bool(review_index is not None and not review_index.empty),
            "caveats": [
                "This is a downstream-safe data packaging layer only.",
                "Endpoint flags must be respected in downstream work.",
                "OOP/coherence is broad; spacing is subset/selected-region only.",
                "Spacing numeric values are retained even when not reportable, with warning flags controlling interpretation.",
                "No disease, clinical, statistical, ML, or biological inference is performed here.",
                "The manual review template is for endpoint QC, not ML training.",
            ],
        }
    )


def write_analysis_dataset_outputs(
    per_image: pd.DataFrame,
    review_template: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["markdown"].parent.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(paths["per_image"], index=False)
    review_template.to_csv(paths["review_template"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_summary_text(summary), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(summary), encoding="utf-8")


def render_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal analysis-ready dataset",
        f"images_total: {summary['images_total']}",
        f"oop_allowed_count: {summary['oop_allowed_count']}",
        f"spacing_allowed_count: {summary['spacing_allowed_count']}",
        f"review_template_rows: {summary['review_template_rows']}",
        f"spacing_reportable_image_list: {summary['spacing_reportable_image_list']}",
        f"images_requiring_review: {summary['images_requiring_review']}",
        "",
        "Caveats:",
    ]
    lines.extend(f"- {item}" for item in summary["caveats"])
    return "\n".join(lines) + "\n"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Confocal Analysis Dataset",
        "",
        "This is the downstream-safe confocal table and manual endpoint-QC template. It packages existing frozen outputs only; it does not rerun algorithms, tune thresholds, adopt the relaxed gate, or change widefield outputs.",
        "",
        "## Outputs",
        "",
        "- `results/confocal_analysis_dataset/confocal_analysis_per_image.csv`",
        "- `results/confocal_analysis_dataset/confocal_manual_review_template.csv`",
        "- `results/confocal_analysis_dataset/confocal_analysis_dataset_summary.json`",
        "- `results/confocal_analysis_dataset/confocal_analysis_dataset_summary.txt`",
        "",
        "## Summary",
        "",
        f"- Images total: {summary['images_total']}",
        f"- OOP allowed count: {summary['oop_allowed_count']}",
        f"- Spacing allowed count: {summary['spacing_allowed_count']}",
        f"- Review template rows: {summary['review_template_rows']}",
        "",
        "Spacing-reportable images:",
    ]
    lines.extend(f"- `{name}`" for name in summary["spacing_reportable_image_list"])
    lines.extend(
        [
            "",
            "## Rules",
            "",
            "- `oop_value_allowed_for_downstream` follows `oop_reportable`.",
            "- `spacing_value_allowed_for_downstream` follows `spacing_reportable`.",
            "- Non-reportable spacing values are retained but marked with `spacing_downstream_warning = not_reportable_endpoint_low_yield`.",
            "- Reportable spacing values remain selected-region/subset endpoints and are marked `selected_region_spacing_only`.",
            "",
            "## Manual Review Template",
            "",
            "The manual review template is for endpoint QC, not ML training. Allowed response values:",
        ]
    )
    lines.extend(f"- `{key}`: {value}" for key, value in ALLOWED_VALUE_GUIDANCE.items())
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {item}" for item in summary["caveats"])
    lines.append("")
    return "\n".join(lines)


def manual_verdicts_from_freeze(freeze: dict[str, Any]) -> dict[str, dict[str, str]]:
    verdicts = {}
    for item in (freeze.get("manual_visual_spot_check") or {}).get("verdicts", []):
        filename = str(item.get("filename", ""))
        if filename:
            verdicts[filename] = {
                "review_status": str(item.get("review_status", "")),
                "verdict": str(item.get("verdict", "")),
                "caveat": str(item.get("caveat", "")),
            }
    return verdicts


def review_groups_by_filename(review_index: pd.DataFrame) -> dict[str, str]:
    if review_index.empty or "filename" not in review_index.columns or "review_group" not in review_index.columns:
        return {}
    return dict(zip(review_index["filename"].astype(str), review_index["review_group"].astype(str)))


def first_image_row(table: pd.DataFrame, image_id: str, filename: str) -> dict[str, Any]:
    if table.empty:
        return {}
    if "confocal_image_id" in table.columns:
        matches = table.loc[table["confocal_image_id"].astype(str) == str(image_id)]
        if not matches.empty:
            return matches.iloc[0].to_dict()
    if "filename" in table.columns:
        matches = table.loc[table["filename"].astype(str) == str(filename)]
        if not matches.empty:
            return matches.iloc[0].to_dict()
    return {}


def combined_notes(
    endpoint_row: pd.Series,
    pipeline_row: dict[str, Any],
    triage_row: dict[str, Any],
    verdict: dict[str, str],
    review_group: str | None,
) -> str:
    pieces = []
    reason = normalize_string(endpoint_row.get("reason"))
    if reason:
        pieces.append(f"automated_reason={reason}")
    flag = normalize_string(pipeline_row.get("interpretation_flag", triage_row.get("interpretation_class")))
    if flag:
        pieces.append(f"pipeline_or_audit_flag={flag}")
    if review_group:
        pieces.append(f"review_group={review_group}")
    caveat = normalize_string(verdict.get("caveat"))
    if caveat:
        pieces.append(f"manual_caveat={caveat}")
    return "; ".join(pieces)


def default_review_group_for_endpoint(endpoint_class: str) -> str:
    if endpoint_class == "spacing_eligible_moderate":
        return "spacing_reportable"
    if endpoint_class == "low_candidate_review_needed":
        return "low_candidate_review"
    return "oop_only_examples"


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required confocal analysis dataset input is missing: {path}")
    return read_csv(path)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return read_csv(path) if path.exists() else pd.DataFrame()


def read_csv(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    dtype = {
        column: str
        for column in ["confocal_image_id", "filename", "patch_id", "source_path"]
        if column in header.columns
    }
    return pd.read_csv(path, dtype=dtype)


def read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required confocal analysis dataset input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_ids(table: pd.DataFrame | None) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    output = table.copy(deep=True)
    for column in ["confocal_image_id", "filename", "patch_id"]:
        if column in output.columns:
            output[column] = output[column].astype(str)
    return output


def bool_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        if values.dtype == object:
            return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y"})
        return values.fillna(False).astype(bool)
    return pd.Series(values).fillna(False).astype(bool)


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


def safe_float(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else np.nan


def safe_int(value: Any) -> int:
    numeric = safe_float(value)
    return int(numeric) if np.isfinite(numeric) else 0


def normalize_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    return "" if text.strip().lower() in {"nan", "none"} else text
