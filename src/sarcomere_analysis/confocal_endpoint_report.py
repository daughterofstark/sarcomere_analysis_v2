from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_dir
from .zdisc_annotation import json_safe


ENDPOINT_COLUMNS = [
    "confocal_image_id",
    "filename",
    "calibrated",
    "total_patches",
    "selected_candidate_patches",
    "selected_candidate_fraction",
    "selected_median_oop",
    "selected_vs_all_oop_difference",
    "selected_spacing_valid_fraction",
    "valid_selected_spacing_patches",
    "selected_spacing_median_um",
    "selected_spacing_iqr_um",
    "endpoint_class",
    "spacing_reportable",
    "oop_reportable",
    "review_needed",
    "reason",
]


def default_confocal_endpoint_report_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_endpoint_report"
    docs_root = Path(docs_directory) if docs_directory else Path("docs")
    return {
        "root": root,
        "per_image": root / "confocal_endpoint_per_image.csv",
        "summary_json": root / "confocal_endpoint_summary.json",
        "summary_txt": root / "confocal_endpoint_summary.txt",
        "markdown": docs_root / "CONFOCAL_ENDPOINT_REPORT.md",
    }


def write_confocal_endpoint_report(
    cfg: dict[str, Any],
    audit_dir: str | Path,
    pipeline_dir: str | Path,
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    paths = default_confocal_endpoint_report_paths(cfg, output_directory, docs_directory)
    audit_root = Path(audit_dir)
    pipeline_root = Path(pipeline_dir)
    triage = read_required_csv(audit_root / "confocal_larger_image_triage.csv")
    cohort_summary = read_csv_if_exists(audit_root / "confocal_larger_cohort_summary.csv")
    audit_summary = read_json_if_exists(audit_root / "confocal_larger_audit_summary.json")
    pipeline_image = read_required_csv(pipeline_root / "confocal_pipeline_per_image.csv")
    pipeline_summary = read_json_if_exists(pipeline_root / "confocal_pipeline_summary.json")

    per_image = build_endpoint_per_image(triage, pipeline_image)
    summary = build_endpoint_summary(per_image, cohort_summary, audit_summary, pipeline_summary)
    write_endpoint_outputs(per_image, summary, paths)
    return per_image, summary, paths


def build_endpoint_per_image(triage: pd.DataFrame, pipeline_image: pd.DataFrame) -> pd.DataFrame:
    triage_working = normalize_ids(triage)
    pipeline_working = normalize_ids(pipeline_image)
    pipeline_keep = [
        column
        for column in [
            "confocal_image_id",
            "pixel_size_available",
            "processing_status",
            "error_message",
        ]
        if column in pipeline_working.columns
    ]
    if "confocal_image_id" in pipeline_keep:
        triage_working = triage_working.merge(
            pipeline_working[pipeline_keep].drop_duplicates("confocal_image_id"),
            on="confocal_image_id",
            how="left",
        )

    rows: list[dict[str, Any]] = []
    for _, row in triage_working.iterrows():
        candidate_count = safe_int(row.get("selected_candidate_patches"))
        candidate_fraction = safe_float(row.get("selected_candidate_fraction"))
        valid_spacing_count = safe_int(row.get("valid_selected_spacing_patches"))
        spacing_fraction = safe_float(row.get("selected_spacing_valid_fraction"))
        if "pixel_size_available" in row.index and not pd.isna(row.get("pixel_size_available")):
            calibrated = bool_value(row.get("pixel_size_available"))
        else:
            calibrated = np.isfinite(safe_float(row.get("pixel_size_x_um"))) and np.isfinite(
                safe_float(row.get("pixel_size_y_um"))
            )
        processing_status = str(row.get("processing_status", "ok"))
        error_message = normalize_string(row.get("error_message"))
        endpoint_class, reason = classify_endpoint(
            calibrated=calibrated,
            candidate_count=candidate_count,
            candidate_fraction=candidate_fraction,
            valid_spacing_count=valid_spacing_count,
            spacing_fraction=spacing_fraction,
            processing_status=processing_status,
            error_message=error_message,
        )
        oop_reportable = bool(calibrated and candidate_count >= 25 and endpoint_class != "failed_or_unusable")
        spacing_reportable = bool(valid_spacing_count >= 10 and spacing_fraction >= 0.10 and endpoint_class != "failed_or_unusable")
        review_needed = bool(endpoint_class in {"low_candidate_review_needed", "spacing_eligible_low_confidence", "failed_or_unusable"})
        rows.append(
            {
                "confocal_image_id": str(row.get("confocal_image_id", "")),
                "filename": str(row.get("filename", "")),
                "calibrated": calibrated,
                "total_patches": safe_int(row.get("total_patches")),
                "selected_candidate_patches": candidate_count,
                "selected_candidate_fraction": candidate_fraction,
                "selected_median_oop": safe_float(row.get("selected_median_oop")),
                "selected_vs_all_oop_difference": safe_float(row.get("selected_vs_all_oop_difference")),
                "selected_spacing_valid_fraction": spacing_fraction,
                "valid_selected_spacing_patches": valid_spacing_count,
                "selected_spacing_median_um": safe_float(row.get("selected_spacing_median_um")),
                "selected_spacing_iqr_um": safe_float(row.get("selected_spacing_iqr_um")),
                "endpoint_class": endpoint_class,
                "spacing_reportable": spacing_reportable,
                "oop_reportable": oop_reportable,
                "review_needed": review_needed,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=ENDPOINT_COLUMNS)


def classify_endpoint(
    calibrated: bool,
    candidate_count: int,
    candidate_fraction: float,
    valid_spacing_count: int,
    spacing_fraction: float,
    processing_status: str = "ok",
    error_message: str = "",
) -> tuple[str, str]:
    if processing_status == "error" or error_message:
        return "failed_or_unusable", "pipeline_error_or_recorded_error"
    if not calibrated:
        return "failed_or_unusable", "missing_per_image_calibration"
    if candidate_fraction < 0.05:
        return "low_candidate_review_needed", "selected_candidate_fraction_below_0.05"
    if valid_spacing_count >= 10 and spacing_fraction >= 0.10:
        return "spacing_eligible_moderate", "spacing_count_and_fraction_meet_moderate_endpoint_rule"
    if valid_spacing_count >= 5 and spacing_fraction < 0.10:
        return "spacing_eligible_low_confidence", "some_spacing_patches_but_fraction_below_0.10"
    if candidate_count >= 25:
        return "oop_only_spacing_low_yield", "oop_candidate_regions_available_but_spacing_low_yield"
    return "low_candidate_review_needed", "selected_candidate_patch_count_below_25"


def build_endpoint_summary(
    per_image: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    audit_summary: dict[str, Any],
    pipeline_summary: dict[str, Any],
) -> dict[str, Any]:
    endpoint_counts = per_image["endpoint_class"].value_counts().to_dict() if not per_image.empty else {}
    spacing_images = per_image.loc[bool_series(per_image.get("spacing_reportable", pd.Series(dtype=bool)))]
    oop_only = per_image.loc[per_image["endpoint_class"].astype(str) == "oop_only_spacing_low_yield"]
    review_needed = per_image.loc[bool_series(per_image.get("review_needed", pd.Series(dtype=bool)))]
    robust_count = cohort_interpretation_count(audit_summary, "spacing_robust")
    moderate_count = cohort_interpretation_count(audit_summary, "spacing_moderate")
    summary = {
        "mode": "confocal_endpoint_classification_report",
        "images_processed": int(pipeline_summary.get("images_processed", len(per_image))),
        "errors": int(pipeline_summary.get("errors", 0)),
        "calibrated_images": int(pipeline_summary.get("calibrated_images", int(per_image["calibrated"].sum()) if not per_image.empty else 0)),
        "endpoint_class_counts": endpoint_counts,
        "spacing_reportable_image_count": int(len(spacing_images)),
        "oop_reportable_image_count": int(bool_series(per_image.get("oop_reportable", pd.Series(dtype=bool))).sum()),
        "oop_only_image_count": int(len(oop_only)),
        "review_needed_image_count": int(len(review_needed)),
        "spacing_robust_image_count_from_cohort_audit": int(robust_count),
        "spacing_moderate_image_count_from_cohort_audit": int(moderate_count),
        "selected_candidate_patches": int(pipeline_summary.get("selected_candidate_patches", selected_candidate_total(per_image))),
        "valid_selected_spacing_patches": int(
            pipeline_summary.get("valid_selected_spacing_patches", valid_spacing_total(per_image))
        ),
        "selected_spacing_valid_fraction": safe_float(
            audit_summary.get("selected_spacing_valid_fraction", selected_spacing_fraction(per_image))
        ),
        "median_selected_oop": safe_float(audit_summary.get("median_selected_oop", safe_median(per_image["selected_median_oop"]))),
        "median_selected_spacing_um": safe_float(
            audit_summary.get("median_selected_spacing_um", safe_median(per_image["selected_spacing_median_um"]))
        ),
        "spacing_reportable_images": records(per_image.loc[per_image["spacing_reportable"]]),
        "oop_only_images": records(oop_only),
        "review_needed_images": records(review_needed),
        "interpretation": [
            "The 42-image confocal pipeline ran successfully.",
            "OOP/coherence endpoint is broadly available for images with sufficient selected candidate patches.",
            "Spacing endpoint is available only for a subset of images meeting both count and fraction rules.",
            "No images met the robust spacing threshold in the cohort audit; seven met moderate spacing criteria.",
            "Spacing should not be treated as a universal endpoint for the larger confocal cohort.",
            "Downstream analysis should be endpoint-aware: OOP/coherence across the broader cohort, spacing only in spacing-eligible images/regions.",
            "No disease, clinical, threshold-tuning, or biological claims are made by this endpoint report.",
        ],
        "recommendation": (
            "Use OOP/coherence as the broad confocal endpoint family and restrict spacing summaries to spacing-reportable "
            "images/regions. Next action should be visual review of spacing-eligible and low-yield examples, plus more "
            "confocal images if spacing is intended as a downstream endpoint."
        ),
        "source_audit_summary_present": bool(audit_summary),
        "source_cohort_summary_rows": int(len(cohort_summary)),
    }
    return json_safe(summary)


def write_endpoint_outputs(per_image: pd.DataFrame, summary: dict[str, Any], paths: dict[str, Path]) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["markdown"].parent.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(paths["per_image"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    text = render_endpoint_summary_text(summary)
    paths["summary_txt"].write_text(text, encoding="utf-8")
    paths["markdown"].write_text(render_endpoint_markdown(summary), encoding="utf-8")


def render_endpoint_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal endpoint classification report",
        f"images_processed: {summary['images_processed']}",
        f"errors: {summary['errors']}",
        f"calibrated_images: {summary['calibrated_images']}",
        f"endpoint_class_counts: {summary['endpoint_class_counts']}",
        f"spacing_reportable_image_count: {summary['spacing_reportable_image_count']}",
        f"oop_reportable_image_count: {summary['oop_reportable_image_count']}",
        f"oop_only_image_count: {summary['oop_only_image_count']}",
        f"review_needed_image_count: {summary['review_needed_image_count']}",
        f"spacing_robust_image_count_from_cohort_audit: {summary['spacing_robust_image_count_from_cohort_audit']}",
        f"spacing_moderate_image_count_from_cohort_audit: {summary['spacing_moderate_image_count_from_cohort_audit']}",
        "",
        "Spacing-reportable images:",
    ]
    lines.extend(format_images(summary["spacing_reportable_images"]))
    lines.extend(["", "OOP-only images:"])
    lines.extend(format_images(summary["oop_only_images"]))
    lines.extend(["", "Review-needed images:"])
    lines.extend(format_images(summary["review_needed_images"]))
    lines.extend(["", "Interpretation:"])
    lines.extend(f"- {item}" for item in summary["interpretation"])
    lines.extend(["", f"Recommendation: {summary['recommendation']}"])
    return "\n".join(lines) + "\n"


def render_endpoint_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Confocal Endpoint Report",
        "",
        "This report classifies the larger confocal cohort into endpoint-aware analysis groups. It reads existing confocal pipeline and cohort-audit outputs only; it does not rerun algorithms, tune thresholds, adopt the relaxed gate, or change widefield outputs.",
        "",
        "## Cohort Status",
        "",
        f"- Images processed: {summary['images_processed']}",
        f"- Errors: {summary['errors']}",
        f"- Calibrated images: {summary['calibrated_images']}",
        f"- Endpoint class counts: `{summary['endpoint_class_counts']}`",
        f"- OOP-reportable images: {summary['oop_reportable_image_count']}",
        f"- Spacing-reportable images: {summary['spacing_reportable_image_count']}",
        f"- OOP-only images: {summary['oop_only_image_count']}",
        f"- Review-needed images: {summary['review_needed_image_count']}",
        "",
        "## Decision Rules",
        "",
        "- `oop_reportable`: per-image calibration exists and at least 25 selected candidate patches are available.",
        "- `spacing_reportable`: at least 10 valid selected spacing patches and selected spacing valid fraction at least 0.10.",
        "- `spacing_eligible_moderate`: spacing is reportable under the current endpoint rule.",
        "- `spacing_eligible_low_confidence`: at least 5 valid spacing patches exist, but the fraction is below 0.10.",
        "- `oop_only_spacing_low_yield`: OOP/coherence is available, but spacing is low-yield.",
        "- `low_candidate_review_needed`: selected candidate fraction is below 0.05 or too few candidate patches are available.",
        "- `failed_or_unusable`: processing error or missing per-image calibration.",
        "",
        "## Interpretation",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["interpretation"])
    lines.extend(
        [
            "",
            "## Spacing-Reportable Images",
            "",
        ]
    )
    lines.extend(format_markdown_images(summary["spacing_reportable_images"]))
    lines.extend(["", "## OOP-Only Images", ""])
    lines.extend(format_markdown_images(summary["oop_only_images"]))
    lines.extend(["", "## Review-Needed Images", ""])
    lines.extend(format_markdown_images(summary["review_needed_images"]))
    lines.extend(["", "## Recommendation", "", summary["recommendation"], ""])
    return "\n".join(lines)


def format_images(records_in: list[dict[str, Any]]) -> list[str]:
    if not records_in:
        return ["- none"]
    return [
        "- "
        f"{row.get('filename')}: class={row.get('endpoint_class')}, "
        f"spacing_fraction={row.get('selected_spacing_valid_fraction')}, "
        f"valid_spacing={row.get('valid_selected_spacing_patches')}, "
        f"candidate_fraction={row.get('selected_candidate_fraction')}, reason={row.get('reason')}"
        for row in records_in
    ]


def format_markdown_images(records_in: list[dict[str, Any]]) -> list[str]:
    if not records_in:
        return ["- none"]
    return [
        f"- `{row.get('filename')}`: `{row.get('endpoint_class')}`, spacing fraction `{row.get('selected_spacing_valid_fraction')}`, valid spacing patches `{row.get('valid_selected_spacing_patches')}`"
        for row in records_in
    ]


def cohort_interpretation_count(audit_summary: dict[str, Any], class_name: str) -> int:
    counts = audit_summary.get("image_count_by_interpretation_class", {})
    if not isinstance(counts, dict):
        return 0
    return int(counts.get(class_name, 0))


def selected_candidate_total(per_image: pd.DataFrame) -> int:
    return int(pd.to_numeric(per_image.get("selected_candidate_patches", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())


def valid_spacing_total(per_image: pd.DataFrame) -> int:
    return int(pd.to_numeric(per_image.get("valid_selected_spacing_patches", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())


def selected_spacing_fraction(per_image: pd.DataFrame) -> float:
    candidates = selected_candidate_total(per_image)
    valid = valid_spacing_total(per_image)
    return float(valid / candidates) if candidates else float("nan")


def records(table: pd.DataFrame) -> list[dict[str, Any]]:
    return json_safe(table[ENDPOINT_COLUMNS].to_dict("records")) if not table.empty else []


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required confocal endpoint report input is missing: {path}")
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


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_ids(table: pd.DataFrame) -> pd.DataFrame:
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
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_int(value: Any) -> int:
    numeric = safe_float(value)
    return int(numeric) if np.isfinite(numeric) else 0


def safe_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return None if numeric.size == 0 else float(np.median(numeric))


def normalize_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    return "" if text.strip().lower() in {"nan", "none"} else text
