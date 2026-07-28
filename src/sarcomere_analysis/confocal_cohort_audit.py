from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_dir
from .zdisc_annotation import json_safe


TRIAGE_COLUMNS = [
    "confocal_image_id",
    "filename",
    "pixel_size_x_um",
    "pixel_size_y_um",
    "total_patches",
    "selected_candidate_patches",
    "selected_candidate_fraction",
    "valid_selected_spacing_patches",
    "selected_spacing_valid_fraction",
    "selected_spacing_median_um",
    "selected_spacing_iqr_um",
    "selected_spacing_range_um",
    "selected_median_oop",
    "all_region_median_oop",
    "selected_vs_all_oop_difference",
    "selected_median_coherence",
    "interpretation_class",
]

COHORT_SUMMARY_COLUMNS = [
    "images_processed",
    "errors",
    "calibrated_images",
    "total_patches",
    "selected_candidate_patches",
    "selected_candidate_fraction",
    "valid_selected_spacing_patches",
    "selected_spacing_valid_fraction",
    "median_selected_oop",
    "median_selected_spacing_um",
    "image_count_by_interpretation_class",
]

SPACING_DISTRIBUTION_COLUMNS = [
    "filename",
    "spacing_estimate_um",
    "candidate_striation_region",
    "patch_oop",
    "spacing_confidence",
]


def default_confocal_cohort_audit_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_larger_audit"
    return {
        "root": root,
        "triage": root / "confocal_larger_image_triage.csv",
        "cohort_summary": root / "confocal_larger_cohort_summary.csv",
        "spacing_distribution": root / "confocal_larger_spacing_distribution.csv",
        "summary_json": root / "confocal_larger_audit_summary.json",
        "summary_txt": root / "confocal_larger_audit_summary.txt",
        "review_previews": root / "review_previews",
    }


def audit_confocal_cohort(
    cfg: dict[str, Any],
    pipeline_dir: str | Path,
    pilot_dir: str | Path | None = None,
    output_directory: str | Path | None = None,
    collect_previews: bool = False,
    spacing_min_um: float = 1.5,
    spacing_max_um: float = 2.4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    paths = default_confocal_cohort_audit_paths(cfg, output_directory)
    pipeline_root = Path(pipeline_dir)
    per_image = read_required_csv(pipeline_root / "confocal_pipeline_per_image.csv")
    per_patch = read_required_csv(pipeline_root / "confocal_pipeline_per_patch.csv")
    manifest = read_required_csv(pipeline_root / "confocal_pipeline_manifest.csv")
    pipeline_summary = read_json_if_exists(pipeline_root / "confocal_pipeline_summary.json")
    pilot_summary = read_json_if_exists(Path(pilot_dir) / "confocal_pipeline_summary.json") if pilot_dir else {}
    pilot_image = read_csv_if_exists(Path(pilot_dir) / "confocal_pipeline_per_image.csv") if pilot_dir else pd.DataFrame()

    triage = build_image_triage(per_image, spacing_min_um=spacing_min_um, spacing_max_um=spacing_max_um)
    cohort_summary = build_cohort_summary_table(triage, per_patch, manifest, pipeline_summary)
    spacing_distribution = build_spacing_distribution(per_patch)
    review_selection = select_review_images(triage)
    preview_paths: list[str] = []
    missing_previews: list[dict[str, str]] = []
    if collect_previews:
        preview_paths, missing_previews = collect_review_previews(
            pipeline_root / "previews",
            paths["review_previews"],
            review_selection["review_image_ids"],
        )

    summary = build_audit_summary(
        triage=triage,
        cohort_summary=cohort_summary,
        spacing_distribution=spacing_distribution,
        pipeline_summary=pipeline_summary,
        pilot_summary=pilot_summary,
        pilot_image=pilot_image,
        review_selection=review_selection,
        spacing_min_um=spacing_min_um,
        spacing_max_um=spacing_max_um,
        collect_previews=collect_previews,
        preview_paths=preview_paths,
        missing_previews=missing_previews,
    )
    write_cohort_audit_outputs(triage, cohort_summary, spacing_distribution, summary, paths)
    return triage, cohort_summary, spacing_distribution, summary, paths


def build_image_triage(
    per_image: pd.DataFrame,
    spacing_min_um: float = 1.5,
    spacing_max_um: float = 2.4,
) -> pd.DataFrame:
    working = normalize_ids(per_image)
    rows: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        total = safe_int(row.get("total_patches"))
        candidate_count = safe_int(row.get("selected_candidate_patches"))
        candidate_fraction = safe_float(row.get("selected_candidate_fraction"))
        if not np.isfinite(candidate_fraction):
            candidate_fraction = fraction(candidate_count, total)
        valid_spacing_count = safe_int(row.get("valid_selected_spacing_patches"))
        spacing_fraction = safe_float(row.get("selected_spacing_valid_fraction"))
        if not np.isfinite(spacing_fraction):
            spacing_fraction = fraction(valid_spacing_count, candidate_count)
        processing_status = str(row.get("processing_status", "ok"))
        error_message = str(row.get("error_message", ""))
        interpretation = classify_image(
            candidate_count=candidate_count,
            candidate_fraction=candidate_fraction,
            valid_spacing_count=valid_spacing_count,
            spacing_fraction=spacing_fraction,
            processing_status=processing_status,
            error_message=error_message,
        )
        rows.append(
            {
                "confocal_image_id": str(row.get("confocal_image_id", "")),
                "filename": str(row.get("filename", "")),
                "pixel_size_x_um": safe_float(row.get("pixel_size_x_um")),
                "pixel_size_y_um": safe_float(row.get("pixel_size_y_um")),
                "total_patches": total,
                "selected_candidate_patches": candidate_count,
                "selected_candidate_fraction": candidate_fraction,
                "valid_selected_spacing_patches": valid_spacing_count,
                "selected_spacing_valid_fraction": spacing_fraction,
                "selected_spacing_median_um": safe_float(row.get("selected_spacing_median_um")),
                "selected_spacing_iqr_um": safe_float(row.get("selected_spacing_iqr_um")),
                "selected_spacing_range_um": normalize_string(row.get("selected_spacing_range_um")),
                "selected_median_oop": safe_float(row.get("selected_median_oop")),
                "all_region_median_oop": safe_float(row.get("all_region_median_oop")),
                "selected_vs_all_oop_difference": safe_float(row.get("selected_vs_all_oop_difference")),
                "selected_median_coherence": safe_float(row.get("selected_median_coherence")),
                "interpretation_class": interpretation,
            }
        )
    output = pd.DataFrame(rows, columns=TRIAGE_COLUMNS)
    if not output.empty:
        output["_spacing_median_outside_expected_range"] = outside_spacing_range(
            output["selected_spacing_median_um"], spacing_min_um, spacing_max_um
        )
    return output


def classify_image(
    candidate_count: int,
    candidate_fraction: float,
    valid_spacing_count: int,
    spacing_fraction: float,
    processing_status: str = "ok",
    error_message: str = "",
) -> str:
    if processing_status == "error" or (error_message and error_message.lower() not in {"nan", "none"}):
        return "failed_or_error"
    if candidate_fraction < 0.05:
        return "low_candidate_fraction_review"
    if candidate_fraction > 0.70:
        return "broad_candidate_fraction_review"
    if spacing_fraction >= 0.25 and valid_spacing_count >= 25:
        return "spacing_robust"
    if spacing_fraction >= 0.10 and valid_spacing_count >= 10:
        return "spacing_moderate"
    if candidate_count > 0 and spacing_fraction < 0.10:
        return "oop_only_low_spacing"
    return "low_candidate_fraction_review"


def build_cohort_summary_table(
    triage: pd.DataFrame,
    per_patch: pd.DataFrame,
    manifest: pd.DataFrame,
    pipeline_summary: dict[str, Any],
) -> pd.DataFrame:
    selected_count = int(triage["selected_candidate_patches"].sum()) if not triage.empty else 0
    total_patches = int(triage["total_patches"].sum()) if not triage.empty else int(len(per_patch))
    valid_spacing_count = int(triage["valid_selected_spacing_patches"].sum()) if not triage.empty else 0
    class_counts = triage["interpretation_class"].value_counts().to_dict() if not triage.empty else {}
    row = {
        "images_processed": int(pipeline_summary.get("images_processed", len(triage))),
        "errors": int(pipeline_summary.get("errors", 0)),
        "calibrated_images": int(pipeline_summary.get("calibrated_images", calibrated_count(manifest))),
        "total_patches": total_patches,
        "selected_candidate_patches": selected_count,
        "selected_candidate_fraction": fraction(selected_count, total_patches),
        "valid_selected_spacing_patches": valid_spacing_count,
        "selected_spacing_valid_fraction": fraction(valid_spacing_count, selected_count),
        "median_selected_oop": safe_median(triage.get("selected_median_oop", pd.Series(dtype=float))),
        "median_selected_spacing_um": safe_median(valid_spacing_values(per_patch)),
        "image_count_by_interpretation_class": json.dumps(json_safe(class_counts), sort_keys=True),
    }
    return pd.DataFrame([row], columns=COHORT_SUMMARY_COLUMNS)


def build_spacing_distribution(per_patch: pd.DataFrame) -> pd.DataFrame:
    working = normalize_ids(per_patch)
    spacing_valid = bool_series(working.get("spacing_valid", pd.Series(False, index=working.index)))
    selected = working.loc[spacing_valid].copy()
    rows = pd.DataFrame()
    rows["filename"] = selected.get("filename", pd.Series(dtype=str)).astype(str)
    rows["spacing_estimate_um"] = pd.to_numeric(selected.get("spacing_estimate_um", pd.Series(dtype=float)), errors="coerce")
    rows["candidate_striation_region"] = bool_series(
        selected.get("candidate_striation_region", pd.Series(False, index=selected.index))
    ).to_numpy()
    rows["patch_oop"] = pd.to_numeric(selected.get("patch_oop", pd.Series(dtype=float)), errors="coerce")
    rows["spacing_confidence"] = pd.to_numeric(selected.get("spacing_confidence", pd.Series(dtype=float)), errors="coerce")
    return rows[SPACING_DISTRIBUTION_COLUMNS].copy()


def build_audit_summary(
    triage: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    spacing_distribution: pd.DataFrame,
    pipeline_summary: dict[str, Any],
    pilot_summary: dict[str, Any],
    pilot_image: pd.DataFrame,
    review_selection: dict[str, Any],
    spacing_min_um: float,
    spacing_max_um: float,
    collect_previews: bool,
    preview_paths: list[str],
    missing_previews: list[dict[str, str]],
) -> dict[str, Any]:
    cohort = cohort_summary.iloc[0].to_dict() if not cohort_summary.empty else {}
    larger_fraction = cohort.get("selected_spacing_valid_fraction")
    pilot_fraction = pilot_selected_spacing_fraction(pilot_summary, pilot_image)
    spacing_flags = spacing_median_flags(triage, spacing_min_um, spacing_max_um)
    summary = {
        "mode": "confocal_larger_cohort_audit",
        "pipeline_dir_type": "confocal_larger_pipeline",
        "images_processed": int(cohort.get("images_processed", len(triage))),
        "errors": int(cohort.get("errors", 0)),
        "calibrated_images": int(cohort.get("calibrated_images", 0)),
        "total_patches": int(cohort.get("total_patches", 0)),
        "selected_candidate_patches": int(cohort.get("selected_candidate_patches", 0)),
        "selected_candidate_fraction": safe_float(cohort.get("selected_candidate_fraction")),
        "valid_selected_spacing_patches": int(cohort.get("valid_selected_spacing_patches", 0)),
        "selected_spacing_valid_fraction": safe_float(larger_fraction),
        "median_selected_oop": safe_float(cohort.get("median_selected_oop")),
        "median_selected_spacing_um": safe_float(cohort.get("median_selected_spacing_um")),
        "image_count_by_interpretation_class": (
            json.loads(cohort.get("image_count_by_interpretation_class", "{}"))
            if isinstance(cohort.get("image_count_by_interpretation_class"), str)
            else {}
        ),
        "pilot_comparison": {
            "pilot_available": bool(pilot_summary or not pilot_image.empty),
            "pilot_images_processed": pilot_summary.get("images_processed"),
            "pilot_selected_candidate_patches": pilot_summary.get("selected_candidate_patches"),
            "pilot_valid_selected_spacing_patches": pilot_summary.get("valid_selected_spacing_patches"),
            "pilot_selected_spacing_valid_fraction": pilot_fraction,
            "larger_selected_spacing_valid_fraction": safe_float(larger_fraction),
            "spacing_yield_drop_vs_pilot": (
                float(pilot_fraction - safe_float(larger_fraction))
                if np.isfinite(pilot_fraction) and np.isfinite(safe_float(larger_fraction))
                else None
            ),
        },
        "top_5_spacing_yield_images": records_for_columns(review_selection["top_spacing_yield"], TRIAGE_COLUMNS),
        "bottom_5_spacing_yield_images": records_for_columns(review_selection["bottom_spacing_yield"], TRIAGE_COLUMNS),
        "broad_candidate_fraction_images": records_for_columns(
            triage.loc[pd.to_numeric(triage["selected_candidate_fraction"], errors="coerce") > 0.70], TRIAGE_COLUMNS
        ),
        "low_candidate_fraction_images": records_for_columns(
            triage.loc[pd.to_numeric(triage["selected_candidate_fraction"], errors="coerce") < 0.05], TRIAGE_COLUMNS
        ),
        "spacing_median_outside_expected_range_images": spacing_flags,
        "top_5_selected_oop_images": records_for_columns(
            triage.sort_values("selected_median_oop", ascending=False, na_position="last").head(5), TRIAGE_COLUMNS
        ),
        "bottom_5_selected_oop_images": records_for_columns(
            triage.sort_values("selected_median_oop", ascending=True, na_position="last").head(5), TRIAGE_COLUMNS
        ),
        "spacing_distribution_rows": int(len(spacing_distribution)),
        "collect_previews": bool(collect_previews),
        "review_preview_paths": preview_paths,
        "missing_review_previews": missing_previews,
        "interpretation": [
            "The larger confocal run processed successfully and used per-image calibration.",
            "Selected-region OOP remains stable/promising as an exploratory image-quality/organisation descriptor.",
            "Spacing yield is much lower than in the 11-image pilot, so spacing should not be claimed robust across all larger-cohort images.",
            "Images are triaged into spacing-eligible, OOP-only, or visual-review-needed classes.",
            "Top and bottom spacing-yield images should be visually reviewed before any downstream use.",
            "No thresholds, gates, spacing algorithm, widefield outputs, clinical analyses, or biological claims are changed by this audit.",
        ],
        "source_pipeline_summary_present": bool(pipeline_summary),
    }
    return json_safe(summary)


def select_review_images(triage: pd.DataFrame) -> dict[str, Any]:
    non_error = triage.loc[triage["interpretation_class"] != "failed_or_error"].copy()
    top = non_error.sort_values(
        ["selected_spacing_valid_fraction", "valid_selected_spacing_patches"],
        ascending=False,
        na_position="last",
    ).head(5)
    bottom = non_error.sort_values(
        ["selected_spacing_valid_fraction", "valid_selected_spacing_patches"],
        ascending=True,
        na_position="last",
    ).head(5)
    ids = list(dict.fromkeys(top["confocal_image_id"].astype(str).tolist() + bottom["confocal_image_id"].astype(str).tolist()))
    return {
        "top_spacing_yield": top,
        "bottom_spacing_yield": bottom,
        "review_image_ids": ids,
    }


def collect_review_previews(
    source_preview_dir: Path,
    output_preview_dir: Path,
    image_ids: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    output_preview_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[dict[str, str]] = []
    suffixes = [
        ("selected_candidate_overlay", "selective_analysis_{image_id}_selected_candidate_overlay.png"),
        ("same_grid_oop_heatmap", "same_grid_oop_{image_id}_same_grid_oop_heatmap.png"),
        ("spacing_candidate_overlay", "spacing_audit_{image_id}_confocal_spacing_candidate_overlay.png"),
        ("valid_spacing_overlay", "spacing_audit_{image_id}_confocal_valid_spacing_overlay.png"),
        ("spacing_um_heatmap", "spacing_audit_{image_id}_confocal_spacing_um_heatmap.png"),
    ]
    for image_id in image_ids:
        for label, template in suffixes:
            source = source_preview_dir / template.format(image_id=image_id)
            destination = output_preview_dir / f"{image_id}_{label}.png"
            if source.exists():
                shutil.copy2(source, destination)
                copied.append(str(destination))
            else:
                missing.append({"image_id": image_id, "preview_type": label, "expected_source": str(source)})
    return copied, missing


def write_cohort_audit_outputs(
    triage: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    spacing_distribution: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    triage.drop(columns=[column for column in triage.columns if column.startswith("_")], errors="ignore").to_csv(
        paths["triage"], index=False
    )
    cohort_summary.to_csv(paths["cohort_summary"], index=False)
    spacing_distribution.to_csv(paths["spacing_distribution"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_cohort_audit_summary_text(summary), encoding="utf-8")


def render_cohort_audit_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal larger cohort audit",
        f"images_processed: {summary['images_processed']}",
        f"errors: {summary['errors']}",
        f"calibrated_images: {summary['calibrated_images']}",
        f"total_patches: {summary['total_patches']}",
        f"selected_candidate_patches: {summary['selected_candidate_patches']}",
        f"selected_candidate_fraction: {summary['selected_candidate_fraction']}",
        f"valid_selected_spacing_patches: {summary['valid_selected_spacing_patches']}",
        f"selected_spacing_valid_fraction: {summary['selected_spacing_valid_fraction']}",
        f"median_selected_oop: {summary['median_selected_oop']}",
        f"median_selected_spacing_um: {summary['median_selected_spacing_um']}",
        f"image_count_by_interpretation_class: {summary['image_count_by_interpretation_class']}",
        "",
        "Pilot comparison:",
    ]
    for key, value in summary["pilot_comparison"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Top 5 spacing-yield images:"])
    lines.extend(format_image_records(summary["top_5_spacing_yield_images"]))
    lines.extend(["", "Bottom 5 spacing-yield images:"])
    lines.extend(format_image_records(summary["bottom_5_spacing_yield_images"]))
    lines.extend(["", "Broad candidate fraction images:"])
    lines.extend(format_image_records(summary["broad_candidate_fraction_images"]))
    lines.extend(["", "Low candidate fraction images:"])
    lines.extend(format_image_records(summary["low_candidate_fraction_images"]))
    lines.extend(["", "Spacing median outside expected range images:"])
    lines.extend(format_image_records(summary["spacing_median_outside_expected_range_images"]))
    lines.extend(["", "Interpretation:"])
    lines.extend(f"- {item}" for item in summary["interpretation"])
    return "\n".join(lines) + "\n"


def format_image_records(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return ["- none"]
    lines = []
    for record in records:
        lines.append(
            "- "
            f"{record.get('filename')}: class={record.get('interpretation_class')}, "
            f"candidate_fraction={record.get('selected_candidate_fraction')}, "
            f"spacing_valid_fraction={record.get('selected_spacing_valid_fraction')}, "
            f"valid_spacing={record.get('valid_selected_spacing_patches')}, "
            f"median_spacing={record.get('selected_spacing_median_um')}, "
            f"selected_oop={record.get('selected_median_oop')}"
        )
    return lines


def spacing_median_flags(triage: pd.DataFrame, spacing_min_um: float, spacing_max_um: float) -> list[dict[str, Any]]:
    flags = outside_spacing_range(triage.get("selected_spacing_median_um", pd.Series(dtype=float)), spacing_min_um, spacing_max_um)
    flagged = triage.loc[flags]
    return records_for_columns(flagged, TRIAGE_COLUMNS)


def outside_spacing_range(values: pd.Series, spacing_min_um: float, spacing_max_um: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.notna() & ((numeric < spacing_min_um) | (numeric > spacing_max_um))


def valid_spacing_values(per_patch: pd.DataFrame) -> pd.Series:
    if per_patch.empty:
        return pd.Series(dtype=float)
    spacing_valid = bool_series(per_patch.get("spacing_valid", pd.Series(False, index=per_patch.index)))
    selected = bool_series(per_patch.get("candidate_striation_region", pd.Series(True, index=per_patch.index)))
    return pd.to_numeric(per_patch.loc[spacing_valid & selected, "spacing_estimate_um"], errors="coerce")


def pilot_selected_spacing_fraction(pilot_summary: dict[str, Any], pilot_image: pd.DataFrame) -> float:
    valid = safe_float(pilot_summary.get("valid_selected_spacing_patches")) if pilot_summary else np.nan
    candidates = safe_float(pilot_summary.get("selected_candidate_patches")) if pilot_summary else np.nan
    if np.isfinite(valid) and np.isfinite(candidates) and candidates:
        return float(valid / candidates)
    if not pilot_image.empty:
        total_valid = pd.to_numeric(pilot_image.get("valid_selected_spacing_patches", pd.Series(dtype=float)), errors="coerce").sum()
        total_candidates = pd.to_numeric(pilot_image.get("selected_candidate_patches", pd.Series(dtype=float)), errors="coerce").sum()
        return float(total_valid / total_candidates) if total_candidates else np.nan
    return np.nan


def records_for_columns(table: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if table.empty:
        return []
    keep = [column for column in columns if column in table.columns]
    return json_safe(table[keep].to_dict("records"))


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required confocal cohort audit input is missing: {path}")
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


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_int(value: Any) -> int:
    numeric = safe_float(value)
    return int(numeric) if np.isfinite(numeric) else 0


def fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def safe_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return None if numeric.size == 0 else float(np.median(numeric))


def calibrated_count(manifest: pd.DataFrame) -> int:
    if "pixel_size_available" not in manifest.columns:
        return 0
    return int(bool_series(manifest["pixel_size_available"]).sum())


def normalize_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text
