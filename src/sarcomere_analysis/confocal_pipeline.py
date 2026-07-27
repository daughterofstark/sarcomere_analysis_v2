from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .confocal_intake import run_confocal_baseline_audit
from .confocal_metadata import audit_confocal_metadata
from .confocal_same_grid_oop import run_confocal_same_grid_oop
from .confocal_selective_analysis import run_confocal_selective_analysis
from .confocal_spacing_audit import run_confocal_spacing_audit
from .confocal_striation_mask import run_confocal_striation_mask_audit
from .confocal_striation_sensitivity import run_confocal_striation_sensitivity
from .config import output_dir
from .zdisc_annotation import json_safe


PRIMARY_CONFOCAL_GATE = "moderate"
RELAXED_GATE_STATUS = "moderate_relaxed_combined_sensitivity_only_not_primary"

CONFOCAL_PIPELINE_IMAGE_COLUMNS = [
    "confocal_image_id",
    "filename",
    "image_shape",
    "pixel_size_x_um",
    "pixel_size_y_um",
    "pixel_size_available",
    "total_patches",
    "selected_candidate_patches",
    "selected_candidate_fraction",
    "selected_median_oop",
    "all_region_median_oop",
    "selected_vs_all_oop_difference",
    "selected_median_coherence",
    "all_region_median_coherence",
    "valid_selected_spacing_patches",
    "selected_spacing_valid_fraction",
    "selected_spacing_median_um",
    "selected_spacing_iqr_um",
    "selected_spacing_range_um",
    "interpretation_flag",
    "processing_status",
    "error_message",
]

CONFOCAL_PIPELINE_PATCH_COLUMNS = [
    "confocal_image_id",
    "filename",
    "patch_id",
    "y0",
    "x0",
    "y1",
    "x1",
    "center_y",
    "center_x",
    "candidate_striation_region",
    "pixel_size_x_um",
    "pixel_size_y_um",
    "patch_oop",
    "patch_mean_orientation_deg",
    "patch_orientation_weight_sum",
    "patch_orientation_valid_pixels",
    "patch_orientation_coherence_mean",
    "gradient_energy",
    "intensity_std",
    "contrast",
    "spacing_estimate_px",
    "spacing_estimate_um",
    "spacing_valid",
    "spacing_failure_reason",
    "processing_status",
    "error_message",
]


def default_confocal_pipeline_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_pipeline"
    return {
        "root": root,
        "intermediate": root / "_intermediate",
        "manifest": root / "confocal_pipeline_manifest.csv",
        "per_patch": root / "confocal_pipeline_per_patch.csv",
        "per_image": root / "confocal_pipeline_per_image.csv",
        "summary_json": root / "confocal_pipeline_summary.json",
        "summary_txt": root / "confocal_pipeline_summary.txt",
        "previews": root / "previews",
    }


def run_confocal_pipeline(
    cfg: dict[str, Any],
    confocal_root: str | Path,
    output_directory: str | Path | None = None,
    write_previews: bool = False,
    spacing_min_um: float = 1.5,
    spacing_max_um: float = 2.4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    paths = default_confocal_pipeline_paths(cfg, output_directory)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["intermediate"].mkdir(parents=True, exist_ok=True)
    run_cfg = pipeline_cfg(cfg, paths["intermediate"])

    baseline_manifest, baseline_image, baseline_patch, baseline_summary, baseline_paths = run_confocal_baseline_audit(
        run_cfg,
        confocal_root=confocal_root,
        output_directory=paths["intermediate"] / "confocal_baseline",
        write_previews=False,
    )
    calibration, metadata_summary, metadata_paths = audit_confocal_metadata(
        run_cfg,
        confocal_manifest=baseline_paths["manifest"],
        output_directory=paths["intermediate"] / "confocal_metadata",
        write_manual_template=False,
    )
    mask_patch, mask_image, mask_summary, mask_paths = run_confocal_striation_mask_audit(
        run_cfg,
        confocal_manifest=baseline_paths["manifest"],
        output_directory=paths["intermediate"] / "confocal_striation_mask",
        write_previews=write_previews,
    )
    sensitivity_variants, sensitivity_image, sensitivity_summary, sensitivity_paths = run_confocal_striation_sensitivity(
        run_cfg,
        patch_table=mask_paths["per_patch"],
        image_table=mask_paths["per_image"],
        output_directory=paths["intermediate"] / "confocal_striation_sensitivity",
        write_previews=False,
    )
    selective_patch, selective_image, selective_summary, selective_paths = run_confocal_selective_analysis(
        run_cfg,
        selected_variant=PRIMARY_CONFOCAL_GATE,
        patch_table=mask_paths["per_patch"],
        baseline_patch_table=baseline_paths["per_patch"],
        sensitivity_variants=sensitivity_paths["variants"],
        sensitivity_per_image=sensitivity_paths["per_image"],
        output_directory=paths["intermediate"] / "confocal_selective_analysis",
        write_previews=write_previews,
    )
    same_patch, same_image, same_summary, same_paths = run_confocal_same_grid_oop(
        run_cfg,
        patch_table=mask_paths["per_patch"],
        manifest=baseline_paths["manifest"],
        output_directory=paths["intermediate"] / "confocal_same_grid_oop",
        write_previews=write_previews,
    )
    spacing_patch, spacing_image, spacing_summary, spacing_paths = run_confocal_spacing_audit(
        run_cfg,
        calibration_table=metadata_paths["calibration"],
        same_grid_oop_table=same_paths["per_patch"],
        output_directory=paths["intermediate"] / "confocal_spacing_audit",
        spacing_min_um=spacing_min_um,
        spacing_max_um=spacing_max_um,
        write_previews=write_previews,
    )

    manifest = build_pipeline_manifest(baseline_manifest, calibration)
    per_patch = build_pipeline_patch_table(same_patch, spacing_patch)
    per_image = build_pipeline_image_table(manifest, baseline_image, same_image, spacing_image, spacing_patch)
    preview_paths = collect_pipeline_previews(paths, write_previews)
    summary = build_pipeline_summary(
        manifest,
        per_patch,
        per_image,
        metadata_summary,
        spacing_summary,
        write_previews=write_previews,
        preview_paths=preview_paths,
        step_summaries={
            "baseline": baseline_summary,
            "mask": mask_summary,
            "sensitivity": sensitivity_summary,
            "selective": selective_summary,
            "same_grid_oop": same_summary,
            "spacing": spacing_summary,
        },
    )
    write_confocal_pipeline_outputs(manifest, per_patch, per_image, summary, paths)
    return manifest, per_patch, per_image, summary, paths


def pipeline_cfg(cfg: dict[str, Any], intermediate_root: Path) -> dict[str, Any]:
    working = copy.deepcopy(cfg)
    working.setdefault("paths", {})
    working["paths"]["output_dir"] = str(intermediate_root)
    return working


def build_pipeline_manifest(manifest: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    working = normalize_ids(manifest)
    calibration = normalize_ids(calibration)
    keep = [
        column
        for column in [
            "confocal_image_id",
            "pixel_size_x_um",
            "pixel_size_y_um",
            "pixel_size_unit",
            "pixel_size_source",
            "pixel_size_available",
            "isotropic_pixels",
            "spacing_um_enabled",
            "calibration_warning",
        ]
        if column in calibration.columns
    ]
    if {"confocal_image_id", "pixel_size_x_um"}.issubset(keep):
        working = working.merge(calibration[keep].drop_duplicates("confocal_image_id"), on="confocal_image_id", how="left")
    return working


def build_pipeline_patch_table(same_patch: pd.DataFrame, spacing_patch: pd.DataFrame) -> pd.DataFrame:
    same = normalize_ids(same_patch)
    spacing = normalize_ids(spacing_patch)
    if spacing.empty:
        output = same.copy(deep=True)
    else:
        output = spacing.copy(deep=True)
    rename = {
        "patch_oop_128": "patch_oop",
        "patch_mean_orientation_deg_128": "patch_mean_orientation_deg",
        "patch_orientation_weight_sum_128": "patch_orientation_weight_sum",
        "patch_orientation_valid_pixels_128": "patch_orientation_valid_pixels",
        "patch_orientation_coherence_mean_128": "patch_orientation_coherence_mean",
    }
    output = output.rename(columns={key: value for key, value in rename.items() if key in output.columns})
    if "processing_status" not in output.columns:
        output["processing_status"] = "ok"
    if "error_message" not in output.columns:
        output["error_message"] = ""
    return ensure_columns(output, CONFOCAL_PIPELINE_PATCH_COLUMNS)


def build_pipeline_image_table(
    manifest: pd.DataFrame,
    baseline_image: pd.DataFrame,
    same_image: pd.DataFrame,
    spacing_image: pd.DataFrame,
    spacing_patch: pd.DataFrame,
) -> pd.DataFrame:
    manifest = normalize_ids(manifest)
    baseline = normalize_ids(baseline_image)
    same = normalize_ids(same_image)
    spacing = normalize_ids(spacing_image)
    rows: list[dict[str, Any]] = []
    for _, manifest_row in manifest.iterrows():
        image_id = str(manifest_row["confocal_image_id"])
        base = first_row(baseline, image_id)
        oop = first_row(same, image_id)
        space = first_row(spacing, image_id)
        image_spacing_patches = spacing_patch.loc[spacing_patch["confocal_image_id"].astype(str) == image_id] if not spacing_patch.empty else pd.DataFrame()
        spacing_min = to_float(space.get("selected_median_spacing_um"))
        valid_range = spacing_range_string(image_spacing_patches)
        processing_status = base.get("processing_status") or "ok"
        error_message = base.get("error_message") or ""
        rows.append(
            {
                "confocal_image_id": image_id,
                "filename": manifest_row.get("filename", base.get("filename", space.get("filename", ""))),
                "image_shape": image_shape_string(manifest_row, base),
                "pixel_size_x_um": manifest_row.get("pixel_size_x_um", np.nan),
                "pixel_size_y_um": manifest_row.get("pixel_size_y_um", np.nan),
                "pixel_size_available": bool_value(manifest_row.get("pixel_size_available", False)),
                "total_patches": int_or_nan(oop.get("total_patches", space.get("total_patches", 0))),
                "selected_candidate_patches": int_or_nan(
                    oop.get("candidate_patch_count", space.get("candidate_patch_count", 0))
                ),
                "selected_candidate_fraction": to_float(
                    oop.get("candidate_patch_fraction", safe_fraction(space.get("candidate_patch_count"), space.get("total_patches")))
                ),
                "selected_median_oop": to_float(oop.get("selected_region_median_oop_128")),
                "all_region_median_oop": to_float(oop.get("all_region_median_oop_128")),
                "selected_vs_all_oop_difference": to_float(oop.get("selected_vs_all_oop_difference_128")),
                "selected_median_coherence": to_float(oop.get("selected_region_median_coherence_128")),
                "all_region_median_coherence": to_float(oop.get("all_region_median_coherence_128")),
                "valid_selected_spacing_patches": int_or_nan(space.get("spacing_valid_patch_count_selected", 0)),
                "selected_spacing_valid_fraction": to_float(space.get("spacing_valid_fraction_selected")),
                "selected_spacing_median_um": spacing_min,
                "selected_spacing_iqr_um": to_float(space.get("selected_iqr_spacing_um")),
                "selected_spacing_range_um": valid_range,
                "interpretation_flag": space.get("interpretation_flag", oop.get("interpretation_flag", "")),
                "processing_status": processing_status,
                "error_message": error_message,
            }
        )
    return pd.DataFrame(rows, columns=CONFOCAL_PIPELINE_IMAGE_COLUMNS)


def build_pipeline_summary(
    manifest: pd.DataFrame,
    per_patch: pd.DataFrame,
    per_image: pd.DataFrame,
    metadata_summary: dict[str, Any],
    spacing_summary: dict[str, Any],
    write_previews: bool,
    preview_paths: list[str],
    step_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected = per_patch.loc[per_patch["candidate_striation_region"].fillna(False).astype(bool)] if not per_patch.empty else pd.DataFrame()
    valid_selected = selected.loc[selected["spacing_valid"].fillna(False).astype(bool)] if not selected.empty else pd.DataFrame()
    return json_safe(
        {
            "mode": "confocal_pipeline",
            "primary_gate_used": PRIMARY_CONFOCAL_GATE,
            "relaxed_gate_status": RELAXED_GATE_STATUS,
            "widefield_calibration_used": False,
            "images_processed": int(len(manifest)),
            "errors": int((per_image["processing_status"].astype(str) == "error").sum()) if not per_image.empty else 0,
            "calibrated_images": int(metadata_summary.get("pixel_size_available_count", 0)),
            "total_patches": int(len(per_patch)),
            "selected_candidate_patches": int(len(selected)),
            "valid_selected_spacing_patches": int(len(valid_selected)),
            "median_selected_oop": safe_median(per_image.get("selected_median_oop", pd.Series(dtype=float))),
            "median_all_region_oop": safe_median(per_image.get("all_region_median_oop", pd.Series(dtype=float))),
            "median_selected_spacing_um": safe_median(valid_selected.get("spacing_estimate_um", pd.Series(dtype=float))),
            "selected_spacing_iqr_um": safe_iqr(valid_selected.get("spacing_estimate_um", pd.Series(dtype=float))),
            "spacing_band_um": spacing_summary.get("spacing_band_um"),
            "previews_written": bool(write_previews),
            "preview_paths": preview_paths,
            "step_summaries_present": {key: bool(value) for key, value in step_summaries.items()},
            "exploratory_caveat": (
                "Confocal selected-region OOP and calibrated spacing are exploratory/manual-review-needed. "
                "The primary gate is moderate; moderate_relaxed_combined remains sensitivity/review only."
            ),
            "interpretation": [
                "Confocal-first orchestration wrapper only.",
                "Existing confocal modules are run in sequence with the primary moderate gate.",
                "Per-image confocal pixel calibration is used for micron spacing; widefield calibration is not used.",
                "Images without valid per-image calibration have micron spacing disabled.",
                "No algorithms, thresholds, widefield outputs, clinical analyses, or publication figures are changed.",
            ],
        }
    )


def collect_pipeline_previews(paths: dict[str, Path], write_previews: bool) -> list[str]:
    if not write_previews:
        return []
    preview_root = paths["previews"]
    preview_root.mkdir(parents=True, exist_ok=True)
    source_dirs = [
        paths["intermediate"] / "confocal_striation_mask" / "previews",
        paths["intermediate"] / "confocal_selective_analysis" / "previews",
        paths["intermediate"] / "confocal_same_grid_oop" / "previews",
        paths["intermediate"] / "confocal_spacing_audit" / "previews",
    ]
    copied: list[str] = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        prefix = source_dir.parent.name.replace("confocal_", "")
        for source in sorted(source_dir.glob("*.png")):
            destination = preview_root / f"{prefix}_{source.name}"
            shutil.copy2(source, destination)
            copied.append(str(destination))
    return copied


def write_confocal_pipeline_outputs(
    manifest: pd.DataFrame,
    per_patch: pd.DataFrame,
    per_image: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    manifest.to_csv(paths["manifest"], index=False)
    per_patch.to_csv(paths["per_patch"], index=False)
    per_image.to_csv(paths["per_image"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_confocal_pipeline_summary_text(summary), encoding="utf-8")


def render_confocal_pipeline_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal pipeline summary",
        f"primary_gate_used: {summary['primary_gate_used']}",
        f"relaxed_gate_status: {summary['relaxed_gate_status']}",
        f"widefield_calibration_used: {summary['widefield_calibration_used']}",
        f"images_processed: {summary['images_processed']}",
        f"errors: {summary['errors']}",
        f"calibrated_images: {summary['calibrated_images']}",
        f"total_patches: {summary['total_patches']}",
        f"selected_candidate_patches: {summary['selected_candidate_patches']}",
        f"valid_selected_spacing_patches: {summary['valid_selected_spacing_patches']}",
        f"median_selected_oop: {summary['median_selected_oop']}",
        f"median_all_region_oop: {summary['median_all_region_oop']}",
        f"median_selected_spacing_um: {summary['median_selected_spacing_um']}",
        f"selected_spacing_iqr_um: {summary['selected_spacing_iqr_um']}",
        "",
        summary["exploratory_caveat"],
        "",
    ]
    lines.extend(summary["interpretation"])
    return "\n".join(lines) + "\n"


def ensure_columns(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = table.copy(deep=True)
    for column in columns:
        if column not in output.columns:
            output[column] = np.nan
    return output[columns].copy()


def normalize_ids(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy(deep=True)
    for column in ["confocal_image_id", "filename", "patch_id"]:
        if column in output.columns:
            output[column] = output[column].astype(str)
    return output


def first_row(table: pd.DataFrame, image_id: str) -> dict[str, Any]:
    if table.empty or "confocal_image_id" not in table.columns:
        return {}
    matches = table.loc[table["confocal_image_id"].astype(str) == str(image_id)]
    return matches.iloc[0].to_dict() if not matches.empty else {}


def image_shape_string(manifest_row: pd.Series, baseline_row: dict[str, Any]) -> str:
    if "image_shape_y" in manifest_row and "image_shape_x" in manifest_row:
        return f"{manifest_row.get('image_shape_y')}x{manifest_row.get('image_shape_x')}"
    return str(baseline_row.get("shape", ""))


def spacing_range_string(spacing_patches: pd.DataFrame) -> str:
    if spacing_patches.empty:
        return ""
    selected = spacing_patches.loc[
        spacing_patches["candidate_striation_region"].fillna(False).astype(bool)
        & spacing_patches["spacing_valid"].fillna(False).astype(bool)
    ]
    values = safe_numeric(selected.get("spacing_estimate_um", pd.Series(dtype=float)))
    if values.size == 0:
        return ""
    return f"{float(np.min(values)):.4f}-{float(np.max(values)):.4f}"


def safe_fraction(numerator: Any, denominator: Any) -> float:
    top = to_float(numerator)
    bottom = to_float(denominator)
    return float(top / bottom) if np.isfinite(top) and np.isfinite(bottom) and bottom else np.nan


def safe_numeric(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def safe_median(values: pd.Series) -> float | None:
    numeric = safe_numeric(values)
    return None if numeric.size == 0 else float(np.median(numeric))


def safe_iqr(values: pd.Series) -> float | None:
    numeric = safe_numeric(values)
    if numeric.size == 0:
        return None
    q75, q25 = np.percentile(numeric, [75, 25])
    return float(q75 - q25)


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def int_or_nan(value: Any) -> int | float:
    numeric = to_float(value)
    return int(numeric) if np.isfinite(numeric) else np.nan


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)
