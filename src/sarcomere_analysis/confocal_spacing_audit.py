from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_dir
from .confocal_intake import load_confocal_image_2d
from .outputs import write_preview_png
from .preprocessing import preprocess_image
from .spacing.base import estimate_patch_spacing, spacing_params
from .zdisc_annotation import json_safe


CONFOCAL_SPACING_PATCH_COLUMNS = [
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
    "spacing_band_min_um",
    "spacing_band_max_um",
    "spacing_band_min_px",
    "spacing_band_max_px",
    "spacing_estimate_px",
    "spacing_estimate_um",
    "spacing_confidence",
    "spacing_valid",
    "spacing_failure_reason",
    "patch_oop_128",
    "patch_mean_orientation_deg_128",
    "patch_orientation_coherence_mean_128",
    "gradient_energy",
    "intensity_std",
    "contrast",
    "expected_positive_example",
    "noted_complex_example",
]

CONFOCAL_SPACING_IMAGE_COLUMNS = [
    "confocal_image_id",
    "filename",
    "pixel_size_um",
    "total_patches",
    "candidate_patch_count",
    "spacing_valid_patch_count_all",
    "spacing_valid_patch_count_selected",
    "spacing_valid_fraction_selected",
    "selected_median_spacing_um",
    "selected_iqr_spacing_um",
    "selected_median_spacing_px",
    "selected_median_spacing_confidence",
    "expected_positive_example",
    "noted_complex_example",
    "interpretation_flag",
]


def default_confocal_spacing_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_spacing_audit"
    return {
        "root": root,
        "per_patch": root / "confocal_spacing_per_patch.csv",
        "per_image": root / "confocal_spacing_per_image.csv",
        "summary_json": root / "confocal_spacing_summary.json",
        "summary_txt": root / "confocal_spacing_summary.txt",
        "previews": root / "previews",
    }


def run_confocal_spacing_audit(
    cfg: dict[str, Any],
    calibration_table: str | Path | None = None,
    same_grid_oop_table: str | Path | None = None,
    output_directory: str | Path | None = None,
    spacing_min_um: float = 1.5,
    spacing_max_um: float = 2.4,
    write_previews: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    root = output_dir(cfg)
    calibration_path = Path(calibration_table) if calibration_table else root / "confocal_metadata" / "confocal_metadata_calibration.csv"
    patch_path = (
        Path(same_grid_oop_table)
        if same_grid_oop_table
        else root / "confocal_same_grid_oop" / "confocal_same_grid_oop_per_patch.csv"
    )
    calibration = pd.read_csv(calibration_path, dtype={"confocal_image_id": str, "filename": str, "source_path": str})
    patches = pd.read_csv(patch_path, dtype={"confocal_image_id": str, "filename": str, "patch_id": str})
    paths = default_confocal_spacing_paths(cfg, output_directory)

    per_image_patch_tables: list[pd.DataFrame] = []
    preview_paths: list[str] = []
    for _, calibration_row in calibration.iterrows():
        image_id = str(calibration_row["confocal_image_id"])
        image_patches = patches.loc[patches["confocal_image_id"].astype(str) == image_id].copy()
        measured, previews = measure_confocal_image_spacing(
            cfg,
            calibration_row,
            image_patches,
            spacing_min_um,
            spacing_max_um,
            paths["previews"],
            write_previews,
        )
        per_image_patch_tables.append(measured)
        preview_paths.extend(str(path) for path in previews)

    per_patch = (
        pd.concat(per_image_patch_tables, ignore_index=True)
        if per_image_patch_tables
        else pd.DataFrame(columns=CONFOCAL_SPACING_PATCH_COLUMNS)
    )
    per_image = summarize_confocal_spacing_images(per_patch)
    summary = build_confocal_spacing_summary(
        calibration,
        per_patch,
        per_image,
        spacing_min_um,
        spacing_max_um,
        write_previews,
        preview_paths,
    )
    write_confocal_spacing_outputs(per_patch, per_image, summary, paths)
    return per_patch, per_image, summary, paths


def measure_confocal_image_spacing(
    cfg: dict[str, Any],
    calibration_row: pd.Series,
    patches: pd.DataFrame,
    spacing_min_um: float,
    spacing_max_um: float,
    preview_dir: Path,
    write_previews: bool,
) -> tuple[pd.DataFrame, list[Path]]:
    previews: list[Path] = []
    if patches.empty:
        return pd.DataFrame(columns=CONFOCAL_SPACING_PATCH_COLUMNS), previews

    pixel_size_x = to_float(calibration_row.get("pixel_size_x_um"))
    pixel_size_y = to_float(calibration_row.get("pixel_size_y_um"))
    calibrated = bool(calibration_row.get("pixel_size_available", False))
    isotropic = bool(calibration_row.get("isotropic_pixels", False))
    pixel_size_um = pixel_size_x if calibrated and isotropic else np.nan
    if not calibrated:
        return mark_image_patches_invalid(calibration_row, patches, spacing_min_um, spacing_max_um, np.nan, "missing_per_image_pixel_size"), previews
    if not isotropic:
        return mark_image_patches_invalid(calibration_row, patches, spacing_min_um, spacing_max_um, np.nan, "anisotropic_pixel_size"), previews

    spacing_cfg = confocal_spacing_config(cfg, pixel_size_um, spacing_min_um, spacing_max_um)
    raw, _ = load_confocal_image_2d(str(calibration_row["source_path"]))
    preprocessed = preprocess_image(raw, spacing_cfg).image
    params = spacing_params(spacing_cfg)
    rows = []
    for _, patch in patches.iterrows():
        rows.append(measure_patch_spacing(preprocessed, patch, calibration_row, spacing_cfg, params, spacing_min_um, spacing_max_um))
    measured = pd.DataFrame(rows, columns=CONFOCAL_SPACING_PATCH_COLUMNS)
    if write_previews:
        previews = write_confocal_spacing_previews(str(calibration_row["confocal_image_id"]), preprocessed, measured, preview_dir)
    return measured, previews


def measure_patch_spacing(
    image: np.ndarray,
    patch: pd.Series,
    calibration_row: pd.Series,
    spacing_cfg: dict[str, Any],
    params: dict[str, Any],
    spacing_min_um: float,
    spacing_max_um: float,
) -> dict[str, Any]:
    pixel_size_x = float(calibration_row["pixel_size_x_um"])
    pixel_size_y = float(calibration_row["pixel_size_y_um"])
    spacing_band_min_px = spacing_min_um / pixel_size_x
    spacing_band_max_px = spacing_max_um / pixel_size_x
    row = base_patch_row(patch, calibration_row, spacing_min_um, spacing_max_um, spacing_band_min_px, spacing_band_max_px)
    if not bool_value(patch.get("candidate_striation_region", False)):
        row["spacing_failure_reason"] = "not_candidate_region"
        return row
    orientation_deg = to_float(patch.get("patch_mean_orientation_deg_128"))
    if not np.isfinite(orientation_deg):
        row["spacing_failure_reason"] = "missing_orientation"
        return row
    try:
        patch_for_spacing = pd.Series(
            {
                "valid_for_spacing": True,
                "invalid_reason": "ok",
                "patch_mean_orientation_rad": np.deg2rad(orientation_deg),
                "y0": int(patch["y0"]),
                "x0": int(patch["x0"]),
                "y1": int(patch["y1"]),
                "x1": int(patch["x1"]),
            }
        )
        result = estimate_patch_spacing(image, patch_for_spacing, params, spacing_cfg)
        row.update(
            {
                "spacing_estimate_px": result.patch_spacing_px,
                "spacing_estimate_um": result.patch_spacing_um,
                "spacing_confidence": result.patch_spacing_confidence,
                "spacing_valid": bool(result.valid_for_spacing_final),
                "spacing_failure_reason": result.spacing_invalid_reason,
            }
        )
    except Exception as exc:
        row["spacing_failure_reason"] = f"spacing_error:{exc}"
    return row


def base_patch_row(
    patch: pd.Series,
    calibration_row: pd.Series,
    spacing_min_um: float,
    spacing_max_um: float,
    spacing_band_min_px: float,
    spacing_band_max_px: float,
) -> dict[str, Any]:
    return {
        "confocal_image_id": str(patch.get("confocal_image_id", calibration_row.get("confocal_image_id"))),
        "filename": str(patch.get("filename", calibration_row.get("filename"))),
        "patch_id": str(patch.get("patch_id", "")),
        "y0": patch.get("y0", np.nan),
        "x0": patch.get("x0", np.nan),
        "y1": patch.get("y1", np.nan),
        "x1": patch.get("x1", np.nan),
        "center_y": patch.get("center_y", np.nan),
        "center_x": patch.get("center_x", np.nan),
        "candidate_striation_region": bool_value(patch.get("candidate_striation_region", False)),
        "pixel_size_x_um": calibration_row.get("pixel_size_x_um", np.nan),
        "pixel_size_y_um": calibration_row.get("pixel_size_y_um", np.nan),
        "spacing_band_min_um": float(spacing_min_um),
        "spacing_band_max_um": float(spacing_max_um),
        "spacing_band_min_px": float(spacing_band_min_px) if np.isfinite(spacing_band_min_px) else np.nan,
        "spacing_band_max_px": float(spacing_band_max_px) if np.isfinite(spacing_band_max_px) else np.nan,
        "spacing_estimate_px": np.nan,
        "spacing_estimate_um": np.nan,
        "spacing_confidence": 0.0,
        "spacing_valid": False,
        "spacing_failure_reason": "not_evaluated",
        "patch_oop_128": patch.get("patch_oop_128", np.nan),
        "patch_mean_orientation_deg_128": patch.get("patch_mean_orientation_deg_128", np.nan),
        "patch_orientation_coherence_mean_128": patch.get("patch_orientation_coherence_mean_128", np.nan),
        "gradient_energy": patch.get("gradient_energy", np.nan),
        "intensity_std": patch.get("intensity_std", np.nan),
        "contrast": patch.get("contrast", np.nan),
        "expected_positive_example": bool_value(patch.get("expected_positive_example", calibration_row.get("expected_positive_example", False))),
        "noted_complex_example": bool_value(patch.get("noted_complex_example", calibration_row.get("noted_complex_example", False))),
    }


def mark_image_patches_invalid(
    calibration_row: pd.Series,
    patches: pd.DataFrame,
    spacing_min_um: float,
    spacing_max_um: float,
    pixel_size_um: float,
    reason: str,
) -> pd.DataFrame:
    rows = []
    min_px = spacing_min_um / pixel_size_um if np.isfinite(pixel_size_um) and pixel_size_um > 0 else np.nan
    max_px = spacing_max_um / pixel_size_um if np.isfinite(pixel_size_um) and pixel_size_um > 0 else np.nan
    for _, patch in patches.iterrows():
        row = base_patch_row(patch, calibration_row, spacing_min_um, spacing_max_um, min_px, max_px)
        row["spacing_failure_reason"] = reason
        rows.append(row)
    return pd.DataFrame(rows, columns=CONFOCAL_SPACING_PATCH_COLUMNS)


def summarize_confocal_spacing_images(per_patch: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if per_patch.empty:
        return pd.DataFrame(columns=CONFOCAL_SPACING_IMAGE_COLUMNS)
    for (image_id, filename), group in per_patch.groupby(["confocal_image_id", "filename"], dropna=False):
        candidates = group.loc[group["candidate_striation_region"].fillna(False).astype(bool)]
        valid_all = group.loc[group["spacing_valid"].fillna(False).astype(bool)]
        valid_selected = candidates.loc[candidates["spacing_valid"].fillna(False).astype(bool)]
        total = int(len(group))
        candidate_count = int(len(candidates))
        selected_valid_count = int(len(valid_selected))
        rows.append(
            {
                "confocal_image_id": str(image_id),
                "filename": str(filename),
                "pixel_size_um": safe_median(group.get("pixel_size_x_um", pd.Series(dtype=float))),
                "total_patches": total,
                "candidate_patch_count": candidate_count,
                "spacing_valid_patch_count_all": int(len(valid_all)),
                "spacing_valid_patch_count_selected": selected_valid_count,
                "spacing_valid_fraction_selected": float(selected_valid_count / candidate_count) if candidate_count else 0.0,
                "selected_median_spacing_um": safe_median(valid_selected.get("spacing_estimate_um", pd.Series(dtype=float))),
                "selected_iqr_spacing_um": safe_iqr(valid_selected.get("spacing_estimate_um", pd.Series(dtype=float))),
                "selected_median_spacing_px": safe_median(valid_selected.get("spacing_estimate_px", pd.Series(dtype=float))),
                "selected_median_spacing_confidence": safe_median(valid_selected.get("spacing_confidence", pd.Series(dtype=float))),
                "expected_positive_example": bool(group["expected_positive_example"].fillna(False).astype(bool).any()),
                "noted_complex_example": bool(group["noted_complex_example"].fillna(False).astype(bool).any()),
                "interpretation_flag": spacing_interpretation_flag(candidate_count, selected_valid_count),
            }
        )
    return pd.DataFrame(rows, columns=CONFOCAL_SPACING_IMAGE_COLUMNS)


def spacing_interpretation_flag(candidate_count: int, selected_valid_count: int) -> str:
    if candidate_count == 0:
        return "no_candidate_regions"
    if selected_valid_count == 0:
        return "no_valid_spacing_selected_regions"
    if selected_valid_count < 5:
        return "very_low_spacing_yield_review_needed"
    return "exploratory_spacing_yield_review_needed"


def build_confocal_spacing_summary(
    calibration: pd.DataFrame,
    per_patch: pd.DataFrame,
    per_image: pd.DataFrame,
    spacing_min_um: float,
    spacing_max_um: float,
    write_previews: bool,
    preview_paths: list[str],
) -> dict[str, Any]:
    selected = per_patch.loc[per_patch["candidate_striation_region"].fillna(False).astype(bool)] if not per_patch.empty else pd.DataFrame()
    valid_selected = selected.loc[selected["spacing_valid"].fillna(False).astype(bool)] if not selected.empty else pd.DataFrame()
    return json_safe(
        {
            "mode": "confocal_calibrated_spacing_audit",
            "image_count": int(len(calibration)),
            "calibrated_image_count": int(calibration["pixel_size_available"].fillna(False).astype(bool).sum())
            if "pixel_size_available" in calibration
            else 0,
            "widefield_calibration_used": False,
            "spacing_band_um": {"min": float(spacing_min_um), "max": float(spacing_max_um)},
            "total_patch_rows": int(len(per_patch)),
            "candidate_patch_count": int(len(selected)),
            "valid_spacing_patch_count_all": int(per_patch["spacing_valid"].fillna(False).astype(bool).sum()) if not per_patch.empty else 0,
            "valid_spacing_patch_count_selected": int(len(valid_selected)),
            "valid_spacing_fraction_selected": float(len(valid_selected) / len(selected)) if len(selected) else 0.0,
            "selected_spacing_um_summary": {
                "median": safe_median(valid_selected.get("spacing_estimate_um", pd.Series(dtype=float))),
                "iqr": safe_iqr(valid_selected.get("spacing_estimate_um", pd.Series(dtype=float))),
                "min": safe_min(valid_selected.get("spacing_estimate_um", pd.Series(dtype=float))),
                "max": safe_max(valid_selected.get("spacing_estimate_um", pd.Series(dtype=float))),
            },
            "failure_reason_counts": per_patch["spacing_failure_reason"].value_counts(dropna=False).to_dict() if not per_patch.empty else {},
            "selected_failure_reason_counts": selected["spacing_failure_reason"].value_counts(dropna=False).to_dict() if not selected.empty else {},
            "special_image_summaries": special_image_summaries(per_image),
            "previews_written": bool(write_previews),
            "preview_paths": preview_paths,
            "comparison_to_widefield_spacing_failure": (
                "Widefield spacing remained exploratory/low-yield. This confocal audit enables micron spacing only through per-image "
                "confocal calibration and remains exploratory until manually reviewed."
            ),
            "interpretation": [
                "Exploratory calibrated confocal spacing audit only.",
                "Spacing is evaluated primarily inside moderate confident-striation candidate patches.",
                "Per-image confocal pixel calibration is used; widefield calibration is not used.",
                "No thresholds were tuned to increase yield.",
                "Low or zero yield should be reported honestly rather than forced.",
                "No biological claims are made.",
            ],
        }
    )


def special_image_summaries(per_image: pd.DataFrame) -> list[dict[str, Any]]:
    if per_image.empty:
        return []
    mask = (
        per_image["expected_positive_example"].fillna(False).astype(bool)
        | per_image["noted_complex_example"].fillna(False).astype(bool)
        | per_image["confocal_image_id"].astype(str).str.contains("7028", case=False, regex=False)
    )
    return json_safe(per_image.loc[mask].to_dict("records"))


def confocal_spacing_config(cfg: dict[str, Any], pixel_size_um: float, spacing_min_um: float, spacing_max_um: float) -> dict[str, Any]:
    working = dict(cfg)
    working["calibration"] = {
        "pixel_size_um": float(pixel_size_um),
        "expected_sarcomere_spacing_um": {"min": float(spacing_min_um), "max": float(spacing_max_um)},
    }
    spacing = dict(working.get("spacing", {}))
    spacing["min_spacing_um"] = float(spacing_min_um)
    spacing["max_spacing_um"] = float(spacing_max_um)
    working["spacing"] = spacing
    return working


def write_confocal_spacing_outputs(
    per_patch: pd.DataFrame,
    per_image: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    per_patch.to_csv(paths["per_patch"], index=False)
    per_image.to_csv(paths["per_image"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_confocal_spacing_summary_text(summary), encoding="utf-8")


def render_confocal_spacing_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal calibrated spacing audit",
        f"image_count: {summary['image_count']}",
        f"calibrated_image_count: {summary['calibrated_image_count']}",
        f"widefield_calibration_used: {summary['widefield_calibration_used']}",
        f"spacing_band_um: {summary['spacing_band_um']}",
        f"total_patch_rows: {summary['total_patch_rows']}",
        f"candidate_patch_count: {summary['candidate_patch_count']}",
        f"valid_spacing_patch_count_all: {summary['valid_spacing_patch_count_all']}",
        f"valid_spacing_patch_count_selected: {summary['valid_spacing_patch_count_selected']}",
        f"valid_spacing_fraction_selected: {summary['valid_spacing_fraction_selected']}",
        f"selected_spacing_um_summary: {summary['selected_spacing_um_summary']}",
        "",
        "Special image summaries:",
    ]
    for row in summary["special_image_summaries"]:
        lines.append(
            f"- {row.get('filename')}: selected_valid={row.get('spacing_valid_patch_count_selected')}, "
            f"selected_fraction={row.get('spacing_valid_fraction_selected')}, "
            f"median_um={row.get('selected_median_spacing_um')}, flag={row.get('interpretation_flag')}"
        )
    lines.extend(["", "Top failure reasons:"])
    for reason, count in list(summary["selected_failure_reason_counts"].items())[:10]:
        lines.append(f"- {reason}: {count}")
    lines.append("")
    lines.extend(summary["interpretation"])
    return "\n".join(lines) + "\n"


def write_confocal_spacing_previews(image_id: str, image: np.ndarray, measured: pd.DataFrame, preview_dir: Path) -> list[Path]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    return [
        write_candidate_overlay(image, measured, preview_dir / f"{image_id}_confocal_spacing_candidate_overlay.png"),
        write_valid_spacing_overlay(image, measured, preview_dir / f"{image_id}_confocal_valid_spacing_overlay.png"),
        write_spacing_heatmap(image, measured, preview_dir / f"{image_id}_confocal_spacing_um_heatmap.png"),
    ]


def write_candidate_overlay(image: np.ndarray, patches: pd.DataFrame, path: str | Path) -> Path:
    rgb = np.dstack([image, image, image]).astype(np.float32)
    color = np.array([1.0, 0.15, 0.05], dtype=np.float32)
    alpha = 0.3
    for _, row in patches.iterrows():
        if not bool_value(row.get("candidate_striation_region", False)):
            continue
        y0, y1, x0, x1 = int(row["y0"]), int(row["y1"]), int(row["x0"]), int(row["x1"])
        rgb[y0:y1, x0:x1] = (1.0 - alpha) * rgb[y0:y1, x0:x1] + alpha * color
    return write_preview_png(rgb, path)


def write_valid_spacing_overlay(image: np.ndarray, patches: pd.DataFrame, path: str | Path) -> Path:
    rgb = np.dstack([image, image, image]).astype(np.float32)
    color = np.array([0.0, 0.9, 0.25], dtype=np.float32)
    alpha = 0.45
    for _, row in patches.iterrows():
        if not bool_value(row.get("spacing_valid", False)):
            continue
        y0, y1, x0, x1 = int(row["y0"]), int(row["y1"]), int(row["x0"]), int(row["x1"])
        rgb[y0:y1, x0:x1] = (1.0 - alpha) * rgb[y0:y1, x0:x1] + alpha * color
    return write_preview_png(rgb, path)


def write_spacing_heatmap(image: np.ndarray, patches: pd.DataFrame, path: str | Path) -> Path:
    heatmap = np.full(image.shape, np.nan, dtype=np.float32)
    for _, row in patches.iterrows():
        value = to_float(row.get("spacing_estimate_um"))
        if not np.isfinite(value):
            continue
        y0, y1, x0, x1 = int(row["y0"]), int(row["y1"]), int(row["x0"]), int(row["x1"])
        heatmap[y0:y1, x0:x1] = float(value)
    return write_preview_png(heatmap, path)


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


def safe_min(values: pd.Series) -> float | None:
    numeric = safe_numeric(values)
    return None if numeric.size == 0 else float(np.min(numeric))


def safe_max(values: pd.Series) -> float | None:
    numeric = safe_numeric(values)
    return None if numeric.size == 0 else float(np.max(numeric))


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)
