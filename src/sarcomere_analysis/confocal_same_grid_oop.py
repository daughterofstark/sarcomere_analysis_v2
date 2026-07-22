from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from .confocal_intake import load_confocal_image_2d
from .config import output_dir
from .masking import compute_tissue_mask
from .orientation import axial_order_parameter, orientation_params, orientation_weights, radians_to_degrees, structure_tensor_orientation
from .outputs import write_preview_png
from .preprocessing import preprocess_image
from .zdisc_annotation import json_safe


SAME_GRID_PATCH_COLUMNS = [
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
    "candidate_source",
    "expected_positive_example",
    "noted_complex_example",
    "patch_oop_128",
    "patch_mean_orientation_deg_128",
    "patch_orientation_weight_sum_128",
    "patch_orientation_valid_pixels_128",
    "patch_orientation_coherence_mean_128",
    "gradient_energy",
    "intensity_std",
    "contrast",
    "processing_status",
    "error_message",
]

SAME_GRID_IMAGE_COLUMNS = [
    "confocal_image_id",
    "filename",
    "total_patches",
    "candidate_patch_count",
    "candidate_patch_fraction",
    "selected_region_median_oop_128",
    "selected_region_iqr_oop_128",
    "all_region_median_oop_128",
    "all_region_iqr_oop_128",
    "selected_vs_all_oop_difference_128",
    "selected_region_median_orientation_valid_pixels_128",
    "all_region_median_orientation_valid_pixels_128",
    "selected_region_median_coherence_128",
    "all_region_median_coherence_128",
    "expected_positive_example",
    "noted_complex_example",
    "interpretation_flag",
]


def default_same_grid_paths(cfg: dict[str, Any], output_directory: str | Path | None = None) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_same_grid_oop"
    return {
        "root": root,
        "per_patch": root / "confocal_same_grid_oop_per_patch.csv",
        "per_image": root / "confocal_same_grid_oop_per_image.csv",
        "summary_json": root / "confocal_same_grid_oop_summary.json",
        "summary_txt": root / "confocal_same_grid_oop_summary.txt",
        "previews": root / "previews",
    }


def run_confocal_same_grid_oop(
    cfg: dict[str, Any],
    patch_table: str | Path | None = None,
    manifest: str | Path | None = None,
    output_directory: str | Path | None = None,
    write_previews: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    root = output_dir(cfg)
    patch_path = Path(patch_table) if patch_table else root / "confocal_striation_mask" / "confocal_striation_mask_per_patch.csv"
    manifest_path = Path(manifest) if manifest else root / "confocal_baseline" / "confocal_manifest.csv"
    patches = pd.read_csv(patch_path, dtype={"confocal_image_id": str, "filename": str, "patch_id": str})
    manifest_df = pd.read_csv(manifest_path, dtype={"confocal_image_id": str, "filename": str, "source_path": str})
    patches, candidate_source = attach_moderate_candidate_flags(patches, root)

    patch_tables: list[pd.DataFrame] = []
    preview_paths: list[str] = []
    paths = default_same_grid_paths(cfg, output_directory)
    for _, image_row in manifest_df.iterrows():
        image_id = str(image_row["confocal_image_id"])
        image_patches = patches.loc[patches["confocal_image_id"].astype(str) == image_id].copy()
        measured, previews = measure_same_grid_image(image_row, image_patches, cfg, paths["previews"], write_previews)
        patch_tables.append(measured)
        preview_paths.extend(str(path) for path in previews)

    per_patch = pd.concat(patch_tables, ignore_index=True) if patch_tables else pd.DataFrame(columns=SAME_GRID_PATCH_COLUMNS)
    per_image = summarize_same_grid_images(per_patch)
    summary = build_same_grid_summary(per_patch, per_image, candidate_source, write_previews, preview_paths)
    write_same_grid_outputs(per_patch, per_image, summary, paths)
    return per_patch, per_image, summary, paths


def attach_moderate_candidate_flags(patches: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, str]:
    working = patches.copy(deep=True)
    selective_path = root / "confocal_selective_analysis" / "confocal_selective_per_patch.csv"
    if selective_path.exists():
        selected = pd.read_csv(selective_path, dtype={"confocal_image_id": str, "patch_id": str})
        selected = selected.loc[selected.get("selected_variant", "moderate").astype(str) == "moderate"].copy() if "selected_variant" in selected else selected
        keep = [
            column
            for column in ["confocal_image_id", "patch_id", "candidate_striation_region", "expected_positive_example", "noted_complex_example"]
            if column in selected.columns
        ]
        if {"confocal_image_id", "patch_id", "candidate_striation_region"}.issubset(keep):
            selected = selected[keep].drop_duplicates(["confocal_image_id", "patch_id"])
            working = working.drop(columns=[column for column in ["candidate_striation_region"] if column in working.columns])
            working = working.merge(selected, on=["confocal_image_id", "patch_id"], how="left", suffixes=("", "_selected"))
            for column in ["expected_positive_example", "noted_complex_example"]:
                selected_column = f"{column}_selected"
                if selected_column in working.columns:
                    working[column] = working[selected_column].where(working[selected_column].notna(), working.get(column))
                    working = working.drop(columns=[selected_column])
            working["candidate_striation_region"] = working["candidate_striation_region"].fillna(False).astype(bool)
            working["candidate_source"] = "confocal_selective_analysis_moderate"
            return working, "confocal_selective_analysis_moderate"
    working["candidate_striation_region"] = working.get("candidate_striation_region", pd.Series(False, index=working.index)).fillna(False).astype(bool)
    working["candidate_source"] = "confocal_striation_mask_existing_candidate_column"
    return working, "confocal_striation_mask_existing_candidate_column"


def measure_same_grid_image(
    manifest_row: pd.Series,
    patches: pd.DataFrame,
    cfg: dict[str, Any],
    preview_dir: Path,
    write_previews: bool,
) -> tuple[pd.DataFrame, list[Path]]:
    image_id = str(manifest_row["confocal_image_id"])
    previews: list[Path] = []
    rows: list[dict[str, Any]] = []
    if patches.empty:
        return pd.DataFrame(columns=SAME_GRID_PATCH_COLUMNS), previews
    try:
        raw, _ = load_confocal_image_2d(str(manifest_row["source_path"]))
        preprocessed = preprocess_image(raw, cfg)
        tissue = compute_tissue_mask(preprocessed.image, cfg)
        params = orientation_params(cfg)
        orientation_map, coherence_map, energy_map = structure_tensor_orientation(preprocessed.image, params)
        weights = orientation_weights(energy_map, coherence_map, str(params["weight_mode"]))
        for _, patch in patches.iterrows():
            rows.append(measure_patch_row(patch, orientation_map, coherence_map, weights, tissue.mask, params, "ok", ""))
        measured = pd.DataFrame(rows, columns=SAME_GRID_PATCH_COLUMNS)
        if write_previews:
            previews = write_same_grid_previews(image_id, preprocessed.image, measured, preview_dir)
        return measured, previews
    except Exception as exc:  # pragma: no cover - real-world image protection.
        for _, patch in patches.iterrows():
            rows.append(measure_patch_row(patch, None, None, None, None, {}, "error", str(exc)))
        return pd.DataFrame(rows, columns=SAME_GRID_PATCH_COLUMNS), previews


def measure_patch_row(
    patch: pd.Series,
    orientation_map: np.ndarray | None,
    coherence_map: np.ndarray | None,
    weights: np.ndarray | None,
    tissue_mask: np.ndarray | None,
    params: dict[str, Any],
    status: str,
    error_message: str,
) -> dict[str, Any]:
    base = {column: patch.get(column, np.nan) for column in SAME_GRID_PATCH_COLUMNS}
    base.update(
        {
            "candidate_source": patch.get("candidate_source", np.nan),
            "patch_oop_128": np.nan,
            "patch_mean_orientation_deg_128": np.nan,
            "patch_orientation_weight_sum_128": 0.0,
            "patch_orientation_valid_pixels_128": 0,
            "patch_orientation_coherence_mean_128": np.nan,
            "processing_status": status,
            "error_message": error_message,
        }
    )
    if status != "ok":
        return base
    try:
        y0, y1, x0, x1 = int(patch["y0"]), int(patch["y1"]), int(patch["x0"]), int(patch["x1"])
        if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0 or orientation_map is None or weights is None or tissue_mask is None:
            raise ValueError("invalid_patch_coordinates")
        if y1 > orientation_map.shape[0] or x1 > orientation_map.shape[1]:
            raise ValueError("patch_coordinates_out_of_bounds")
        valid = (
            tissue_mask[y0:y1, x0:x1]
            & np.isfinite(orientation_map[y0:y1, x0:x1])
            & np.isfinite(weights[y0:y1, x0:x1])
            & (weights[y0:y1, x0:x1] > 0)
        )
        oop, mean_rad, weight_sum, valid_pixels = axial_order_parameter(
            orientation_map[y0:y1, x0:x1],
            weights[y0:y1, x0:x1],
            valid,
            float(params["min_orientation_weight_sum"]),
            int(params["min_orientation_valid_pixels"]),
        )
        coherence_values = coherence_map[y0:y1, x0:x1][valid] if coherence_map is not None and np.any(valid) else np.array([])
        base.update(
            {
                "patch_oop_128": oop,
                "patch_mean_orientation_deg_128": radians_to_degrees(mean_rad),
                "patch_orientation_weight_sum_128": weight_sum,
                "patch_orientation_valid_pixels_128": valid_pixels,
                "patch_orientation_coherence_mean_128": float(np.mean(coherence_values)) if coherence_values.size else np.nan,
                "processing_status": "ok",
                "error_message": "",
            }
        )
    except Exception as exc:
        base["processing_status"] = "error"
        base["error_message"] = str(exc)
    return base


def summarize_same_grid_images(per_patch: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (image_id, filename), group in per_patch.groupby(["confocal_image_id", "filename"], dropna=False):
        candidates = group.loc[group["candidate_striation_region"].fillna(False).astype(bool)]
        total = int(len(group))
        count = int(len(candidates))
        selected_oop = safe_median(candidates.get("patch_oop_128", pd.Series(dtype=float)))
        all_oop = safe_median(group.get("patch_oop_128", pd.Series(dtype=float)))
        rows.append(
            {
                "confocal_image_id": str(image_id),
                "filename": str(filename),
                "total_patches": total,
                "candidate_patch_count": count,
                "candidate_patch_fraction": float(count / total) if total else 0.0,
                "selected_region_median_oop_128": selected_oop,
                "selected_region_iqr_oop_128": safe_iqr(candidates.get("patch_oop_128", pd.Series(dtype=float))),
                "all_region_median_oop_128": all_oop,
                "all_region_iqr_oop_128": safe_iqr(group.get("patch_oop_128", pd.Series(dtype=float))),
                "selected_vs_all_oop_difference_128": difference_or_nan(selected_oop, all_oop),
                "selected_region_median_orientation_valid_pixels_128": safe_median(candidates.get("patch_orientation_valid_pixels_128", pd.Series(dtype=float))),
                "all_region_median_orientation_valid_pixels_128": safe_median(group.get("patch_orientation_valid_pixels_128", pd.Series(dtype=float))),
                "selected_region_median_coherence_128": safe_median(candidates.get("patch_orientation_coherence_mean_128", pd.Series(dtype=float))),
                "all_region_median_coherence_128": safe_median(group.get("patch_orientation_coherence_mean_128", pd.Series(dtype=float))),
                "expected_positive_example": bool(group["expected_positive_example"].fillna(False).astype(bool).any())
                if "expected_positive_example" in group
                else False,
                "noted_complex_example": bool(group["noted_complex_example"].fillna(False).astype(bool).any())
                if "noted_complex_example" in group
                else False,
                "interpretation_flag": same_grid_interpretation_flag(count, float(count / total) if total else 0.0),
            }
        )
    return pd.DataFrame(rows, columns=SAME_GRID_IMAGE_COLUMNS)


def same_grid_interpretation_flag(candidate_count: int, candidate_fraction: float) -> str:
    flags: list[str] = []
    if candidate_count < 10:
        flags.append("too_few_candidates")
    if candidate_fraction > 0.60:
        flags.append("broad_candidate_fraction_review_needed")
    if not flags:
        flags.append("review_needed")
    return ";".join(flags)


def build_same_grid_summary(
    per_patch: pd.DataFrame,
    per_image: pd.DataFrame,
    candidate_source: str,
    write_previews: bool,
    preview_paths: list[str],
) -> dict[str, Any]:
    return json_safe(
        {
            "mode": "confocal_same_grid_oop",
            "candidate_source": candidate_source,
            "same_grid_patch_rows": int(len(per_patch)),
            "patches_processed_ok": int((per_patch["processing_status"] == "ok").sum()) if not per_patch.empty else 0,
            "patches_error": int((per_patch["processing_status"] == "error").sum()) if not per_patch.empty else 0,
            "candidate_patch_count": int(per_patch["candidate_striation_region"].fillna(False).astype(bool).sum())
            if not per_patch.empty
            else 0,
            "selected_vs_all_oop_summary": {
                "median_selected_region_oop_128": safe_median(per_image.get("selected_region_median_oop_128", pd.Series(dtype=float))),
                "median_all_region_oop_128": safe_median(per_image.get("all_region_median_oop_128", pd.Series(dtype=float))),
                "median_selected_vs_all_oop_difference_128": safe_median(
                    per_image.get("selected_vs_all_oop_difference_128", pd.Series(dtype=float))
                ),
                "median_selected_region_coherence_128": safe_median(per_image.get("selected_region_median_coherence_128", pd.Series(dtype=float))),
                "median_all_region_coherence_128": safe_median(per_image.get("all_region_median_coherence_128", pd.Series(dtype=float))),
            },
            "selected_region_summaries": special_image_records(per_image, include_7028=True),
            "spacing_status": "not_computed_in_microns_confocal_pixel_size_unknown",
            "previews_written": bool(write_previews),
            "preview_paths": preview_paths,
            "interpretation": [
                "Exploratory confocal same-grid OOP/orientation analysis only.",
                "OOP/orientation was computed directly on the 128 px patch grid used by the moderate candidate mask.",
                "This avoids the rejected 256 px baseline-grid OOP join.",
                "No spacing in microns is computed without confocal pixel calibration.",
                "Not manually validated on confocal annotations yet.",
                "No biological claims are made.",
            ],
        }
    )


def special_image_records(per_image: pd.DataFrame, include_7028: bool = False) -> list[dict[str, Any]]:
    if per_image.empty:
        return []
    mask = per_image["expected_positive_example"].fillna(False).astype(bool) | per_image["noted_complex_example"].fillna(False).astype(bool)
    if include_7028:
        mask |= per_image["confocal_image_id"].astype(str).str.contains("7028", case=False, regex=False)
    return json_safe(per_image.loc[mask].to_dict("records"))


def write_same_grid_previews(image_id: str, image: np.ndarray, measured: pd.DataFrame, preview_dir: Path) -> list[Path]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    return [
        write_same_grid_candidate_overlay(image, measured, preview_dir / f"{image_id}_same_grid_candidate_overlay.png"),
        write_same_grid_heatmap(image, measured, "patch_oop_128", preview_dir / f"{image_id}_same_grid_oop_heatmap.png"),
    ]


def write_same_grid_candidate_overlay(image: np.ndarray, patches: pd.DataFrame, path: str | Path) -> Path:
    rgb = np.dstack([image, image, image]).astype(np.float32)
    alpha = 0.35
    color = np.array([1.0, 0.1, 0.1], dtype=np.float32)
    for _, row in patches.iterrows():
        if not bool(row.get("candidate_striation_region", False)):
            continue
        y0, y1, x0, x1 = int(row["y0"]), int(row["y1"]), int(row["x0"]), int(row["x1"])
        rgb[y0:y1, x0:x1] = (1.0 - alpha) * rgb[y0:y1, x0:x1] + alpha * color
    return write_preview_png(rgb, path)


def write_same_grid_heatmap(image: np.ndarray, patches: pd.DataFrame, column: str, path: str | Path) -> Path:
    heatmap = np.full(image.shape, np.nan, dtype=np.float32)
    for _, row in patches.iterrows():
        value = row.get(column, np.nan)
        if not np.isfinite(value):
            continue
        y0, y1, x0, x1 = int(row["y0"]), int(row["y1"]), int(row["x0"]), int(row["x1"])
        heatmap[y0:y1, x0:x1] = float(value)
    return write_preview_png(heatmap, path)


def write_same_grid_outputs(
    per_patch: pd.DataFrame,
    per_image: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    per_patch.to_csv(paths["per_patch"], index=False)
    per_image.to_csv(paths["per_image"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_same_grid_summary_text(summary), encoding="utf-8")


def render_same_grid_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal same-grid OOP/orientation analysis",
        f"candidate_source: {summary['candidate_source']}",
        f"same_grid_patch_rows: {summary['same_grid_patch_rows']}",
        f"patches_processed_ok: {summary['patches_processed_ok']}",
        f"patches_error: {summary['patches_error']}",
        f"candidate_patch_count: {summary['candidate_patch_count']}",
        f"spacing_status: {summary['spacing_status']}",
        "",
        f"Selected-vs-all OOP summary: {summary['selected_vs_all_oop_summary']}",
        "",
        "Special image summaries:",
    ]
    for row in summary["selected_region_summaries"]:
        lines.append(
            f"- {row.get('filename')}: selected_oop={row.get('selected_region_median_oop_128')}, "
            f"all_oop={row.get('all_region_median_oop_128')}, diff={row.get('selected_vs_all_oop_difference_128')}, "
            f"candidate_fraction={row.get('candidate_patch_fraction')}, flag={row.get('interpretation_flag')}"
        )
    lines.append("")
    lines.extend(summary["interpretation"])
    return "\n".join(lines) + "\n"


def safe_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(np.median(numeric))


def safe_iqr(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    q75, q25 = np.percentile(numeric, [75, 25])
    return float(q75 - q25)


def difference_or_nan(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return float("nan")
    return float(left - right)
