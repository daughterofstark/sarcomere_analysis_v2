from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .confocal_intake import load_confocal_image_2d
from .config import output_dir
from .masking import compute_tissue_mask
from .orientation import orientation_params, orientation_weights, structure_tensor_orientation
from .outputs import write_preview_png
from .patches import generate_patch_grid
from .preprocessing import preprocess_image
from .zdisc_annotation import json_safe


STRIATION_PATCH_COLUMNS = [
    "confocal_image_id",
    "filename",
    "patch_id",
    "y0",
    "x0",
    "y1",
    "x1",
    "center_y",
    "center_x",
    "tissue_fraction",
    "signal_fraction",
    "gradient_energy",
    "orientation_coherence",
    "intensity_mean",
    "intensity_std",
    "contrast",
    "saturation_fraction",
    "candidate_striation_region",
    "candidate_reason",
    "rejection_reason",
    "expected_positive_example",
    "noted_complex_example",
]

STRIATION_IMAGE_COLUMNS = [
    "confocal_image_id",
    "filename",
    "total_patches",
    "candidate_patch_count",
    "candidate_patch_fraction",
    "median_candidate_coherence",
    "median_candidate_gradient_energy",
    "median_candidate_intensity_std",
    "expected_positive_example",
    "noted_complex_example",
    "processing_status",
    "error_message",
]


def default_striation_mask_paths(cfg: dict[str, Any], output_directory: str | Path | None = None) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_striation_mask"
    return {
        "root": root,
        "per_patch": root / "confocal_striation_mask_per_patch.csv",
        "per_image": root / "confocal_striation_mask_per_image.csv",
        "summary_json": root / "confocal_striation_mask_summary.json",
        "summary_txt": root / "confocal_striation_mask_summary.txt",
        "previews": root / "previews",
    }


def run_confocal_striation_mask_audit(
    cfg: dict[str, Any],
    confocal_manifest: str | Path,
    output_directory: str | Path | None = None,
    write_previews: bool = False,
    patch_size: int | None = None,
    stride: int | None = None,
    min_gradient_energy: float | None = None,
    min_orientation_coherence: float | None = None,
    min_intensity_std: float | None = None,
    max_saturation_fraction: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    manifest = pd.read_csv(confocal_manifest, dtype={"confocal_image_id": str, "filename": str, "source_path": str})
    params = striation_mask_params(
        cfg,
        patch_size=patch_size,
        stride=stride,
        min_gradient_energy=min_gradient_energy,
        min_orientation_coherence=min_orientation_coherence,
        min_intensity_std=min_intensity_std,
        max_saturation_fraction=max_saturation_fraction,
    )
    run_cfg = confocal_patch_config(cfg, params)
    paths = default_striation_mask_paths(cfg, output_directory)
    image_rows: list[dict[str, Any]] = []
    patch_tables: list[pd.DataFrame] = []
    preview_paths: list[str] = []

    for _, row in manifest.iterrows():
        image_row, patch_table, previews = audit_confocal_striation_image(row, run_cfg, params, paths["previews"], write_previews)
        image_rows.append(image_row)
        if patch_table is not None and not patch_table.empty:
            patch_tables.append(patch_table)
        preview_paths.extend(str(path) for path in previews)

    per_patch = pd.concat(patch_tables, ignore_index=True) if patch_tables else pd.DataFrame(columns=STRIATION_PATCH_COLUMNS)
    per_image = pd.DataFrame(image_rows, columns=STRIATION_IMAGE_COLUMNS)
    summary = build_striation_mask_summary(manifest, per_image, per_patch, params, preview_paths, write_previews)
    write_striation_mask_outputs(per_patch, per_image, summary, paths)
    return per_patch, per_image, summary, paths


def striation_mask_params(
    cfg: dict[str, Any],
    patch_size: int | None = None,
    stride: int | None = None,
    min_gradient_energy: float | None = None,
    min_orientation_coherence: float | None = None,
    min_intensity_std: float | None = None,
    max_saturation_fraction: float | None = None,
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "patch_size_px": 128,
        "stride_px": 64,
        "min_signal_fraction": 0.05,
        "min_gradient_energy": 0.0002,
        "min_orientation_coherence": 0.20,
        "min_intensity_std": 0.03,
        "max_saturation_fraction": 0.10,
    }
    params = dict(defaults)
    params.update(cfg.get("confocal_striation_mask", {}))
    overrides = {
        "patch_size_px": patch_size,
        "stride_px": stride,
        "min_gradient_energy": min_gradient_energy,
        "min_orientation_coherence": min_orientation_coherence,
        "min_intensity_std": min_intensity_std,
        "max_saturation_fraction": max_saturation_fraction,
    }
    params.update({key: value for key, value in overrides.items() if value is not None})
    params["patch_size_px"] = int(params["patch_size_px"])
    params["stride_px"] = int(params["stride_px"])
    for key in [
        "min_signal_fraction",
        "min_gradient_energy",
        "min_orientation_coherence",
        "min_intensity_std",
        "max_saturation_fraction",
    ]:
        params[key] = float(params[key])
    if params["patch_size_px"] <= 0 or params["stride_px"] <= 0:
        raise ValueError("Confocal striation mask patch size and stride must be positive")
    if not 0.0 <= params["min_signal_fraction"] <= 1.0:
        raise ValueError("min_signal_fraction must be between 0 and 1")
    if not 0.0 <= params["min_orientation_coherence"] <= 1.0:
        raise ValueError("min_orientation_coherence must be between 0 and 1")
    if not 0.0 <= params["max_saturation_fraction"] <= 1.0:
        raise ValueError("max_saturation_fraction must be between 0 and 1")
    return params


def confocal_patch_config(cfg: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    run_cfg = copy.deepcopy(cfg)
    run_cfg["patches"] = {
        "patch_size_px": int(params["patch_size_px"]),
        "stride_px": int(params["stride_px"]),
        "margin_px": 0,
    }
    return run_cfg


def audit_confocal_striation_image(
    manifest_row: pd.Series,
    cfg: dict[str, Any],
    params: dict[str, Any],
    preview_dir: Path,
    write_previews: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None, list[Path]]:
    image_id = str(manifest_row["confocal_image_id"])
    filename = str(manifest_row["filename"])
    expected_positive = bool(str(manifest_row.get("expected_positive_example", "")).lower() == "true" or manifest_row.get("expected_positive_example") is True)
    noted_complex = bool(str(manifest_row.get("noted_complex_example", "")).lower() == "true" or manifest_row.get("noted_complex_example") is True)
    previews: list[Path] = []
    try:
        raw, _ = load_confocal_image_2d(str(manifest_row["source_path"]))
        preprocessed = preprocess_image(raw, cfg)
        tissue = compute_tissue_mask(preprocessed.image, cfg)
        orientation_map, coherence_map, energy_map = structure_tensor_orientation(preprocessed.image, orientation_params(cfg))
        _ = orientation_weights(energy_map, coherence_map, str(orientation_params(cfg)["weight_mode"]))
        patch_table = striation_patch_table(
            preprocessed.image,
            tissue.mask,
            coherence_map,
            energy_map,
            image_id,
            filename,
            expected_positive,
            noted_complex,
            cfg,
            params,
        )
        image_row = summarize_striation_image(patch_table, image_id, filename, expected_positive, noted_complex, "ok", "")
        if write_previews:
            previews = write_striation_previews(image_id, preprocessed.image, patch_table, preview_dir)
        return image_row, patch_table, previews
    except Exception as exc:  # pragma: no cover - protects real-world intake from odd image files.
        image_row = summarize_striation_image(
            pd.DataFrame(columns=STRIATION_PATCH_COLUMNS),
            image_id,
            filename,
            expected_positive,
            noted_complex,
            "error",
            str(exc),
        )
        return image_row, None, previews


def striation_patch_table(
    image: np.ndarray,
    tissue_mask: np.ndarray,
    coherence_map: np.ndarray,
    energy_map: np.ndarray,
    image_id: str,
    filename: str,
    expected_positive: bool,
    noted_complex: bool,
    cfg: dict[str, Any],
    params: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patch in generate_patch_grid(image.shape, image_id, cfg):
        image_patch = image[patch.y0 : patch.y1, patch.x0 : patch.x1]
        tissue_patch = tissue_mask[patch.y0 : patch.y1, patch.x0 : patch.x1]
        coherence_patch = coherence_map[patch.y0 : patch.y1, patch.x0 : patch.x1]
        energy_patch = energy_map[patch.y0 : patch.y1, patch.x0 : patch.x1]
        metrics = patch_striation_metrics(image_patch, tissue_patch, coherence_patch, energy_patch)
        decision = candidate_decision(metrics, params)
        rows.append(
            {
                "confocal_image_id": image_id,
                "filename": filename,
                "patch_id": patch.patch_id,
                "y0": patch.y0,
                "x0": patch.x0,
                "y1": patch.y1,
                "x1": patch.x1,
                "center_y": patch.center_y,
                "center_x": patch.center_x,
                **metrics,
                **decision,
                "expected_positive_example": bool(expected_positive),
                "noted_complex_example": bool(noted_complex),
            }
        )
    return pd.DataFrame(rows, columns=STRIATION_PATCH_COLUMNS)


def patch_striation_metrics(
    image_patch: np.ndarray,
    tissue_patch: np.ndarray,
    coherence_patch: np.ndarray,
    energy_patch: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(image_patch, dtype=np.float32)
    tissue = np.asarray(tissue_patch, dtype=bool)
    if values.size == 0:
        return {
            "tissue_fraction": 0.0,
            "signal_fraction": 0.0,
            "gradient_energy": 0.0,
            "orientation_coherence": np.nan,
            "intensity_mean": np.nan,
            "intensity_std": np.nan,
            "contrast": np.nan,
            "saturation_fraction": 0.0,
        }
    signal_fraction = float(np.mean(tissue))
    measured = values[tissue] if np.any(tissue) else values.ravel()
    finite = measured[np.isfinite(measured)]
    if finite.size == 0:
        finite = values[np.isfinite(values)]
    coherence_values = coherence_patch[tissue] if np.any(tissue) else coherence_patch.ravel()
    energy_values = energy_patch[tissue] if np.any(tissue) else energy_patch.ravel()
    p5, p95 = (np.nan, np.nan) if finite.size == 0 else np.percentile(finite, [5, 95])
    return {
        "tissue_fraction": signal_fraction,
        "signal_fraction": signal_fraction,
        "gradient_energy": float(np.nanmedian(energy_values)) if energy_values.size else 0.0,
        "orientation_coherence": float(np.nanmedian(coherence_values)) if coherence_values.size else np.nan,
        "intensity_mean": float(np.nanmean(finite)) if finite.size else np.nan,
        "intensity_std": float(np.nanstd(finite)) if finite.size else np.nan,
        "contrast": float(p95 - p5) if np.isfinite(p5) and np.isfinite(p95) else np.nan,
        "saturation_fraction": float(np.mean(values >= 0.98)),
    }


def candidate_decision(metrics: dict[str, float], params: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if not np.isfinite(metrics["intensity_mean"]):
        reasons.append("empty_patch")
    if metrics["signal_fraction"] < params["min_signal_fraction"]:
        reasons.append("low_signal_fraction")
    if metrics["gradient_energy"] < params["min_gradient_energy"]:
        reasons.append("low_gradient_energy")
    if not np.isfinite(metrics["orientation_coherence"]) or metrics["orientation_coherence"] < params["min_orientation_coherence"]:
        reasons.append("low_orientation_coherence")
    if not np.isfinite(metrics["intensity_std"]) or metrics["intensity_std"] < params["min_intensity_std"]:
        reasons.append("low_intensity_std")
    if metrics["saturation_fraction"] > params["max_saturation_fraction"]:
        reasons.append("high_saturation_fraction")
    candidate = len(reasons) == 0
    return {
        "candidate_striation_region": bool(candidate),
        "candidate_reason": "passes_confident_striation_candidate_gate" if candidate else "",
        "rejection_reason": "ok" if candidate else ";".join(reasons),
    }


def summarize_striation_image(
    patch_table: pd.DataFrame,
    image_id: str,
    filename: str,
    expected_positive: bool,
    noted_complex: bool,
    status: str,
    error_message: str,
) -> dict[str, Any]:
    total = int(len(patch_table))
    candidates = patch_table.loc[patch_table["candidate_striation_region"].fillna(False).astype(bool)] if total else pd.DataFrame()
    return {
        "confocal_image_id": image_id,
        "filename": filename,
        "total_patches": total,
        "candidate_patch_count": int(len(candidates)),
        "candidate_patch_fraction": float(len(candidates) / total) if total else 0.0,
        "median_candidate_coherence": safe_median(candidates.get("orientation_coherence", pd.Series(dtype=float))),
        "median_candidate_gradient_energy": safe_median(candidates.get("gradient_energy", pd.Series(dtype=float))),
        "median_candidate_intensity_std": safe_median(candidates.get("intensity_std", pd.Series(dtype=float))),
        "expected_positive_example": bool(expected_positive),
        "noted_complex_example": bool(noted_complex),
        "processing_status": status,
        "error_message": error_message,
    }


def safe_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(np.median(numeric))


def write_striation_previews(image_id: str, image: np.ndarray, patch_table: pd.DataFrame, preview_dir: Path) -> list[Path]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    candidate_mask = candidate_patch_mask(image.shape, patch_table)
    return [
        write_preview_png(image, preview_dir / f"{image_id}_normalized.png"),
        write_candidate_overlay(image, candidate_mask, preview_dir / f"{image_id}_candidate_mask_overlay.png"),
        write_candidate_patch_grid_overlay(image, patch_table, preview_dir / f"{image_id}_candidate_patch_grid.png"),
        write_rejection_map(image, patch_table, preview_dir / f"{image_id}_rejection_map.png"),
    ]


def candidate_patch_mask(image_shape: tuple[int, int], patch_table: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=bool)
    for _, row in patch_table.iterrows():
        if bool(row.get("candidate_striation_region", False)):
            mask[int(row["y0"]) : int(row["y1"]), int(row["x0"]) : int(row["x1"])] = True
    return mask


def write_candidate_overlay(image: np.ndarray, candidate_mask: np.ndarray, path: str | Path) -> Path:
    display = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    rgb = np.dstack([display, display, display])
    alpha = 0.40
    color = np.array([1.0, 0.1, 0.1], dtype=np.float32)
    rgb[candidate_mask] = (1.0 - alpha) * rgb[candidate_mask] + alpha * color
    return write_preview_png(rgb, path)


def write_candidate_patch_grid_overlay(image: np.ndarray, patch_table: pd.DataFrame, path: str | Path) -> Path:
    display = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    rgb = np.dstack([display, display, display])
    for _, row in patch_table.iterrows():
        y0, y1, x0, x1 = int(row["y0"]), int(row["y1"]), int(row["x0"]), int(row["x1"])
        color = np.array([1.0, 0.1, 0.1]) if bool(row.get("candidate_striation_region", False)) else np.array([0.2, 0.5, 1.0])
        rgb[y0:y1, x0] = color
        rgb[y0:y1, x1 - 1] = color
        rgb[y0, x0:x1] = color
        rgb[y1 - 1, x0:x1] = color
    return write_preview_png(rgb, path)


def write_rejection_map(image: np.ndarray, patch_table: pd.DataFrame, path: str | Path) -> Path:
    display = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    rgb = np.dstack([display, display, display])
    colors = {
        "low_signal_fraction": np.array([0.1, 0.1, 0.8]),
        "low_gradient_energy": np.array([0.8, 0.8, 0.1]),
        "low_orientation_coherence": np.array([0.8, 0.2, 0.8]),
        "low_intensity_std": np.array([0.1, 0.8, 0.8]),
        "high_saturation_fraction": np.array([1.0, 0.5, 0.0]),
        "ok": np.array([1.0, 0.1, 0.1]),
    }
    alpha = 0.25
    for _, row in patch_table.iterrows():
        reason = str(row.get("rejection_reason", "ok")).split(";")[0]
        color = colors.get(reason, np.array([0.7, 0.7, 0.7]))
        if bool(row.get("candidate_striation_region", False)):
            color = colors["ok"]
        y0, y1, x0, x1 = int(row["y0"]), int(row["y1"]), int(row["x0"]), int(row["x1"])
        rgb[y0:y1, x0:x1] = (1.0 - alpha) * rgb[y0:y1, x0:x1] + alpha * color
    return write_preview_png(rgb, path)


def build_striation_mask_summary(
    manifest: pd.DataFrame,
    per_image: pd.DataFrame,
    per_patch: pd.DataFrame,
    params: dict[str, Any],
    preview_paths: list[str],
    write_previews: bool,
) -> dict[str, Any]:
    expected = per_image.loc[per_image["expected_positive_example"].fillna(False).astype(bool)] if not per_image.empty else pd.DataFrame()
    complex_examples = per_image.loc[per_image["noted_complex_example"].fillna(False).astype(bool)] if not per_image.empty else pd.DataFrame()
    total_patches = int(len(per_patch))
    candidate_count = int(per_patch["candidate_striation_region"].fillna(False).astype(bool).sum()) if not per_patch.empty else 0
    fractions = pd.to_numeric(per_image.get("candidate_patch_fraction", pd.Series(dtype=float)), errors="coerce")
    return json_safe(
        {
            "mode": "confocal_confident_striation_candidate_mask_audit",
            "confocal_image_count": int(len(manifest)),
            "processed_ok": int((per_image["processing_status"] == "ok").sum()) if not per_image.empty else 0,
            "processed_error": int((per_image["processing_status"] == "error").sum()) if not per_image.empty else 0,
            "total_patches": total_patches,
            "candidate_patch_count": candidate_count,
            "candidate_patch_fraction": float(candidate_count / total_patches) if total_patches else 0.0,
            "candidate_fraction_median_by_image": safe_median(fractions),
            "images_with_extremely_low_candidate_fraction": per_image.loc[fractions < 0.01, "confocal_image_id"].astype(str).tolist() if not per_image.empty else [],
            "images_with_extremely_high_candidate_fraction": per_image.loc[fractions > 0.80, "confocal_image_id"].astype(str).tolist() if not per_image.empty else [],
            "expected_positive_examples": expected[
                ["confocal_image_id", "filename", "candidate_patch_count", "candidate_patch_fraction"]
            ].to_dict("records")
            if not expected.empty
            else [],
            "noted_complex_examples": complex_examples[
                ["confocal_image_id", "filename", "candidate_patch_count", "candidate_patch_fraction"]
            ].to_dict("records")
            if not complex_examples.empty
            else [],
            "top_rejection_reasons": rejection_reason_counts(per_patch),
            "parameters": params,
            "previews_written": bool(write_previews),
            "preview_paths": preview_paths,
            "spacing_status": "not_computed_in_microns_confocal_pixel_size_unknown",
            "interpretation": [
                "Exploratory confocal confident-region candidate mask only.",
                "Not a validated Z-disc segmentation or biological endpoint.",
                "Needed because the widefield QC gate did not transfer to confocal images.",
                "Candidate regions use local signal, contrast, gradient energy, and orientation coherence.",
                "6052 and 5138 are reported as expected positive examples but are not hard-coded to pass.",
                "3112 is reported separately as a complex image with possible non-striated Z-disc-like structures.",
                "No spacing in microns is reported without confocal pixel calibration.",
                "No biological claims are made.",
            ],
        }
    )


def rejection_reason_counts(per_patch: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
    if per_patch.empty or "rejection_reason" not in per_patch.columns:
        return []
    exploded: list[str] = []
    for value in per_patch["rejection_reason"].fillna("").astype(str):
        if value == "ok":
            continue
        exploded.extend(reason for reason in value.split(";") if reason)
    counts = pd.Series(exploded, dtype=str).value_counts().head(n)
    return [{"reason": str(reason), "count": int(count)} for reason, count in counts.items()]


def write_striation_mask_outputs(
    per_patch: pd.DataFrame,
    per_image: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    per_patch.to_csv(paths["per_patch"], index=False)
    per_image.to_csv(paths["per_image"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_striation_mask_summary_text(summary), encoding="utf-8")


def render_striation_mask_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal confident-striation candidate mask audit",
        f"confocal_image_count: {summary['confocal_image_count']}",
        f"processed_ok: {summary['processed_ok']}",
        f"processed_error: {summary['processed_error']}",
        f"total_patches: {summary['total_patches']}",
        f"candidate_patch_count: {summary['candidate_patch_count']}",
        f"candidate_patch_fraction: {summary['candidate_patch_fraction']}",
        f"spacing_status: {summary['spacing_status']}",
        "",
        "Expected positive examples:",
    ]
    lines.extend(
        f"- {row.get('filename')}: {row.get('candidate_patch_count')} candidates ({row.get('candidate_patch_fraction')})"
        for row in summary["expected_positive_examples"]
    )
    lines.append("")
    lines.append("Noted complex examples:")
    lines.extend(
        f"- {row.get('filename')}: {row.get('candidate_patch_count')} candidates ({row.get('candidate_patch_fraction')})"
        for row in summary["noted_complex_examples"]
    )
    lines.append("")
    lines.append("Top rejection reasons:")
    lines.extend(f"- {row.get('reason')}: {row.get('count')}" for row in summary["top_rejection_reasons"])
    lines.append("")
    lines.append("Interpretation:")
    lines.extend(f"- {item}" for item in summary["interpretation"])
    return "\n".join(lines) + "\n"
