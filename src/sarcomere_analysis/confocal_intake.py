from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import tifffile

from .config import output_dir
from .masking import compute_tissue_mask
from .orientation import compute_orientation_analysis
from .outputs import write_heatmap, write_mask_overlay, write_preview_png
from .preprocessing import preprocess_image
from .qc import compute_patch_qc
from .zdisc_annotation import json_safe


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

CONFOCAL_MANIFEST_COLUMNS = [
    "confocal_image_id",
    "filename",
    "source_path",
    "extension",
    "image_shape_y",
    "image_shape_x",
    "dtype",
    "inferred_sample_id",
    "expected_positive_example",
    "noted_complex_example",
    "notes",
]

CONFOCAL_IMAGE_COLUMNS = [
    "confocal_image_id",
    "filename",
    "shape",
    "dtype",
    "intensity_min",
    "intensity_max",
    "intensity_mean",
    "intensity_std",
    "tissue_fraction",
    "valid_orientation_patch_count",
    "total_patch_count",
    "valid_orientation_patch_fraction",
    "image_oop",
    "image_orientation_heterogeneity",
    "spacing_valid_patch_count",
    "spacing_valid_patch_fraction",
    "spacing_status",
    "processing_status",
    "error_message",
]


def default_confocal_output_paths(cfg: dict[str, Any], output_directory: str | Path | None = None) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_baseline"
    return {
        "root": root,
        "manifest": root / "confocal_manifest.csv",
        "per_image": root / "confocal_baseline_per_image.csv",
        "per_patch": root / "confocal_baseline_per_patch.csv",
        "summary_json": root / "confocal_baseline_summary.json",
        "summary_txt": root / "confocal_baseline_summary.txt",
        "previews": root / "previews",
    }


def run_confocal_baseline_audit(
    cfg: dict[str, Any],
    confocal_root: str | Path,
    output_directory: str | Path | None = None,
    write_previews: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    paths = default_confocal_output_paths(cfg, output_directory)
    manifest = build_confocal_manifest(confocal_root)
    image_rows: list[dict[str, Any]] = []
    patch_tables: list[pd.DataFrame] = []
    preview_paths: list[str] = []

    for _, row in manifest.iterrows():
        image_row, patch_table, previews = audit_confocal_image(row, cfg, paths["previews"], write_previews=write_previews)
        image_rows.append(image_row)
        if patch_table is not None and not patch_table.empty:
            patch_tables.append(patch_table)
        preview_paths.extend(str(path) for path in previews)

    per_image = pd.DataFrame(image_rows, columns=CONFOCAL_IMAGE_COLUMNS)
    per_patch = pd.concat(patch_tables, ignore_index=True) if patch_tables else pd.DataFrame()
    summary = build_confocal_summary(manifest, per_image, per_patch, preview_paths, write_previews)
    write_confocal_outputs(manifest, per_image, per_patch, summary, paths)
    return manifest, per_image, per_patch, summary, paths


def discover_confocal_images(confocal_root: str | Path) -> list[Path]:
    root = Path(confocal_root)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_confocal_manifest(confocal_root: str | Path) -> pd.DataFrame:
    rows = []
    for path in discover_confocal_images(confocal_root):
        metadata = read_image_metadata(path)
        stem = path.stem
        sample_id = infer_sample_id(stem)
        lower_name = path.name.lower()
        notes = list(metadata["notes"])
        if "6052" in lower_name or "5138" in lower_name:
            notes.append("expected_positive_example_from_natalia")
        if "3112" in lower_name:
            notes.append("noted_complex_example_from_natalia")
        rows.append(
            {
                "confocal_image_id": stem,
                "filename": path.name,
                "source_path": str(path),
                "extension": path.suffix.lower(),
                "image_shape_y": metadata["image_shape_y"],
                "image_shape_x": metadata["image_shape_x"],
                "dtype": metadata["dtype"],
                "inferred_sample_id": sample_id,
                "expected_positive_example": bool("6052" in lower_name or "5138" in lower_name),
                "noted_complex_example": bool("3112" in lower_name),
                "notes": ";".join(notes),
            }
        )
    return pd.DataFrame(rows, columns=CONFOCAL_MANIFEST_COLUMNS)


def read_image_metadata(path: str | Path) -> dict[str, Any]:
    array, notes = load_confocal_image_2d(path)
    return {
        "image_shape_y": int(array.shape[0]),
        "image_shape_x": int(array.shape[1]),
        "dtype": str(array.dtype),
        "notes": notes,
    }


def infer_sample_id(name: str) -> str:
    digits = "".join(ch if ch.isdigit() else " " for ch in str(name)).split()
    return digits[0] if digits else ""


def load_confocal_image_2d(path: str | Path) -> tuple[np.ndarray, list[str]]:
    path = Path(path)
    notes: list[str] = []
    if path.suffix.lower() in {".tif", ".tiff"}:
        raw = tifffile.imread(path)
    else:
        with Image.open(path) as image:
            raw = np.asarray(image)
    array = np.asarray(raw)
    original_shape = array.shape
    array = np.squeeze(array)
    if array.ndim == 2:
        return array, notes
    if array.ndim == 3 and array.shape[-1] in {3, 4}:
        rgb = array[..., :3].astype(np.float32, copy=False)
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        notes.append(f"converted_rgb_to_grayscale_from_shape_{original_shape}")
        return gray.astype(array.dtype if np.issubdtype(array.dtype, np.integer) else np.float32), notes
    if array.ndim == 3:
        projected = np.max(array, axis=0)
        notes.append(f"max_projected_3d_stack_from_shape_{original_shape}")
        return projected, notes
    raise ValueError(f"Unsupported confocal image shape after squeeze: {array.shape} for {path}")


def audit_confocal_image(
    manifest_row: pd.Series,
    cfg: dict[str, Any],
    preview_dir: Path,
    write_previews: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None, list[Path]]:
    image_id = str(manifest_row["confocal_image_id"])
    filename = str(manifest_row["filename"])
    path = Path(str(manifest_row["source_path"]))
    previews: list[Path] = []
    try:
        raw, load_notes = load_confocal_image_2d(path)
        raw_stats = intensity_stats(raw)
        preprocessed = preprocess_image(raw, cfg)
        tissue = compute_tissue_mask(preprocessed.image, cfg)
        patch_qc = compute_patch_qc(preprocessed.image, tissue.mask, image_id, cfg)
        orientation = compute_orientation_analysis(preprocessed.image, tissue.mask, patch_qc, cfg)
        patch_table = orientation.patch_metrics.copy()
        if not patch_table.empty:
            patch_table = patch_table.rename(columns={"image_id": "confocal_image_id"})
            patch_table.insert(1, "filename", filename)
        total_patches = int(len(patch_qc))
        valid_orientation = int(patch_qc["valid_for_orientation"].sum()) if "valid_for_orientation" in patch_qc else 0
        row = {
            "confocal_image_id": image_id,
            "filename": filename,
            "shape": f"{raw.shape[0]}x{raw.shape[1]}",
            "dtype": str(raw.dtype),
            **raw_stats,
            "tissue_fraction": tissue.metadata.get("tissue_fraction", float(np.mean(tissue.mask))),
            "valid_orientation_patch_count": valid_orientation,
            "total_patch_count": total_patches,
            "valid_orientation_patch_fraction": valid_orientation / total_patches if total_patches else 0.0,
            "image_oop": orientation.image_metrics.get("image_oop"),
            "image_orientation_heterogeneity": orientation.image_metrics.get("image_oop_heterogeneity"),
            "spacing_valid_patch_count": 0,
            "spacing_valid_patch_fraction": 0.0,
            "spacing_status": "not_computed_missing_confocal_pixel_size",
            "processing_status": "ok",
            "error_message": "",
        }
        if write_previews:
            previews = write_confocal_previews(image_id, preprocessed.image, tissue.mask, patch_table, orientation, preview_dir, cfg)
        if load_notes and patch_table is not None and not patch_table.empty:
            patch_table["confocal_load_notes"] = ";".join(load_notes)
        return row, patch_table, previews
    except Exception as exc:  # pragma: no cover - exercised through CLI behaviour on unexpected files.
        row = {
            "confocal_image_id": image_id,
            "filename": filename,
            "shape": "",
            "dtype": str(manifest_row.get("dtype", "")),
            "intensity_min": np.nan,
            "intensity_max": np.nan,
            "intensity_mean": np.nan,
            "intensity_std": np.nan,
            "tissue_fraction": np.nan,
            "valid_orientation_patch_count": 0,
            "total_patch_count": 0,
            "valid_orientation_patch_fraction": 0.0,
            "image_oop": np.nan,
            "image_orientation_heterogeneity": np.nan,
            "spacing_valid_patch_count": 0,
            "spacing_valid_patch_fraction": 0.0,
            "spacing_status": "not_computed_missing_confocal_pixel_size",
            "processing_status": "error",
            "error_message": str(exc),
        }
        return row, None, previews


def intensity_stats(raw: np.ndarray) -> dict[str, float]:
    values = np.asarray(raw, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"intensity_min": np.nan, "intensity_max": np.nan, "intensity_mean": np.nan, "intensity_std": np.nan}
    return {
        "intensity_min": float(np.min(finite)),
        "intensity_max": float(np.max(finite)),
        "intensity_mean": float(np.mean(finite)),
        "intensity_std": float(np.std(finite)),
    }


def write_confocal_previews(
    image_id: str,
    image: np.ndarray,
    tissue_mask: np.ndarray,
    patch_table: pd.DataFrame,
    orientation: Any,
    preview_dir: Path,
    cfg: dict[str, Any],
) -> list[Path]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    written = [
        write_preview_png(image, preview_dir / f"{image_id}_display_normalized.png"),
        write_mask_overlay(image, tissue_mask, preview_dir / f"{image_id}_tissue_mask_overlay.png"),
        write_preview_png(orientation.coherence_map, preview_dir / f"{image_id}_coherence.png"),
        write_preview_png(orientation.orientation_map, preview_dir / f"{image_id}_orientation.png"),
    ]
    if not patch_table.empty and "patch_oop" in patch_table:
        written.append(write_heatmap("patch_oop", patch_table, image.shape, preview_dir / f"{image_id}_oop_heatmap.png", cfg))
        written.append(write_patch_grid_overlay(image, patch_table, preview_dir / f"{image_id}_valid_orientation_patch_grid.png"))
    return written


def write_patch_grid_overlay(image: np.ndarray, patch_table: pd.DataFrame, path: str | Path) -> Path:
    display = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    rgb = np.dstack([display, display, display])
    for _, row in patch_table.iterrows():
        y0, y1, x0, x1 = int(row["y0"]), int(row["y1"]), int(row["x0"]), int(row["x1"])
        color = np.array([0.0, 0.85, 0.25]) if bool(row.get("valid_for_orientation", False)) else np.array([1.0, 0.2, 0.1])
        rgb[y0:y1, x0] = color
        rgb[y0:y1, x1 - 1] = color
        rgb[y0, x0:x1] = color
        rgb[y1 - 1, x0:x1] = color
    return write_preview_png(rgb, path)


def build_confocal_summary(
    manifest: pd.DataFrame,
    per_image: pd.DataFrame,
    per_patch: pd.DataFrame,
    preview_paths: list[str],
    write_previews: bool,
) -> dict[str, Any]:
    ok = per_image.loc[per_image["processing_status"] == "ok"] if not per_image.empty else pd.DataFrame()
    positive = manifest.loc[manifest["expected_positive_example"].fillna(False)] if not manifest.empty else pd.DataFrame()
    complex_examples = manifest.loc[manifest["noted_complex_example"].fillna(False)] if not manifest.empty else pd.DataFrame()
    return json_safe(
        {
            "mode": "confocal_baseline_audit",
            "confocal_image_count": int(len(manifest)),
            "processed_ok": int(len(ok)),
            "processed_error": int((per_image["processing_status"] == "error").sum()) if not per_image.empty else 0,
            "filenames": manifest["filename"].astype(str).tolist() if not manifest.empty else [],
            "expected_positive_examples": positive[["confocal_image_id", "filename"]].to_dict("records") if not positive.empty else [],
            "noted_complex_examples": complex_examples[["confocal_image_id", "filename"]].to_dict("records") if not complex_examples.empty else [],
            "patch_rows": int(len(per_patch)),
            "orientation_summary": {
                "image_oop_median": safe_median(ok.get("image_oop", pd.Series(dtype=float))),
                "valid_orientation_patch_fraction_median": safe_median(ok.get("valid_orientation_patch_fraction", pd.Series(dtype=float))),
                "valid_orientation_patch_count_total": int(ok.get("valid_orientation_patch_count", pd.Series(dtype=float)).sum()) if not ok.empty else 0,
            },
            "spacing_calibration_status": "confocal_pixel_size_unknown_spacing_um_not_reported",
            "spacing_policy": "Spacing was not computed with widefield calibration; calibrated confocal spacing requires explicit confocal pixel size.",
            "previews_written": bool(write_previews),
            "preview_paths": preview_paths,
            "interpretation": [
                "Baseline transfer audit only.",
                "No confocal-specific optimisation has been performed.",
                "6052 and 5138 are expected positive examples from Natalia.",
                "3112 may contain Z-disc-like structures that do not form striations.",
                "Output is meant to decide whether a confident-striation/Z-disc-region mask is needed.",
                "No biological claims are made.",
            ],
        }
    )


def safe_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(np.median(numeric))


def write_confocal_outputs(
    manifest: pd.DataFrame,
    per_image: pd.DataFrame,
    per_patch: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    manifest.to_csv(paths["manifest"], index=False)
    per_image.to_csv(paths["per_image"], index=False)
    per_patch.to_csv(paths["per_patch"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_confocal_summary_text(summary), encoding="utf-8")


def render_confocal_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal baseline audit",
        f"confocal_image_count: {summary['confocal_image_count']}",
        f"processed_ok: {summary['processed_ok']}",
        f"processed_error: {summary['processed_error']}",
        f"patch_rows: {summary['patch_rows']}",
        f"spacing_calibration_status: {summary['spacing_calibration_status']}",
        "",
        "Expected positive examples:",
    ]
    lines.extend(f"- {row.get('filename')}" for row in summary["expected_positive_examples"])
    lines.append("")
    lines.append("Noted complex examples:")
    lines.extend(f"- {row.get('filename')}" for row in summary["noted_complex_examples"])
    lines.append("")
    lines.append("Interpretation:")
    lines.extend(f"- {item}" for item in summary["interpretation"])
    return "\n".join(lines) + "\n"
