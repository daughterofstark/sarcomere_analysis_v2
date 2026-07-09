from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image

from .config import output_dir
from .schemas import (
    IMAGE_METRICS_COLUMNS,
    PATCH_METRICS_COLUMNS,
    stabilize_columns,
)


PATCH_METRICS_REQUIRED_COLUMNS = PATCH_METRICS_COLUMNS
IMAGE_METRICS_REQUIRED_COLUMNS = IMAGE_METRICS_COLUMNS


def ensure_output_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    root = output_dir(cfg)
    dirs = {
        "root": root,
        "tables": root / "tables",
        "previews": root / "previews",
        "provenance": root / "provenance",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def write_patch_metrics(df: pd.DataFrame, image_id: str, cfg: dict[str, Any]) -> Path:
    path = ensure_output_dirs(cfg)["tables"] / f"{image_id}_per_patch_metrics.csv"
    ordered = stabilize_columns(df, PATCH_METRICS_COLUMNS, required_core=["image_id", "patch_id"])
    ordered.to_csv(path, index=False)
    return path


def write_image_metrics(mapping_or_df: Mapping[str, Any] | pd.DataFrame, image_id: str, cfg: dict[str, Any]) -> Path:
    path = ensure_output_dirs(cfg)["tables"] / f"{image_id}_per_image_metrics.csv"
    if isinstance(mapping_or_df, pd.DataFrame):
        df = mapping_or_df.copy()
    else:
        df = pd.DataFrame([dict(mapping_or_df)])
    ordered = stabilize_columns(df, IMAGE_METRICS_COLUMNS, required_core=["image_id"])
    ordered.to_csv(path, index=False)
    return path


def write_preview_png(array_or_rgb: np.ndarray, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(array_or_rgb)
    if values.ndim == 2:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            scaled = np.zeros(values.shape, dtype=np.float32)
        else:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            scaled = _scale_unit(values, vmin, vmax)
        png = (np.nan_to_num(scaled, nan=0.0) * 255).astype(np.uint8)
        Image.fromarray(png, mode="L").save(out)
    elif values.ndim == 3 and values.shape[2] in {3, 4}:
        png = (np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0) * 255).astype(np.uint8)
        mode = "RGBA" if values.shape[2] == 4 else "RGB"
        Image.fromarray(png, mode=mode).save(out)
    else:
        raise ValueError(f"Expected 2D grayscale or 3/4-channel RGB(A), got shape {values.shape}")
    return out


def write_mask_overlay(image: np.ndarray, mask: np.ndarray, path: str | Path) -> Path:
    display = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    rgb = np.dstack([display, display, display])
    overlay_color = np.array([0.0, 0.85, 0.25], dtype=np.float32)
    alpha = 0.35
    mask_bool = np.asarray(mask, dtype=bool)
    rgb[mask_bool] = (1.0 - alpha) * rgb[mask_bool] + alpha * overlay_color
    return write_preview_png(rgb, path)


def write_heatmap(
    values: str | np.ndarray,
    patch_table: pd.DataFrame,
    image_shape: tuple[int, int],
    path: str | Path,
    cfg: dict[str, Any],
) -> Path:
    _ = cfg
    heatmap = np.full(image_shape, np.nan, dtype=np.float32)
    if isinstance(values, str):
        value_series = patch_table[values]
    else:
        value_series = pd.Series(values)
    for (_, row), value in zip(patch_table.iterrows(), value_series):
        if np.isfinite(value):
            heatmap[int(row["y0"]) : int(row["y1"]), int(row["x0"]) : int(row["x1"])] = float(value)
    return write_preview_png(heatmap, path)


def preview_paths(image_id: str, cfg: dict[str, Any]) -> dict[str, Path]:
    previews = ensure_output_dirs(cfg)["previews"]
    return {
        "tissue_mask_overlay": previews / f"{image_id}_tissue_mask_overlay.png",
        "orientation": previews / f"{image_id}_orientation.png",
        "coherence": previews / f"{image_id}_coherence.png",
        "oop_heatmap": previews / f"{image_id}_oop_heatmap.png",
        "spacing_heatmap": previews / f"{image_id}_spacing_heatmap.png",
    }


def _scale_unit(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    span = vmax - vmin
    if not np.isfinite(span) or span <= 0:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values.astype(np.float32, copy=False) - vmin) / span, 0.0, 1.0)
