from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .config import output_dir
from .zdisc_annotation import default_zdisc_output_dir, json_safe, shape_string
from .zdisc_draw_ui import load_crop_image, load_draw_index, load_mask


ZDISC_FEATURE_COLUMNS = [
    "annotation_id",
    "image_id",
    "donor_id",
    "patch_id",
    "mask_path",
    "image_path",
    "mask_shape",
    "zdisc_pixel_count",
    "ignore_pixel_count",
    "zdisc_pixel_fraction",
    "ignore_pixel_fraction",
    "has_zdisc_labels",
    "has_ignore_labels",
    "annotation_status",
    "zdisc_component_count",
    "median_component_size",
    "total_component_length_proxy",
    "manual_mask_orientation_deg",
    "manual_mask_orientation_confidence",
    "orientation_estimable",
    "reason_not_estimable",
]


def default_feature_paths(cfg: dict[str, Any], output_directory: str | Path | None = None) -> dict[str, Path]:
    out_dir = Path(output_directory) if output_directory else default_zdisc_output_dir(cfg)
    return {
        "features_csv": out_dir / "zdisc_annotation_features.csv",
        "summary_json": out_dir / "zdisc_annotation_feature_summary.json",
        "summary_txt": out_dir / "zdisc_annotation_feature_summary.txt",
    }


def extract_zdisc_annotation_features(
    cfg: dict[str, Any],
    index_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    min_zdisc_pixels: int = 10,
    min_components: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    index_file = Path(index_path) if index_path else output_dir(cfg) / "zdisc_annotation" / "zdisc_annotation_index.csv"
    index = load_draw_index(index_file)
    rows = [
        extract_one_mask_features(row, min_zdisc_pixels=int(min_zdisc_pixels), min_components=int(min_components))
        for _, row in index.iterrows()
    ]
    features = stabilize_feature_table(pd.DataFrame(rows))
    summary = build_feature_summary(features, index_file, min_zdisc_pixels, min_components)
    paths = default_feature_paths(cfg, output_directory)
    write_feature_outputs(features, summary, paths)
    return features, summary, paths


def extract_one_mask_features(row: pd.Series, min_zdisc_pixels: int = 10, min_components: int = 1) -> dict[str, Any]:
    image_path = Path(str(row["annotation_image_path"]))
    mask_path = Path(str(row["mask_path"]))
    image = load_crop_image(image_path)
    mask = load_mask(mask_path, expected_shape=image.shape)
    zdisc = mask == 1
    ignore = mask == 2
    zdisc_count = int(np.sum(zdisc))
    ignore_count = int(np.sum(ignore))
    total_pixels = int(mask.size)
    component_labels, component_count = connected_zdisc_components(zdisc)
    component_sizes = component_size_array(component_labels, component_count)
    orientation = estimate_mask_orientation(
        zdisc,
        component_count=component_count,
        min_zdisc_pixels=min_zdisc_pixels,
        min_components=min_components,
    )
    return {
        "annotation_id": str(row["annotation_id"]),
        "image_id": str(row["image_id"]),
        "donor_id": str(row["donor_id"]),
        "patch_id": str(row["patch_id"]),
        "mask_path": str(mask_path),
        "image_path": str(image_path),
        "mask_shape": shape_string(mask.shape),
        "zdisc_pixel_count": zdisc_count,
        "ignore_pixel_count": ignore_count,
        "zdisc_pixel_fraction": zdisc_count / total_pixels if total_pixels else np.nan,
        "ignore_pixel_fraction": ignore_count / total_pixels if total_pixels else np.nan,
        "has_zdisc_labels": bool(zdisc_count > 0),
        "has_ignore_labels": bool(ignore_count > 0),
        "annotation_status": annotation_status(zdisc_count, ignore_count),
        "zdisc_component_count": int(component_count),
        "median_component_size": float(np.median(component_sizes)) if component_sizes.size else np.nan,
        "total_component_length_proxy": component_length_proxy(component_labels, component_count),
        "manual_mask_orientation_deg": orientation["manual_mask_orientation_deg"],
        "manual_mask_orientation_confidence": orientation["manual_mask_orientation_confidence"],
        "orientation_estimable": orientation["orientation_estimable"],
        "reason_not_estimable": orientation["reason_not_estimable"],
    }


def annotation_status(zdisc_pixel_count: int, ignore_pixel_count: int) -> str:
    if zdisc_pixel_count > 0 and ignore_pixel_count > 0:
        return "mixed"
    if zdisc_pixel_count > 0:
        return "zdisc_labeled"
    if ignore_pixel_count > 0:
        return "ignore_only"
    return "empty"


def connected_zdisc_components(zdisc_mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = np.ones((3, 3), dtype=bool)
    labels, count = ndi.label(np.asarray(zdisc_mask, dtype=bool), structure=structure)
    return labels.astype(np.int32, copy=False), int(count)


def component_size_array(component_labels: np.ndarray, component_count: int) -> np.ndarray:
    if component_count <= 0:
        return np.array([], dtype=float)
    return np.asarray(ndi.sum(np.ones_like(component_labels, dtype=np.float32), component_labels, index=np.arange(1, component_count + 1)))


def component_length_proxy(component_labels: np.ndarray, component_count: int) -> float:
    if component_count <= 0:
        return float("nan")
    total = 0.0
    objects = ndi.find_objects(component_labels)
    for slices in objects:
        if slices is None:
            continue
        y_slice, x_slice = slices
        height = int(y_slice.stop - y_slice.start)
        width = int(x_slice.stop - x_slice.start)
        total += float(max(height, width))
    return total


def estimate_mask_orientation(
    zdisc_mask: np.ndarray,
    component_count: int,
    min_zdisc_pixels: int = 10,
    min_components: int = 1,
) -> dict[str, Any]:
    ys, xs = np.nonzero(zdisc_mask)
    pixel_count = int(len(xs))
    if pixel_count == 0:
        return orientation_result(np.nan, np.nan, False, "no_zdisc_pixels")
    if pixel_count < int(min_zdisc_pixels):
        return orientation_result(np.nan, np.nan, False, "too_few_zdisc_pixels")
    if component_count < int(min_components):
        return orientation_result(np.nan, np.nan, False, "too_few_components")
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    coords -= np.mean(coords, axis=0, keepdims=True)
    if coords.shape[0] < 2:
        return orientation_result(np.nan, np.nan, False, "too_few_zdisc_pixels")
    covariance = np.cov(coords, rowvar=False)
    if not np.all(np.isfinite(covariance)):
        return orientation_result(np.nan, np.nan, False, "degenerate_orientation")
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)
    major_value = float(eigvals[order[-1]])
    minor_value = float(eigvals[order[0]])
    if major_value <= 0 or not np.isfinite(major_value):
        return orientation_result(np.nan, np.nan, False, "degenerate_orientation")
    vector = eigvecs[:, order[-1]]
    angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)
    confidence = float((major_value - minor_value) / (major_value + minor_value + 1.0e-12))
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return orientation_result(angle, confidence, True, "ok")


def orientation_result(angle: float, confidence: float, estimable: bool, reason: str) -> dict[str, Any]:
    return {
        "manual_mask_orientation_deg": angle,
        "manual_mask_orientation_confidence": confidence,
        "orientation_estimable": bool(estimable),
        "reason_not_estimable": "" if estimable else reason,
    }


def stabilize_feature_table(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy(deep=True)
    for column in ZDISC_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    for column in ["annotation_id", "image_id", "donor_id", "patch_id"]:
        result[column] = result[column].fillna("").astype(str)
    return result[ZDISC_FEATURE_COLUMNS]


def build_feature_summary(
    features: pd.DataFrame,
    index_path: str | Path,
    min_zdisc_pixels: int,
    min_components: int,
) -> dict[str, Any]:
    status_counts = features["annotation_status"].fillna("missing").astype(str).value_counts().to_dict()
    return json_safe(
        {
            "mode": "extract_zdisc_annotation_features",
            "index_path": str(index_path),
            "mask_count": int(len(features)),
            "masks_with_zdisc_labels": int(features["has_zdisc_labels"].fillna(False).astype(bool).sum()) if len(features) else 0,
            "empty_masks": int((features["annotation_status"] == "empty").sum()) if len(features) else 0,
            "ignore_only_masks": int((features["annotation_status"] == "ignore_only").sum()) if len(features) else 0,
            "mixed_masks": int((features["annotation_status"] == "mixed").sum()) if len(features) else 0,
            "orientation_estimable_masks": int(features["orientation_estimable"].fillna(False).astype(bool).sum()) if len(features) else 0,
            "annotation_status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "total_zdisc_pixels": int(pd.to_numeric(features["zdisc_pixel_count"], errors="coerce").fillna(0).sum()),
            "total_ignore_pixels": int(pd.to_numeric(features["ignore_pixel_count"], errors="coerce").fillna(0).sum()),
            "min_zdisc_pixels": int(min_zdisc_pixels),
            "min_components": int(min_components),
            "orientation_method": "PCA over label-1 mask pixel coordinates; axial angle in 0-180 degrees",
            "component_method": "8-connected components over label-1 mask pixels",
            "length_proxy_method": "sum of each component bounding-box major extent in pixels",
            "scope": "manual annotation feature extraction only; no validation statistics or production metric changes",
        }
    )


def write_feature_outputs(features: pd.DataFrame, summary: dict[str, Any], paths: dict[str, Path]) -> None:
    paths["features_csv"].parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(paths["features_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    lines = [f"{key}: {value}" for key, value in summary.items()]
    paths["summary_txt"].write_text("\n".join(lines) + "\n", encoding="utf-8")
