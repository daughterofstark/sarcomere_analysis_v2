from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class TissueMaskResult:
    mask: np.ndarray
    metadata: dict[str, object]


def compute_tissue_mask(image: np.ndarray, config: dict[str, Any]) -> TissueMaskResult:
    params = masking_params(config)
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D image for tissue masking, got shape {values.shape}")

    finite = values[np.isfinite(values)]
    if finite.size == 0 or float(np.max(finite)) == float(np.min(finite)):
        mask = np.zeros(values.shape, dtype=bool)
        return TissueMaskResult(mask=mask, metadata=_metadata(params, threshold=float("nan"), raw_fraction=0.0))

    method = str(params["tissue_method"])
    if method == "otsu":
        threshold = otsu_threshold(finite)
    elif method == "percentile":
        threshold = float(np.percentile(finite, float(params["tissue_percentile"])))
    else:
        raise ValueError(f"Unsupported masking.tissue_method: {method}")

    raw_mask = values > threshold
    mask = clean_binary_mask(
        raw_mask,
        min_object_size_px=int(params["min_object_size_px"]),
        fill_holes=bool(params["fill_holes"]),
    )
    metadata = _metadata(params, threshold=threshold, raw_fraction=float(np.mean(raw_mask)))
    metadata["tissue_fraction"] = float(np.mean(mask))
    return TissueMaskResult(mask=mask.astype(bool, copy=False), metadata=metadata)


def masking_params(config: dict[str, Any]) -> dict[str, object]:
    defaults: dict[str, object] = {
        "tissue_method": "otsu",
        "tissue_percentile": 35.0,
        "min_object_size_px": 256,
        "fill_holes": True,
    }
    params = dict(defaults)
    params.update(config.get("masking", {}))
    params["tissue_method"] = str(params["tissue_method"]).lower()

    if params["tissue_method"] not in {"otsu", "percentile"}:
        raise ValueError("masking.tissue_method must be 'otsu' or 'percentile'")
    percentile = float(params["tissue_percentile"])
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("masking.tissue_percentile must be between 0 and 100")
    if int(params["min_object_size_px"]) < 0:
        raise ValueError("masking.min_object_size_px must be non-negative")
    return params


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    if min_value == max_value:
        return min_value

    counts, edges = np.histogram(finite, bins=bins, range=(min_value, max_value))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = counts.sum()
    if total == 0:
        return min_value

    weight_background = np.cumsum(counts)
    weight_foreground = total - weight_background
    mean_background = np.cumsum(counts * centers) / np.maximum(weight_background, 1)
    mean_foreground = (
        np.cumsum((counts * centers)[::-1]) / np.maximum(np.cumsum(counts[::-1]), 1)
    )[::-1]
    variance_between = weight_background[:-1] * weight_foreground[:-1] * (
        mean_background[:-1] - mean_foreground[1:]
    ) ** 2
    if variance_between.size == 0:
        return min_value
    return float(centers[int(np.argmax(variance_between))])


def clean_binary_mask(mask: np.ndarray, min_object_size_px: int, fill_holes: bool) -> np.ndarray:
    cleaned = np.asarray(mask, dtype=bool)
    if fill_holes:
        cleaned = ndi.binary_fill_holes(cleaned)

    if min_object_size_px > 0:
        labeled, n_labels = ndi.label(cleaned)
        if n_labels == 0:
            return np.zeros(cleaned.shape, dtype=bool)
        counts = np.bincount(labeled.ravel())
        keep = counts >= min_object_size_px
        keep[0] = False
        cleaned = keep[labeled]

    return cleaned.astype(bool, copy=False)


def _metadata(params: dict[str, object], threshold: float, raw_fraction: float) -> dict[str, object]:
    return {
        "tissue_method": params["tissue_method"],
        "tissue_threshold": threshold,
        "tissue_percentile": float(params["tissue_percentile"]),
        "min_object_size_px": int(params["min_object_size_px"]),
        "fill_holes": bool(params["fill_holes"]),
        "raw_tissue_fraction": raw_fraction,
    }
