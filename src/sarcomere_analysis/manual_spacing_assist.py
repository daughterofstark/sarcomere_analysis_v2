from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks

from .config import get_calibration, output_dir


MANUAL_SPACING_COLUMNS = [
    "annotation_id",
    "crop_path",
    "image_id",
    "patch_id",
    "x0",
    "y0",
    "x1",
    "y1",
    "line_length_px",
    "line_length_um",
    "user_interval_count",
    "detected_peak_count",
    "estimated_spacing_px",
    "estimated_spacing_um",
    "confidence_score",
    "accepted_by_user",
    "notes",
]


@dataclass(frozen=True)
class LineProfileResult:
    crop: np.ndarray
    profile: np.ndarray
    smoothed_profile: np.ndarray
    sample_x: np.ndarray
    sample_y: np.ndarray
    peak_indices: np.ndarray
    trough_indices: np.ndarray
    line_length_px: float
    line_length_um: float
    estimated_spacing_px: float
    estimated_spacing_um: float
    candidate_kind: str


def load_crop_png(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    values = np.asarray(image, dtype=np.float32)
    if values.size == 0:
        raise ValueError(f"Empty crop image: {path}")
    return values / 255.0


def sample_line_profile(
    image: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    samples_per_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D crop image, got shape {values.shape}")
    line_length = float(np.hypot(float(x1) - float(x0), float(y1) - float(y0)))
    if not np.isfinite(line_length) or line_length <= 0:
        raise ValueError("Line length must be positive.")
    n_samples = max(int(np.ceil(line_length * max(samples_per_px, 1.0))) + 1, 2)
    xs = np.linspace(float(x0), float(x1), n_samples)
    ys = np.linspace(float(y0), float(y1), n_samples)
    profile = ndi.map_coordinates(values, [ys, xs], order=1, mode="nearest")
    return profile.astype(np.float32), xs.astype(np.float32), ys.astype(np.float32), line_length


def analyze_spacing_profile(
    crop: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    pixel_size_um: float,
    intervals: int | None = None,
    smooth_sigma: float = 1.0,
    min_peak_prominence: float | None = None,
) -> LineProfileResult:
    profile, xs, ys, line_length_px = sample_line_profile(crop, x0, y0, x1, y1)
    if profile.size < 3 or float(np.nanstd(profile)) <= 1.0e-8:
        smoothed = profile.astype(np.float32, copy=False)
        peaks = np.array([], dtype=int)
        troughs = np.array([], dtype=int)
    else:
        smoothed = ndi.gaussian_filter1d(profile.astype(np.float32, copy=False), sigma=max(float(smooth_sigma), 0.0))
        prominence = float(min_peak_prominence) if min_peak_prominence is not None else adaptive_prominence(smoothed)
        peaks, _ = find_peaks(smoothed, prominence=prominence)
        troughs, _ = find_peaks(-smoothed, prominence=prominence)

    line_length_um = line_length_px * float(pixel_size_um)
    if intervals is not None and int(intervals) > 0:
        estimated_spacing_px = line_length_px / int(intervals)
        candidate_kind = "user_interval_count"
    else:
        candidate_positions, candidate_kind = choose_candidate_positions(peaks, troughs)
        estimated_spacing_px = median_spacing(candidate_positions)
    estimated_spacing_um = estimated_spacing_px * float(pixel_size_um) if np.isfinite(estimated_spacing_px) else float("nan")
    return LineProfileResult(
        crop=crop,
        profile=profile,
        smoothed_profile=smoothed,
        sample_x=xs,
        sample_y=ys,
        peak_indices=peaks.astype(int),
        trough_indices=troughs.astype(int),
        line_length_px=line_length_px,
        line_length_um=line_length_um,
        estimated_spacing_px=float(estimated_spacing_px),
        estimated_spacing_um=float(estimated_spacing_um),
        candidate_kind=candidate_kind,
    )


def adaptive_prominence(profile: np.ndarray) -> float:
    finite = profile[np.isfinite(profile)]
    if finite.size == 0:
        return 0.05
    return max(float(np.nanstd(finite)) * 0.5, 0.02)


def choose_candidate_positions(peaks: np.ndarray, troughs: np.ndarray) -> tuple[np.ndarray, str]:
    if len(peaks) >= len(troughs) and len(peaks) >= 2:
        return peaks.astype(float), "peaks"
    if len(troughs) >= 2:
        return troughs.astype(float), "troughs"
    return np.array([], dtype=float), "insufficient_candidates"


def median_spacing(candidate_positions: np.ndarray) -> float:
    positions = np.asarray(candidate_positions, dtype=float)
    if positions.size < 2:
        return float("nan")
    diffs = np.diff(np.sort(positions))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float("nan")
    return float(np.median(diffs))


def build_manual_spacing_row(
    crop_path: str | Path,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    result: LineProfileResult,
    intervals: int | None = None,
    confidence_score: float | str | None = None,
    accepted_by_user: str | None = None,
    notes: str | None = None,
) -> dict[str, object]:
    parsed = parse_crop_filename(crop_path)
    return {
        "annotation_id": parsed.get("annotation_id", ""),
        "crop_path": str(crop_path),
        "image_id": parsed.get("image_id", ""),
        "patch_id": parsed.get("patch_id", ""),
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
        "line_length_px": result.line_length_px,
        "line_length_um": result.line_length_um,
        "user_interval_count": int(intervals) if intervals is not None else np.nan,
        "detected_peak_count": int(max(len(result.peak_indices), len(result.trough_indices))),
        "estimated_spacing_px": result.estimated_spacing_px,
        "estimated_spacing_um": result.estimated_spacing_um,
        "confidence_score": confidence_score if confidence_score is not None else "",
        "accepted_by_user": accepted_by_user or "",
        "notes": notes or "",
    }


def parse_crop_filename(path: str | Path) -> dict[str, str]:
    stem = Path(path).stem
    parts = stem.split("__")
    if len(parts) >= 3:
        return {"annotation_id": parts[0], "image_id": parts[1], "patch_id": "__".join(parts[2:])}
    match = re.match(r"(?P<annotation_id>ANN_\d+)", stem)
    if match:
        return {"annotation_id": match.group("annotation_id"), "image_id": "", "patch_id": ""}
    return {"annotation_id": stem, "image_id": "", "patch_id": ""}


def default_manual_spacing_output_csv(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "annotation_pack" / "manual_spacing_assist_results.csv"


def default_manual_spacing_panel_path(cfg: dict[str, Any], crop_path: str | Path) -> Path:
    parsed = parse_crop_filename(crop_path)
    annotation_id = parsed.get("annotation_id") or Path(crop_path).stem
    return output_dir(cfg) / "annotation_pack" / "manual_spacing_panels" / f"{safe_name(annotation_id)}_spacing_assist.png"


def run_manual_spacing_assist(
    cfg: dict[str, Any],
    crop_path: str | Path,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    intervals: int | None = None,
    smooth_sigma: float = 1.0,
    min_peak_prominence: float | None = None,
    confidence_score: float | str | None = None,
    accepted_by_user: str | None = None,
    notes: str | None = None,
) -> tuple[dict[str, object], LineProfileResult]:
    crop = load_crop_png(crop_path)
    result = analyze_spacing_profile(
        crop,
        x0,
        y0,
        x1,
        y1,
        pixel_size_um=get_calibration(cfg).pixel_size_um,
        intervals=intervals,
        smooth_sigma=smooth_sigma,
        min_peak_prominence=min_peak_prominence,
    )
    row = build_manual_spacing_row(
        crop_path,
        x0,
        y0,
        x1,
        y1,
        result,
        intervals=intervals,
        confidence_score=confidence_score,
        accepted_by_user=accepted_by_user,
        notes=notes,
    )
    return row, result


def write_manual_spacing_result(row: dict[str, object], output_csv: str | Path, append: bool = False) -> Path:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    for column in MANUAL_SPACING_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[MANUAL_SPACING_COLUMNS]
    if append and path.exists():
        existing = pd.read_csv(path, dtype={"annotation_id": str, "image_id": str, "patch_id": str})
        for column in MANUAL_SPACING_COLUMNS:
            if column not in existing.columns:
                existing[column] = np.nan
        frame = pd.concat([existing[MANUAL_SPACING_COLUMNS], frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return path


def write_diagnostic_panel(
    result: LineProfileResult,
    panel_path: str | Path,
    row: dict[str, object],
) -> Path:
    out = Path(panel_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out.parent / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out.parent / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(result.crop, cmap="gray", vmin=0, vmax=1)
    axes[0].plot(result.sample_x, result.sample_y, color="yellow", linewidth=2)
    axes[0].set_title("Crop with measurement line")
    axes[0].axis("off")

    x_axis = np.arange(result.profile.size)
    axes[1].plot(x_axis, result.profile, color="0.65", linewidth=1, label="raw profile")
    axes[1].plot(x_axis, result.smoothed_profile, color="black", linewidth=1.5, label="smoothed")
    if result.peak_indices.size:
        axes[1].plot(result.peak_indices, result.smoothed_profile[result.peak_indices], "o", color="tab:blue", label="peaks")
    if result.trough_indices.size:
        axes[1].plot(result.trough_indices, result.smoothed_profile[result.trough_indices], "v", color="tab:orange", label="troughs")
    axes[1].set_title(
        "Estimated spacing: "
        f"{format_float(row.get('estimated_spacing_px'))} px / "
        f"{format_float(row.get('estimated_spacing_um'))} um"
    )
    axes[1].set_xlabel("Distance along line (px samples)")
    axes[1].set_ylabel("Intensity")
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def interactive_line_from_clicks(crop_path: str | Path) -> tuple[float, float, float, float]:
    crop = load_crop_png(crop_path)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    axis.imshow(crop, cmap="gray", vmin=0, vmax=1)
    axis.set_title("Click line start and end, then close/press Enter")
    points = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(points) != 2:
        raise ValueError("Interactive mode requires exactly two clicks.")
    (x0, y0), (x1, y1) = points
    return float(x0), float(y0), float(x1), float(y1)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))


def format_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.3f}"
