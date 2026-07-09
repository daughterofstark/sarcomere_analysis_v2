from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sarcomere_analysis.config import manifest_csv_path, output_dir
from sarcomere_analysis.io import build_manifest, load_tiff
from sarcomere_analysis.masking import compute_tissue_mask
from sarcomere_analysis.orientation import compute_orientation_analysis
from sarcomere_analysis.preprocessing import preprocess_image
from sarcomere_analysis.qc import compute_patch_qc
from sarcomere_analysis.spacing.autocorrelation import (
    directional_profile,
    local_maxima_lags,
    normalized_autocorrelation,
    prepare_autocorrelation_profile,
)


REVIEW_CLASSES = [
    "accepted_current",
    "no_local_peak",
    "low_periodicity_confidence",
    "global_out_of_band",
    "borderline_in_band",
]

REVIEW_INDEX_COLUMNS = [
    "review_class",
    "image_id",
    "donor_id",
    "patch_id",
    "panel_path",
    "render_status",
    "render_error",
    "final_valid_for_spacing",
    "final_invalid_reason",
    "selected_lag_px",
    "selected_lag_um",
    "best_in_band_lag_px",
    "best_global_lag_px",
    "peak_confidence",
    "rejected_reason_diagnostic",
]


@dataclass(frozen=True)
class ReviewImageContext:
    image_id: str
    donor_id: str
    preprocessed_image: np.ndarray
    tissue_mask: np.ndarray
    patch_metrics: pd.DataFrame


def default_candidate_table_path(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "diagnostics" / "spacing_candidates.csv"


def default_patch_table_path(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "tables" / "per_patch_metrics.csv"


def default_review_output_dir(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "diagnostics" / "spacing_candidate_review"


def load_manifest_for_review(cfg: dict[str, Any]) -> pd.DataFrame:
    path = manifest_csv_path(cfg)
    if path.exists():
        return pd.read_csv(path, dtype={"image_id": str, "donor_id": str, "region_id": str})
    return build_manifest(cfg)


def load_review_inputs(
    cfg: dict[str, Any],
    candidate_table: str | Path | None = None,
    patch_table: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_path = Path(candidate_table) if candidate_table else default_candidate_table_path(cfg)
    patch_path = Path(patch_table) if patch_table else default_patch_table_path(cfg)
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate diagnostics table not found: {candidate_path}. "
            "Run: ../sarcgraph-env/bin/python scripts/diagnose_spacing_candidates.py --config configs/default.yaml --all --compare-main-table"
        )
    candidates = pd.read_csv(candidate_path, dtype={"image_id": str, "donor_id": str, "patch_id": str})
    patches = pd.read_csv(patch_path, dtype={"image_id": str, "donor_id": str, "patch_id": str}) if patch_path.exists() else pd.DataFrame()
    manifest = load_manifest_for_review(cfg)
    return candidates, patches, manifest


def select_review_candidates(
    candidates: pd.DataFrame,
    classes: Iterable[str] | None = None,
    max_per_class: int = 10,
    seed: int = 123,
    image_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    table = candidates.copy(deep=True)
    if image_ids:
        table = table.loc[table["image_id"].astype(str).isin(image_ids)].copy()
    requested_classes = list(classes) if classes else list(REVIEW_CLASSES)
    unknown = [name for name in requested_classes if name not in REVIEW_CLASSES]
    if unknown:
        raise ValueError(f"Unknown review class(es): {unknown}")

    selected_frames: list[pd.DataFrame] = []
    selected_keys: set[tuple[str, str]] = set()
    available_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}

    for class_index, class_name in enumerate(requested_classes):
        mask = class_mask(table, class_name)
        available = table.loc[mask].copy()
        available_counts[class_name] = int(len(available))
        if not available.empty and selected_keys:
            keys = list(zip(available["image_id"].astype(str), available["patch_id"].astype(str)))
            available = available.loc[[key not in selected_keys for key in keys]].copy()
        chosen = diverse_deterministic_sample(available, max_per_class, seed + class_index)
        if not chosen.empty:
            chosen.insert(0, "review_class", class_name)
            selected_keys.update(zip(chosen["image_id"].astype(str), chosen["patch_id"].astype(str)))
            selected_frames.append(chosen)
        selected_counts[class_name] = int(len(chosen))

    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame(columns=["review_class", *table.columns])
    summary = {
        "available_counts_by_class": available_counts,
        "selected_counts_by_class": selected_counts,
        "missing_classes": [name for name, count in available_counts.items() if count == 0],
    }
    return selected, summary


def class_mask(table: pd.DataFrame, class_name: str) -> pd.Series:
    final_valid = bool_series(table, "final_valid_for_spacing")
    reason_text = text_series(table, "rejected_reason_diagnostic") + ";" + text_series(table, "final_invalid_reason")
    in_band = numeric_series(table, "best_in_band_lag_px")
    in_band_peak = numeric_series(table, "best_in_band_peak_value")
    global_lag = numeric_series(table, "best_global_lag_px")
    global_peak = numeric_series(table, "best_global_peak_value")
    expected_min = numeric_series(table, "expected_min_lag_px")
    expected_max = numeric_series(table, "expected_max_lag_px")
    confidence = numeric_series(table, "peak_confidence")

    if class_name == "accepted_current":
        return final_valid
    if class_name == "no_local_peak":
        return reason_text.str.contains("no_local_peak", case=False, na=False)
    if class_name == "low_periodicity_confidence":
        return reason_text.str.contains("low_periodicity_confidence", case=False, na=False) & in_band.notna()
    if class_name == "global_out_of_band":
        outside = global_lag.notna() & ((global_lag < np.ceil(expected_min)) | (global_lag > np.floor(expected_max)))
        absent_or_weaker_in_band = in_band.isna() | in_band_peak.isna() | (global_peak.fillna(-np.inf) > in_band_peak.fillna(-np.inf))
        return outside & absent_or_weaker_in_band
    if class_name == "borderline_in_band":
        return in_band.notna() & confidence.notna() & ((confidence - 0.15).abs() <= 0.02)
    raise ValueError(f"Unknown review class: {class_name}")


def diverse_deterministic_sample(table: pd.DataFrame, max_count: int, seed: int) -> pd.DataFrame:
    if max_count <= 0 or table.empty:
        return table.head(0).copy()
    shuffled = table.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    first_per_image = shuffled.drop_duplicates(subset=["image_id"], keep="first")
    selected = first_per_image.head(max_count)
    if len(selected) < max_count:
        remaining = shuffled.drop(index=selected.index, errors="ignore")
        selected = pd.concat([selected, remaining.head(max_count - len(selected))], ignore_index=True)
    return selected.head(max_count).reset_index(drop=True)


def recompute_review_context(manifest: pd.DataFrame, image_id: str, cfg: dict[str, Any]) -> ReviewImageContext:
    rows = manifest.loc[manifest["image_id"].astype(str) == str(image_id)]
    if rows.empty:
        raise ValueError(f"Image id not found in manifest: {image_id}")
    row = rows.iloc[0]
    donor_id = str(row["donor_id"]) if "donor_id" in row and pd.notna(row["donor_id"]) else ""
    raw = load_tiff(Path(str(row["image_path"])))
    preprocessing = preprocess_image(raw, cfg)
    mask = compute_tissue_mask(preprocessing.image, cfg)
    patch_qc = compute_patch_qc(preprocessing.image, mask.mask, str(image_id), cfg)
    orientation = compute_orientation_analysis(preprocessing.image, mask.mask, patch_qc, cfg)
    patch_metrics = orientation.patch_metrics.copy()
    if "donor_id" not in patch_metrics.columns:
        patch_metrics.insert(1, "donor_id", donor_id)
    return ReviewImageContext(
        image_id=str(image_id),
        donor_id=donor_id,
        preprocessed_image=preprocessing.image,
        tissue_mask=mask.mask,
        patch_metrics=patch_metrics,
    )


def export_review_pack(
    cfg: dict[str, Any],
    candidate_table: str | Path | None = None,
    patch_table: str | Path | None = None,
    output_directory: str | Path | None = None,
    classes: Iterable[str] | None = None,
    max_per_class: int = 10,
    seed: int = 123,
    image_ids: set[str] | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    candidates, main_patches, manifest = load_review_inputs(cfg, candidate_table, patch_table)
    selected, selection_summary = select_review_candidates(candidates, classes, max_per_class, seed, image_ids)
    out_dir = Path(output_directory) if output_directory else default_review_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    contexts: dict[str, ReviewImageContext] = {}
    rows: list[dict[str, object]] = []
    for _, candidate in selected.iterrows():
        image_id = str(candidate["image_id"])
        patch_id = str(candidate["patch_id"])
        panel_path = out_dir / f"{safe_name(str(candidate['review_class']))}__{safe_name(image_id)}__{safe_name(patch_id)}.png"
        status = "ok"
        error = ""
        try:
            if image_id not in contexts:
                contexts[image_id] = recompute_review_context(manifest, image_id, cfg)
            context = contexts[image_id]
            patch_rows = context.patch_metrics.loc[context.patch_metrics["patch_id"].astype(str) == patch_id]
            if patch_rows.empty:
                raise ValueError(f"Patch id not found after recomputing image context: {patch_id}")
            if overwrite or not panel_path.exists():
                write_spacing_review_panel(context, patch_rows.iloc[0], candidate, panel_path, cfg)
        except Exception as exc:  # pragma: no cover - exercised by real data edge cases
            status = "error"
            error = str(exc)
        rows.append(index_row(candidate, panel_path, status, error))

    index = stabilize_review_index(pd.DataFrame(rows))
    summary = build_review_summary(candidates, selected, selection_summary, manifest, main_patches, index, out_dir)
    paths = write_review_outputs(index, summary, out_dir)
    return index, summary, paths


def write_spacing_review_panel(
    context: ReviewImageContext,
    patch_row: pd.Series,
    candidate_row: pd.Series,
    path: str | Path,
    cfg: dict[str, Any],
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out.parent / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out.parent / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y0, y1 = int(patch_row["y0"]), int(patch_row["y1"])
    x0, x1 = int(patch_row["x0"]), int(patch_row["x1"])
    patch = context.preprocessed_image[y0:y1, x0:x1]
    mask_patch = context.tissue_mask[y0:y1, x0:x1]
    theta = float(patch_row.get("patch_mean_orientation_rad", np.nan))
    ac_params = cfg.get("spacing", {}).get("autocorrelation", {})
    raw_profile = directional_profile(patch, theta, bin_px=float(ac_params.get("profile_bin_px", 1.0)))
    profile = prepare_autocorrelation_profile(
        patch,
        theta,
        bin_px=float(ac_params.get("profile_bin_px", 1.0)),
        min_length=int(ac_params.get("min_profile_length_px", 32)),
    )
    autocorr = normalized_autocorrelation(profile)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(patch, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Preprocessed patch")
    draw_orientation_indicator(axes[0], patch.shape, theta)
    axes[0].axis("off")

    axes[1].imshow(patch, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(np.ma.masked_where(~mask_patch.astype(bool), mask_patch), cmap="Greens", alpha=0.35)
    axes[1].set_title("Tissue mask overlay")
    axes[1].axis("off")

    axes[2].plot(raw_profile, color="black", linewidth=1.2)
    axes[2].set_title("Directional intensity profile")
    axes[2].set_xlabel("Profile bin")
    axes[2].set_ylabel("Intensity")

    axes[3].set_title("Autocorrelation")
    if autocorr.size:
        lags = np.arange(autocorr.size)
        axes[3].plot(lags, autocorr, color="black", linewidth=1.1)
        min_lag = float(candidate_row.get("expected_min_lag_px", np.nan))
        max_lag = float(candidate_row.get("expected_max_lag_px", np.nan))
        if np.isfinite(min_lag) and np.isfinite(max_lag):
            axes[3].axvspan(np.ceil(min_lag), np.floor(max_lag), color="tab:blue", alpha=0.12, label="expected band")
            for lag in local_maxima_lags(autocorr, int(max(1, np.ceil(min_lag))), int(min(np.floor(max_lag), autocorr.size - 1))):
                axes[3].plot(lag, autocorr[lag], marker="o", color="tab:blue", markersize=3)
        mark_peak(axes[3], autocorr, candidate_row.get("selected_lag_px", np.nan), "selected", "tab:green")
        mark_peak(axes[3], autocorr, candidate_row.get("best_in_band_lag_px", np.nan), "best in band", "tab:orange")
        mark_peak(axes[3], autocorr, candidate_row.get("best_global_lag_px", np.nan), "best global", "tab:red")
        axes[3].legend(loc="best", fontsize=8)
    else:
        axes[3].text(0.5, 0.5, "No valid autocorrelation", ha="center", va="center")
    axes[3].set_xlabel("Lag (px)")
    axes[3].set_ylabel("Normalized autocorr.")

    axes[4].axis("off")
    axes[4].text(0.0, 1.0, annotation_text(candidate_row, patch_row), va="top", family="monospace", fontsize=9)

    axes[5].axis("off")
    axes[5].text(
        0.0,
        1.0,
        "Diagnostic-only review panel\n"
        "These plots explain estimator inputs and candidate peaks.\n"
        "They do not change saved metrics or validate spacing.",
        va="top",
        fontsize=10,
    )
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def draw_orientation_indicator(axis: Any, shape: tuple[int, int], theta: float) -> None:
    if not np.isfinite(theta):
        return
    height, width = shape
    cx, cy = width / 2.0, height / 2.0
    length = min(height, width) * 0.28
    dx = np.cos(theta) * length
    dy = np.sin(theta) * length
    axis.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color="yellow", linewidth=2)


def mark_peak(axis: Any, autocorr: np.ndarray, lag_value: object, label: str, color: str) -> None:
    try:
        lag = float(lag_value)
    except (TypeError, ValueError):
        return
    if not np.isfinite(lag):
        return
    lag_int = int(round(lag))
    if 0 <= lag_int < autocorr.size:
        axis.axvline(lag_int, color=color, linestyle="--", linewidth=1.0, label=label)
        axis.plot(lag_int, autocorr[lag_int], marker="o", color=color, markersize=5)


def annotation_text(candidate_row: pd.Series, patch_row: pd.Series) -> str:
    fields = [
        ("image_id", candidate_row.get("image_id", "")),
        ("patch_id", candidate_row.get("patch_id", "")),
        ("class", candidate_row.get("review_class", "")),
        ("final_valid", candidate_row.get("final_valid_for_spacing", "")),
        ("selected_px", candidate_row.get("selected_lag_px", np.nan)),
        ("selected_um", candidate_row.get("selected_lag_um", np.nan)),
        ("best_in_band_px", candidate_row.get("best_in_band_lag_px", np.nan)),
        ("best_global_px", candidate_row.get("best_global_lag_px", np.nan)),
        ("confidence", candidate_row.get("peak_confidence", np.nan)),
        ("reason", candidate_row.get("rejected_reason_diagnostic", candidate_row.get("final_invalid_reason", ""))),
        ("orientation_rad", patch_row.get("patch_mean_orientation_rad", np.nan)),
    ]
    return "\n".join(f"{name}: {format_value(value)}" for name, value in fields)


def build_review_summary(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    selection_summary: dict[str, Any],
    manifest: pd.DataFrame,
    main_patches: pd.DataFrame,
    index: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:
    candidate_image_count = int(candidates["image_id"].astype(str).nunique()) if "image_id" in candidates.columns else 0
    manifest_image_count = int(manifest["image_id"].astype(str).nunique()) if "image_id" in manifest.columns else 0
    main_patch_image_count = int(main_patches["image_id"].astype(str).nunique()) if "image_id" in main_patches.columns else 0
    summary = {
        "candidate_rows": int(len(candidates)),
        "candidate_image_count": candidate_image_count,
        "manifest_image_count": manifest_image_count,
        "main_patch_table_image_count": main_patch_image_count,
        "review_is_limited_to_candidate_images": bool(manifest_image_count and candidate_image_count < manifest_image_count),
        "coverage_note": (
            f"Review pack is limited to {candidate_image_count} candidate-diagnostic image(s), not the full manifest."
            if manifest_image_count and candidate_image_count < manifest_image_count
            else "Review pack candidate diagnostics cover the manifest image count."
        ),
        "selected_rows": int(len(selected)),
        "rendered_panels": int((index["render_status"] == "ok").sum()) if "render_status" in index.columns else 0,
        "render_errors": int((index["render_status"] == "error").sum()) if "render_status" in index.columns else 0,
        "output_dir": str(out_dir),
        **selection_summary,
        "safety_note": (
            "This review pack is diagnostic-only. It does not alter spacing thresholds, saved endpoint metrics, "
            "or final valid_for_spacing decisions."
        ),
    }
    return json_safe(summary)


def write_review_outputs(index: pd.DataFrame, summary: dict[str, Any], output_directory: str | Path) -> dict[str, Path]:
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "review_index": out_dir / "review_index.csv",
        "review_summary_json": out_dir / "review_summary.json",
    }
    stabilize_review_index(index).to_csv(paths["review_index"], index=False)
    with paths["review_summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
    return paths


def index_row(candidate: pd.Series, panel_path: Path, status: str, error: str) -> dict[str, object]:
    return {
        "review_class": str(candidate.get("review_class", "")),
        "image_id": str(candidate.get("image_id", "")),
        "donor_id": str(candidate.get("donor_id", "")) if pd.notna(candidate.get("donor_id", "")) else "",
        "patch_id": str(candidate.get("patch_id", "")),
        "panel_path": str(panel_path),
        "render_status": status,
        "render_error": error,
        "final_valid_for_spacing": bool(candidate.get("final_valid_for_spacing", False)),
        "final_invalid_reason": str(candidate.get("final_invalid_reason", "")),
        "selected_lag_px": candidate.get("selected_lag_px", np.nan),
        "selected_lag_um": candidate.get("selected_lag_um", np.nan),
        "best_in_band_lag_px": candidate.get("best_in_band_lag_px", np.nan),
        "best_global_lag_px": candidate.get("best_global_lag_px", np.nan),
        "peak_confidence": candidate.get("peak_confidence", np.nan),
        "rejected_reason_diagnostic": str(candidate.get("rejected_reason_diagnostic", "")),
    }


def stabilize_review_index(index: pd.DataFrame) -> pd.DataFrame:
    result = index.copy()
    for column in REVIEW_INDEX_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    return result.loc[:, REVIEW_INDEX_COLUMNS]


def bool_series(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(False, index=table.index)
    values = table[column]
    if values.dtype == object:
        return values.astype(str).str.lower().isin({"true", "1", "yes"})
    return values.fillna(False).astype(bool)


def numeric_series(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(np.nan, index=table.index, dtype=float)
    return pd.to_numeric(table[column], errors="coerce")


def text_series(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series("", index=table.index, dtype=str)
    return table[column].fillna("").astype(str)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def format_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isfinite(numeric):
        return f"{numeric:.4g}"
    return "nan"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
