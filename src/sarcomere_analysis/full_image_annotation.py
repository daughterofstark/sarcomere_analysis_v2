from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import output_dir
from .io import load_tiff
from .preprocessing import preprocess_image
from .zdisc_annotation import ZDISC_LABELS, bool_column, json_safe, safe_name, shape_string, write_label_overlay
from .zdisc_draw_ui import headless_check as draw_headless_check
from .zdisc_draw_ui import load_crop_image, load_draw_index, load_mask, run_draw_ui


FULL_IMAGE_INDEX_COLUMNS = [
    "annotation_id",
    "image_id",
    "donor_id",
    "patch_id",
    "oop_bin",
    "image_oop",
    "orientation_valid_fraction",
    "status",
    "source_image_path",
    "annotation_image_path",
    "mask_path",
    "overlay_path",
]


def default_full_image_annotation_dir(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "full_image_zdisc_annotation"


def default_full_image_index_path(cfg: dict[str, Any]) -> Path:
    return default_full_image_annotation_dir(cfg) / "full_image_annotation_index.csv"


def default_full_image_progress_path(cfg: dict[str, Any]) -> Path:
    return default_full_image_annotation_dir(cfg) / "full_image_draw_progress.json"


def load_full_image_inputs(
    cfg: dict[str, Any],
    analysis_table: str | Path | None = None,
    feature_table: str | Path | None = None,
    manifest_table: str | Path | None = None,
) -> pd.DataFrame:
    tables = output_dir(cfg) / "tables"
    analysis_path = Path(analysis_table) if analysis_table else tables / "analysis_per_image.csv"
    feature_path = Path(feature_table) if feature_table else tables / "features_per_image.csv"
    manifest_path = Path(manifest_table) if manifest_table else tables / "enriched_manifest.csv"
    analysis = pd.read_csv(analysis_path, dtype={"image_id": str, "donor_id": str})
    features = pd.read_csv(feature_path, dtype={"image_id": str, "donor_id": str}) if feature_path.exists() else pd.DataFrame()
    manifest = pd.read_csv(manifest_path, dtype={"image_id": str, "donor_id": str}) if manifest_path.exists() else pd.DataFrame()
    require_columns(analysis, ["image_id", "donor_id"], "analysis_per_image")
    if "image_path" not in analysis.columns and not manifest.empty:
        analysis = analysis.merge(manifest[["image_id", "donor_id", "image_path"]], on=["image_id", "donor_id"], how="left")
    if "image_oop" not in analysis.columns and not features.empty and "image_oop" in features.columns:
        analysis = analysis.merge(features[["image_id", "donor_id", "image_oop"]], on=["image_id", "donor_id"], how="left")
    require_columns(analysis, ["image_id", "donor_id", "image_path"], "full-image annotation inputs")
    analysis["image_id"] = analysis["image_id"].astype(str)
    analysis["donor_id"] = analysis["donor_id"].astype(str)
    analysis["image_oop"] = pd.to_numeric(analysis.get("image_oop", np.nan), errors="coerce")
    return analysis


def select_full_images(table: pd.DataFrame, n_images: int = 12, seed: int = 123) -> pd.DataFrame:
    data = table.copy(deep=True)
    if n_images <= 0:
        return data.head(0).copy()
    data["oop_bin"] = assign_image_oop_bins(data.get("image_oop", pd.Series(np.nan, index=data.index)))
    bins = ["low_oop", "medium_oop", "high_oop"]
    selected_frames: list[pd.DataFrame] = []
    per_bin = distribute_counts(int(n_images), bins)
    for idx, bin_name in enumerate(bins):
        subset = data.loc[data["oop_bin"] == bin_name].copy()
        selected = diverse_sample(subset, per_bin[bin_name], seed + idx)
        if not selected.empty:
            selected_frames.append(selected)
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else data.head(0).copy()
    if len(selected) < n_images:
        used = set(selected["image_id"].astype(str))
        fill_pool = data.loc[~data["image_id"].astype(str).isin(used)].copy()
        fill = diverse_sample(fill_pool, n_images - len(selected), seed + 99)
        selected = pd.concat([selected, fill], ignore_index=True) if not fill.empty else selected
    selected = selected.head(n_images).copy()
    selected.insert(0, "annotation_id", [f"FULL_{idx + 1:04d}" for idx in range(len(selected))])
    selected["patch_id"] = selected["image_id"].astype(str)
    return selected


def prepare_full_image_annotation_set(
    cfg: dict[str, Any],
    n_images: int = 12,
    seed: int = 123,
    output_directory: str | Path | None = None,
    analysis_table: str | Path | None = None,
    feature_table: str | Path | None = None,
    manifest_table: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    out_dir = Path(output_directory) if output_directory else default_full_image_annotation_dir(cfg)
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    for directory in [out_dir, images_dir, masks_dir, overlays_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    inputs = load_full_image_inputs(cfg, analysis_table=analysis_table, feature_table=feature_table, manifest_table=manifest_table)
    selected = select_full_images(inputs, n_images=n_images, seed=seed)
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        source_path = Path(str(row["image_path"]))
        if not source_path.exists():
            raise FileNotFoundError(f"Raw source image for annotation not found: {source_path}")
        safe_id = safe_name(str(row["annotation_id"]))
        annotation_image_path = images_dir / f"{safe_id}__{safe_name(str(row['image_id']))}.png"
        mask_path = masks_dir / f"{safe_id}__{safe_name(str(row['image_id']))}_mask.png"
        overlay_path = overlays_dir / f"{safe_id}__{safe_name(str(row['image_id']))}_overlay.png"
        if overwrite or not annotation_image_path.exists() or not mask_path.exists():
            raw = load_tiff(source_path)
            processed = preprocess_image(raw, cfg).image
            if overwrite or not annotation_image_path.exists():
                write_grayscale_png(processed, annotation_image_path)
            if overwrite or not mask_path.exists():
                blank = np.zeros(processed.shape, dtype=np.uint8)
                Image.fromarray(blank, mode="L").save(mask_path)
        rows.append(build_full_image_index_row(row, source_path, annotation_image_path, mask_path, overlay_path))
    index = pd.DataFrame(rows)
    index = stabilize_full_image_index(index)
    paths = full_image_paths(out_dir)
    summary = build_prepare_summary(index, seed=seed)
    index.to_csv(paths["index"], index=False)
    write_summary(summary, paths["summary_json"], paths["summary_txt"])
    return index, summary, paths


def audit_full_image_annotations(
    cfg: dict[str, Any],
    index_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    write_overlays: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    out_dir = Path(output_directory) if output_directory else default_full_image_annotation_dir(cfg)
    index_file = Path(index_path) if index_path else out_dir / "full_image_annotation_index.csv"
    if not index_file.exists():
        raise FileNotFoundError(f"Full-image annotation index not found: {index_file}")
    index = load_draw_index(index_file)
    rows = []
    for _, row in index.iterrows():
        rows.append(audit_full_image_row(row, write_overlays=write_overlays))
    audit = pd.DataFrame(rows)
    summary = build_audit_summary(audit)
    paths = full_image_paths(out_dir)
    write_summary(summary, paths["summary_json"], paths["summary_txt"])
    return audit, summary, paths


def headless_check_full_image_annotations(cfg: dict[str, Any], index_path: str | Path | None = None) -> dict[str, Any]:
    return draw_headless_check(cfg, index_path=index_path or default_full_image_index_path(cfg))


def run_full_image_draw_ui(
    cfg: dict[str, Any],
    index_path: str | Path | None = None,
    start_annotation_id: str | None = None,
    brush_size: int = 2,
    alpha: float = 0.45,
    overwrite_progress: bool = False,
) -> Path:
    return run_draw_ui(
        cfg,
        index_path=index_path or default_full_image_index_path(cfg),
        start_annotation_id=start_annotation_id,
        brush_size=brush_size,
        alpha=alpha,
        overwrite_progress=overwrite_progress,
        progress_path=default_full_image_progress_path(cfg),
    )


def audit_full_image_row(row: pd.Series, write_overlays: bool = False) -> dict[str, Any]:
    image_path = Path(str(row["annotation_image_path"]))
    mask_path = Path(str(row["mask_path"]))
    result = {column: row.get(column, "") for column in FULL_IMAGE_INDEX_COLUMNS}
    result.update(
        {
            "mask_exists": mask_path.exists(),
            "image_shape": "",
            "mask_shape": "",
            "shape_matches": False,
            "allowed_labels_only": False,
            "invalid_label_values": "",
            "label_0_pixels": 0,
            "label_1_pixels": 0,
            "label_2_pixels": 0,
            "empty_mask": True,
            "has_zdisc_labels": False,
        }
    )
    if not image_path.exists():
        result["image_shape"] = "missing_image"
        return result
    image = load_crop_image(image_path)
    result["image_shape"] = shape_string(image.shape)
    if not mask_path.exists():
        result["mask_shape"] = "missing_mask"
        return result
    try:
        mask = load_mask(mask_path, expected_shape=image.shape)
    except ValueError as exc:
        result["mask_shape"] = str(exc)
        return result
    result["mask_shape"] = shape_string(mask.shape)
    result["shape_matches"] = True
    unique = sorted(int(value) for value in np.unique(mask))
    invalid = [value for value in unique if value not in ZDISC_LABELS]
    result["allowed_labels_only"] = len(invalid) == 0
    result["invalid_label_values"] = ";".join(str(value) for value in invalid)
    for label in ZDISC_LABELS:
        result[f"label_{label}_pixels"] = int(np.sum(mask == label))
    result["empty_mask"] = int(result["label_1_pixels"]) == 0 and int(result["label_2_pixels"]) == 0
    result["has_zdisc_labels"] = int(result["label_1_pixels"]) > 0
    if write_overlays and result["shape_matches"]:
        write_label_overlay(image, mask, row["overlay_path"])
    return result


def build_full_image_index_row(
    row: pd.Series,
    source_image_path: Path,
    annotation_image_path: Path,
    mask_path: Path,
    overlay_path: Path,
) -> dict[str, Any]:
    return {
        "annotation_id": str(row["annotation_id"]),
        "image_id": str(row["image_id"]),
        "donor_id": str(row["donor_id"]),
        "patch_id": str(row["patch_id"]),
        "oop_bin": str(row.get("oop_bin", "")),
        "image_oop": row.get("image_oop", np.nan),
        "orientation_valid_fraction": row.get("orientation_valid_fraction", np.nan),
        "status": row.get("status", ""),
        "source_image_path": str(source_image_path),
        "annotation_image_path": str(annotation_image_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
    }


def stabilize_full_image_index(index: pd.DataFrame) -> pd.DataFrame:
    result = index.copy(deep=True)
    for column in FULL_IMAGE_INDEX_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    for column in ["annotation_id", "image_id", "donor_id", "patch_id"]:
        result[column] = result[column].fillna("").astype(str)
    return result[FULL_IMAGE_INDEX_COLUMNS]


def build_prepare_summary(index: pd.DataFrame, seed: int) -> dict[str, Any]:
    return json_safe(
        {
            "mode": "prepare_full_image_zdisc_annotation_set",
            "selected_images": int(len(index)),
            "seed": int(seed),
            "oop_bin_counts": value_counts(index.get("oop_bin", pd.Series(dtype=object))),
            "unique_donors": int(index["donor_id"].nunique()) if len(index) else 0,
            "label_convention": ZDISC_LABELS,
            "purpose": "local full-image manual Z-disc/striation annotation; exported PNGs are working copies, not raw TIFF copies",
        }
    )


def build_audit_summary(audit: pd.DataFrame) -> dict[str, Any]:
    return json_safe(
        {
            "mode": "audit_full_image_zdisc_annotations",
            "selected_images": int(len(audit)),
            "missing_masks": int((~bool_column(audit, "mask_exists")).sum()) if len(audit) else 0,
            "shape_mismatch_masks": int((~bool_column(audit, "shape_matches")).sum()) if len(audit) else 0,
            "invalid_label_masks": int((~bool_column(audit, "allowed_labels_only")).sum()) if len(audit) else 0,
            "empty_masks": int(bool_column(audit, "empty_mask").sum()) if len(audit) else 0,
            "masks_with_zdisc_labels": int(bool_column(audit, "has_zdisc_labels").sum()) if len(audit) else 0,
            "label_pixel_totals": {
                str(label): int(pd.to_numeric(audit.get(f"label_{label}_pixels", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
                for label in ZDISC_LABELS
            },
            "label_convention": ZDISC_LABELS,
            "empty_masks_allowed": True,
        }
    )


def full_image_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "index": out_dir / "full_image_annotation_index.csv",
        "summary_json": out_dir / "full_image_annotation_summary.json",
        "summary_txt": out_dir / "full_image_annotation_summary.txt",
        "images_dir": out_dir / "images",
        "masks_dir": out_dir / "masks",
        "overlays_dir": out_dir / "overlays",
    }


def write_grayscale_png(image: np.ndarray, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    values = np.clip(np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0), 0.0, 1.0)
    Image.fromarray((values * 255).astype(np.uint8), mode="L").save(out)
    return out


def write_summary(summary: dict[str, Any], json_path: str | Path, txt_path: str | Path) -> None:
    json_out = Path(json_path)
    txt_out = Path(txt_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    txt_out.write_text("\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n", encoding="utf-8")


def assign_image_oop_bins(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    bins = pd.Series("missing_oop", index=numeric.index, dtype=object)
    valid = numeric.dropna()
    if valid.empty:
        return bins
    try:
        ranked = valid.rank(method="first")
        quantiles = pd.qcut(ranked, q=3, labels=["low_oop", "medium_oop", "high_oop"])
        bins.loc[valid.index] = quantiles.astype(str)
    except ValueError:
        bins.loc[valid.index] = "medium_oop"
    return bins


def distribute_counts(total: int, names: list[str]) -> dict[str, int]:
    if total <= 0:
        return {name: 0 for name in names}
    base = total // len(names)
    remainder = total % len(names)
    return {name: base + (1 if idx < remainder else 0) for idx, name in enumerate(names)}


def diverse_sample(table: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0 or table.empty:
        return table.head(0).copy()
    shuffled = table.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    selected_frames = []
    selected_count = 0
    remaining = shuffled.copy()
    while selected_count < count and not remaining.empty:
        take = remaining.drop_duplicates("donor_id", keep="first").head(count - selected_count)
        if take.empty:
            take = remaining.head(count - selected_count)
        selected_frames.append(take)
        selected_count += len(take)
        used = set(take["image_id"].astype(str))
        remaining = remaining.loc[~remaining["image_id"].astype(str).isin(used)].copy()
    return pd.concat(selected_frames, ignore_index=True).head(count) if selected_frames else table.head(0).copy()


def value_counts(values: pd.Series) -> dict[str, int]:
    if values.empty:
        return {}
    counts = values.fillna("missing").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required {label} columns: {missing}")
