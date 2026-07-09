from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .annotation_ui import read_annotation_index
from .config import output_dir


ZDISC_LABELS = {
    0: "background_unlabeled",
    1: "visible_zdisc_striation",
    2: "ignore_uncertain_autofluorescence_ambiguous",
}

ZDISC_INDEX_COLUMNS = [
    "annotation_id",
    "image_id",
    "donor_id",
    "patch_id",
    "oop_bin",
    "valid_for_orientation",
    "patch_oop",
    "patch_mean_orientation_deg",
    "suggested_annotation_task",
    "source_crop_path",
    "annotation_image_path",
    "mask_path",
    "overlay_path",
]

ZDISC_AUDIT_COLUMNS = ZDISC_INDEX_COLUMNS + [
    "mask_exists",
    "image_shape",
    "mask_shape",
    "shape_matches",
    "allowed_labels_only",
    "invalid_label_values",
    "label_0_pixels",
    "label_1_pixels",
    "label_2_pixels",
    "empty_mask",
    "has_zdisc_labels",
]


def default_zdisc_output_dir(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "zdisc_annotation"


def load_annotation_pack_index(index_path: str | Path) -> pd.DataFrame:
    index = read_annotation_index(index_path)
    if "oop_bin" not in index.columns:
        index["oop_bin"] = assign_oop_bins(index.get("patch_oop", pd.Series(np.nan, index=index.index)))
    index["image_id"] = index["image_id"].astype(str)
    index["donor_id"] = index["donor_id"].astype(str)
    index["patch_id"] = index["patch_id"].astype(str)
    index["annotation_id"] = index["annotation_id"].astype(str)
    return index


def select_zdisc_annotation_crops(index: pd.DataFrame, n_crops: int = 40, seed: int = 123) -> pd.DataFrame:
    table = index.copy(deep=True)
    if n_crops <= 0:
        return table.head(0).copy()
    if "oop_bin" not in table.columns:
        table["oop_bin"] = assign_oop_bins(table.get("patch_oop", pd.Series(np.nan, index=table.index)))
    table["valid_for_orientation_bool"] = bool_column(table, "valid_for_orientation")

    invalid_target = min(max(int(round(n_crops * 0.15)), 1), n_crops)
    invalid = table.loc[~table["valid_for_orientation_bool"]].copy()
    selected_frames: list[pd.DataFrame] = []
    invalid_selected = diverse_sample(invalid, invalid_target, seed + 11)
    if not invalid_selected.empty:
        selected_frames.append(invalid_selected)

    remaining = n_crops - len(invalid_selected)
    valid = table.loc[table["valid_for_orientation_bool"]].copy()
    if not invalid_selected.empty:
        valid = valid.loc[~valid["annotation_id"].isin(invalid_selected["annotation_id"])].copy()
    bins = ["low_oop", "medium_oop", "high_oop"]
    per_bin = distribute_counts(remaining, bins)
    for idx, bin_name in enumerate(bins):
        subset = valid.loc[valid["oop_bin"].astype(str) == bin_name].copy()
        selected = diverse_sample(subset, per_bin[bin_name], seed + idx)
        if not selected.empty:
            selected_frames.append(selected)

    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else table.head(0).copy()
    if len(selected) < n_crops:
        used = set(selected["annotation_id"].astype(str))
        fill_pool = table.loc[~table["annotation_id"].astype(str).isin(used)].copy()
        fill = diverse_sample(fill_pool, n_crops - len(selected), seed + 99)
        selected = pd.concat([selected, fill], ignore_index=True) if not fill.empty else selected

    return selected.head(n_crops).drop(columns=["valid_for_orientation_bool"], errors="ignore").copy()


def prepare_zdisc_annotation_set(
    cfg: dict[str, Any],
    n_crops: int = 40,
    seed: int = 123,
    annotation_index_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    out_dir = Path(output_directory) if output_directory else default_zdisc_output_dir(cfg)
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    for directory in [out_dir, images_dir, masks_dir, overlays_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    source_index = Path(annotation_index_path) if annotation_index_path else output_dir(cfg) / "annotation_pack" / "annotation_patch_index.csv"
    index = load_annotation_pack_index(source_index)
    selected = select_zdisc_annotation_crops(index, n_crops=n_crops, seed=seed)
    output_rows = []
    for _, row in selected.iterrows():
        crop_path = Path(str(row["crop_path"]))
        if not crop_path.exists():
            raise FileNotFoundError(f"Selected crop does not exist: {crop_path}")
        safe_id = safe_name(str(row["annotation_id"]))
        image_path = images_dir / f"{safe_id}.png"
        mask_path = masks_dir / f"{safe_id}_mask.png"
        overlay_path = overlays_dir / f"{safe_id}_overlay.png"
        if overwrite or not image_path.exists():
            shutil.copy2(crop_path, image_path)
        crop = Image.open(crop_path)
        if overwrite or not mask_path.exists():
            blank = np.zeros((crop.height, crop.width), dtype=np.uint8)
            Image.fromarray(blank, mode="L").save(mask_path)
        output_rows.append(build_zdisc_index_row(row, crop_path, image_path, mask_path, overlay_path))

    zdisc_index = pd.DataFrame(output_rows)
    for column in ZDISC_INDEX_COLUMNS:
        if column not in zdisc_index.columns:
            zdisc_index[column] = ""
    zdisc_index = zdisc_index[ZDISC_INDEX_COLUMNS]
    paths = {
        "index": out_dir / "zdisc_annotation_index.csv",
        "summary_json": out_dir / "zdisc_annotation_summary.json",
        "summary_txt": out_dir / "zdisc_annotation_summary.txt",
        "images_dir": images_dir,
        "masks_dir": masks_dir,
        "overlays_dir": overlays_dir,
    }
    summary = build_prepare_summary(zdisc_index, seed=seed, source_index=source_index)
    zdisc_index.to_csv(paths["index"], index=False)
    write_summary(summary, paths["summary_json"], paths["summary_txt"])
    return zdisc_index, summary, paths


def audit_zdisc_annotations(
    cfg: dict[str, Any],
    annotation_index_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    write_overlays: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    out_dir = Path(output_directory) if output_directory else default_zdisc_output_dir(cfg)
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(annotation_index_path) if annotation_index_path else out_dir / "zdisc_annotation_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Z-disc annotation index not found: {index_path}")
    index = pd.read_csv(index_path, dtype={"annotation_id": str, "image_id": str, "donor_id": str, "patch_id": str})
    rows = [audit_one_annotation(row, write_overlays=write_overlays) for _, row in index.iterrows()]
    audit = pd.DataFrame(rows)
    for column in ZDISC_AUDIT_COLUMNS:
        if column not in audit.columns:
            audit[column] = ""
    audit = audit[ZDISC_AUDIT_COLUMNS]
    summary = build_audit_summary(audit)
    paths = {
        "index": index_path,
        "summary_json": out_dir / "zdisc_annotation_summary.json",
        "summary_txt": out_dir / "zdisc_annotation_summary.txt",
        "overlays_dir": overlays_dir,
    }
    write_summary(summary, paths["summary_json"], paths["summary_txt"])
    return audit, summary, paths


def audit_one_annotation(row: pd.Series, write_overlays: bool = False) -> dict[str, Any]:
    result = {column: row.get(column, "") for column in ZDISC_INDEX_COLUMNS}
    image_path = Path(str(row.get("annotation_image_path", "")))
    mask_path = Path(str(row.get("mask_path", "")))
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
    image = np.asarray(Image.open(image_path))
    result["image_shape"] = shape_string(image.shape)
    if not mask_path.exists():
        result["mask_shape"] = "missing_mask"
        return result
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    result["mask_shape"] = shape_string(mask.shape)
    result["shape_matches"] = tuple(mask.shape[:2]) == tuple(image.shape[:2])
    unique = sorted(int(value) for value in np.unique(mask))
    invalid = [value for value in unique if value not in ZDISC_LABELS]
    result["allowed_labels_only"] = len(invalid) == 0
    result["invalid_label_values"] = ";".join(str(value) for value in invalid)
    for label in ZDISC_LABELS:
        result[f"label_{label}_pixels"] = int(np.sum(mask == label))
    result["empty_mask"] = int(result["label_1_pixels"]) == 0 and int(result["label_2_pixels"]) == 0
    result["has_zdisc_labels"] = int(result["label_1_pixels"]) > 0
    if write_overlays and result["shape_matches"]:
        overlay_path = Path(str(row.get("overlay_path", "")))
        write_label_overlay(image, mask, overlay_path)
    return result


def build_zdisc_index_row(
    row: pd.Series,
    source_crop_path: Path,
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
        "valid_for_orientation": str(row.get("valid_for_orientation", "")),
        "patch_oop": row.get("patch_oop", ""),
        "patch_mean_orientation_deg": row.get("patch_mean_orientation_deg", ""),
        "suggested_annotation_task": row.get("suggested_annotation_task", ""),
        "source_crop_path": str(source_crop_path),
        "annotation_image_path": str(annotation_image_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
    }


def build_prepare_summary(index: pd.DataFrame, seed: int, source_index: str | Path) -> dict[str, Any]:
    valid = bool_column(index, "valid_for_orientation")
    return json_safe(
        {
            "mode": "prepare_zdisc_annotation_set",
            "source_index": str(source_index),
            "selected_crops": int(len(index)),
            "seed": int(seed),
            "valid_orientation_crops": int(valid.sum()),
            "invalid_or_low_quality_controls": int((~valid).sum()),
            "oop_bin_counts": value_counts(index.get("oop_bin", pd.Series(dtype=object))),
            "unique_donors": int(index["donor_id"].nunique()) if "donor_id" in index else 0,
            "unique_images": int(index["image_id"].nunique()) if "image_id" in index else 0,
            "label_convention": ZDISC_LABELS,
            "purpose": "local manual Z-disc/striation mask annotation; not production segmentation and not ML training",
        }
    )


def build_audit_summary(audit: pd.DataFrame) -> dict[str, Any]:
    return json_safe(
        {
            "mode": "audit_zdisc_annotations",
            "selected_crops": int(len(audit)),
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


def write_label_overlay(image: np.ndarray, mask: np.ndarray, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(image)
    if values.ndim == 2:
        display = normalize_uint8(values)
        rgb = np.dstack([display, display, display]).astype(np.float32) / 255.0
    else:
        rgb = values[..., :3].astype(np.float32)
        if rgb.max(initial=0) > 1.0:
            rgb = rgb / 255.0
    label_mask = np.asarray(mask)
    colors = {
        1: np.array([1.0, 0.1, 0.1], dtype=np.float32),
        2: np.array([1.0, 0.8, 0.0], dtype=np.float32),
    }
    alpha = 0.45
    for label, color in colors.items():
        pixels = label_mask == label
        if np.any(pixels):
            rgb[pixels] = (1.0 - alpha) * rgb[pixels] + alpha * color
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB").save(out)
    return out


def write_summary(summary: dict[str, Any], json_path: str | Path, txt_path: str | Path) -> None:
    json_out = Path(json_path)
    txt_out = Path(txt_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    lines = [f"{key}: {value}" for key, value in summary.items()]
    txt_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assign_oop_bins(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    bins = pd.Series("medium_oop", index=values.index, dtype=object)
    bins.loc[numeric < 0.33] = "low_oop"
    bins.loc[numeric >= 0.66] = "high_oop"
    bins.loc[numeric.isna()] = "missing_oop"
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
    selected = []
    remaining = shuffled.copy()
    selected_count = 0
    while selected_count < count and not remaining.empty:
        take = remaining.drop_duplicates("donor_id", keep="first").drop_duplicates("image_id", keep="first")
        if take.empty:
            take = remaining
        take = take.head(count - selected_count)
        selected.append(take)
        selected_count += len(take)
        used = set(take["annotation_id"].astype(str))
        remaining = remaining.loc[~remaining["annotation_id"].astype(str).isin(used)].copy()
    return pd.concat(selected, ignore_index=True).head(count) if selected else table.head(0).copy()


def bool_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    values = df[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_string_dtype(values) or values.dtype == object:
        return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    return values.fillna(False).astype(bool)


def value_counts(values: pd.Series) -> dict[str, int]:
    if values.empty:
        return {}
    counts = values.fillna("missing").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def normalize_uint8(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)
    return (np.clip((arr - low) / (high - low), 0.0, 1.0) * 255).astype(np.uint8)


def shape_string(shape: tuple[int, ...]) -> str:
    return "x".join(str(int(value)) for value in shape[:2])


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value
