from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_dir
from .io import load_tiff
from .outputs import write_preview_png


ANNOTATION_INDEX_COLUMNS = [
    "annotation_id",
    "image_id",
    "donor_id",
    "patch_id",
    "x0",
    "y0",
    "x1",
    "y1",
    "patch_oop",
    "patch_mean_orientation_deg",
    "valid_for_orientation",
    "suggested_annotation_task",
]

ANNOTATION_TEMPLATE_COLUMNS = [
    "annotation_id",
    "image_id",
    "donor_id",
    "patch_id",
    "manual_dominant_orientation_deg",
    "manual_organisation_score",
    "manual_organisation_label",
    "visible_striations_yes_no",
    "manual_sarcomere_length_um_optional",
    "confidence_score",
    "annotator_id",
    "notes",
]

REQUIRED_PATCH_INPUT_COLUMNS = [
    "image_id",
    "donor_id",
    "patch_id",
    "x0",
    "y0",
    "x1",
    "y1",
    "patch_oop",
    "patch_mean_orientation_deg",
    "valid_for_orientation",
]


def default_annotation_output_dir(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "annotation_pack"


def load_annotation_inputs(
    cfg: dict[str, Any],
    patch_table: str | Path | None = None,
    image_table: str | Path | None = None,
    analysis_table: str | Path | None = None,
    manifest_table: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = output_dir(cfg) / "tables"
    patch_path = Path(patch_table) if patch_table else tables / "features_per_patch.csv"
    image_path = Path(image_table) if image_table else tables / "features_per_image.csv"
    analysis_path = Path(analysis_table) if analysis_table else tables / "analysis_per_image.csv"
    manifest_path = Path(manifest_table) if manifest_table else tables / "enriched_manifest.csv"
    return (
        pd.read_csv(patch_path, dtype={"image_id": str, "donor_id": str, "patch_id": str}),
        pd.read_csv(image_path, dtype={"image_id": str, "donor_id": str}),
        pd.read_csv(analysis_path, dtype={"image_id": str, "donor_id": str}),
        pd.read_csv(manifest_path, dtype={"image_id": str, "donor_id": str, "region_id": str}),
    )


def prepare_patch_candidates(patches: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    table = patches.copy(deep=True)
    require_columns(table, REQUIRED_PATCH_INPUT_COLUMNS, "patch table")
    table["image_id"] = table["image_id"].astype(str)
    table["donor_id"] = table["donor_id"].astype(str)
    table["patch_id"] = table["patch_id"].astype(str)
    table["patch_oop"] = pd.to_numeric(table["patch_oop"], errors="coerce")
    table["patch_mean_orientation_deg"] = pd.to_numeric(table["patch_mean_orientation_deg"], errors="coerce")
    table["valid_for_orientation"] = bool_column(table, "valid_for_orientation")
    image_paths = manifest[["image_id", "donor_id", "image_path"]].copy()
    image_paths["image_id"] = image_paths["image_id"].astype(str)
    image_paths["donor_id"] = image_paths["donor_id"].astype(str)
    merged = table.merge(image_paths, on=["image_id", "donor_id"], how="left")
    return merged


def select_annotation_patches(
    patch_candidates: pd.DataFrame,
    n_patches: int = 80,
    seed: int = 123,
    negative_control_fraction: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = patch_candidates.copy(deep=True)
    selected_frames: list[pd.DataFrame] = []
    selected_keys: set[str] = set()
    negative_target = max(0, min(int(round(n_patches * negative_control_fraction)), n_patches))
    invalid = candidates.loc[~bool_column(candidates, "valid_for_orientation")].copy()
    invalid_selected = diverse_sample(invalid, negative_target, seed + 17)
    if not invalid_selected.empty:
        invalid_selected["oop_bin"] = "invalid_control"
        invalid_selected["suggested_annotation_task"] = "negative_control_quality_review"
        selected_frames.append(invalid_selected)
        selected_keys.update(invalid_selected["patch_id"].astype(str))

    remaining_target = max(int(n_patches) - len(invalid_selected), 0)
    valid = candidates.loc[bool_column(candidates, "valid_for_orientation") & candidates["patch_oop"].notna()].copy()
    if selected_keys:
        valid = valid.loc[~valid["patch_id"].astype(str).isin(selected_keys)].copy()
    valid["oop_bin"] = assign_oop_bins(valid["patch_oop"])
    bins = ["low_oop", "medium_oop", "high_oop"]
    per_bin = distribute_counts(remaining_target, bins)
    for index, bin_name in enumerate(bins):
        subset = valid.loc[valid["oop_bin"] == bin_name].copy()
        selected = diverse_sample(subset, per_bin[bin_name], seed + index)
        if not selected.empty:
            selected["suggested_annotation_task"] = f"manual_orientation_oop_review_{bin_name}"
            selected_frames.append(selected)
            selected_keys.update(selected["patch_id"].astype(str))

    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame(columns=candidates.columns)
    if len(selected_all) < n_patches:
        already = set(selected_all["patch_id"].astype(str)) if "patch_id" in selected_all.columns else set()
        fill_pool = candidates.loc[~candidates["patch_id"].astype(str).isin(already)].copy()
        fill_pool["oop_bin"] = fill_pool.get("oop_bin", np.nan)
        fill_pool["suggested_annotation_task"] = np.where(
            bool_column(fill_pool, "valid_for_orientation"),
            "manual_orientation_oop_review_fill",
            "negative_control_quality_review",
        )
        fill = diverse_sample(fill_pool, n_patches - len(selected_all), seed + 99)
        selected_all = pd.concat([selected_all, fill], ignore_index=True) if not fill.empty else selected_all

    selected_all = selected_all.head(n_patches).copy()
    selected_all.insert(0, "annotation_id", [f"ANN_{idx + 1:04d}" for idx in range(len(selected_all))])
    index = stabilize_annotation_index(selected_all)
    summary = annotation_summary(index, candidates)
    return index, summary


def assign_oop_bins(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    bins = pd.Series("medium_oop", index=values.index, dtype=object)
    bins.loc[numeric < 0.33] = "low_oop"
    bins.loc[numeric >= 0.66] = "high_oop"
    bins.loc[numeric.isna()] = "missing_oop"
    return bins


def distribute_counts(total: int, bins: list[str]) -> dict[str, int]:
    if total <= 0:
        return {name: 0 for name in bins}
    base = total // len(bins)
    remainder = total % len(bins)
    return {name: base + (1 if idx < remainder else 0) for idx, name in enumerate(bins)}


def diverse_sample(table: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0 or table.empty:
        return table.head(0).copy()
    shuffled = table.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rounds = []
    remaining = shuffled.copy()
    selected_count = 0
    while selected_count < count:
        if remaining.empty:
            break
        one_per_donor = remaining.drop_duplicates("donor_id", keep="first")
        one_per_image = one_per_donor.drop_duplicates("image_id", keep="first")
        take = one_per_image.head(count - selected_count)
        if take.empty:
            take = remaining.head(count - selected_count)
        rounds.append(take)
        selected_count += len(take)
        used_patch_ids = set(take["patch_id"].astype(str))
        remaining = remaining.loc[~remaining["patch_id"].astype(str).isin(used_patch_ids)].copy()
    return pd.concat(rounds, ignore_index=True).head(count) if rounds else table.head(0).copy()


def stabilize_annotation_index(selected: pd.DataFrame) -> pd.DataFrame:
    result = selected.copy(deep=True)
    coord_map = {"x0": "x0", "y0": "y0", "x1": "x1", "y1": "y1"}
    _ = coord_map
    for column in ANNOTATION_INDEX_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    for column in ["x0", "y0", "x1", "y1"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    result["valid_for_orientation"] = bool_column(result, "valid_for_orientation")
    return result[ANNOTATION_INDEX_COLUMNS + [column for column in result.columns if column not in ANNOTATION_INDEX_COLUMNS]]


def annotation_template_from_index(index: pd.DataFrame) -> pd.DataFrame:
    template = pd.DataFrame(columns=ANNOTATION_TEMPLATE_COLUMNS)
    for column in ["annotation_id", "image_id", "donor_id", "patch_id"]:
        template[column] = index[column].astype(str).to_numpy()
    for column in ANNOTATION_TEMPLATE_COLUMNS:
        if column not in template.columns:
            template[column] = ""
    return template[ANNOTATION_TEMPLATE_COLUMNS]


def export_annotation_pack(
    cfg: dict[str, Any],
    patch_table: str | Path | None = None,
    image_table: str | Path | None = None,
    analysis_table: str | Path | None = None,
    manifest_table: str | Path | None = None,
    output_directory: str | Path | None = None,
    n_patches: int = 80,
    seed: int = 123,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    patches, images, analysis, manifest = load_annotation_inputs(cfg, patch_table, image_table, analysis_table, manifest_table)
    _ = images, analysis
    candidates = prepare_patch_candidates(patches, manifest)
    index, summary = select_annotation_patches(candidates, n_patches=n_patches, seed=seed)
    template = annotation_template_from_index(index)
    out_dir = Path(output_directory) if output_directory else default_annotation_output_dir(cfg)
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_paths = export_patch_crops(index, candidates, crops_dir, overwrite=overwrite)
    if crop_paths:
        index = index.copy()
        index["crop_path"] = index["annotation_id"].map(crop_paths)
    summary = {**summary, "crop_count": int(len(crop_paths)), "output_dir": str(out_dir)}
    paths = write_annotation_pack_outputs(index, template, summary, out_dir)
    return index, template, summary, paths


def export_patch_crops(index: pd.DataFrame, candidates: pd.DataFrame, crops_dir: Path, overwrite: bool = False) -> dict[str, str]:
    by_image_path = candidates.drop_duplicates("image_id").set_index("image_id")["image_path"].to_dict()
    image_cache: dict[str, np.ndarray] = {}
    crop_paths: dict[str, str] = {}
    for _, row in index.iterrows():
        image_id = str(row["image_id"])
        image_path = by_image_path.get(image_id)
        if not image_path or pd.isna(image_path):
            continue
        if image_id not in image_cache:
            image_cache[image_id] = load_tiff(Path(str(image_path)))
        image = image_cache[image_id]
        y0, y1 = int(row["y0"]), int(row["y1"])
        x0, x1 = int(row["x0"]), int(row["x1"])
        crop = image[max(y0, 0) : min(y1, image.shape[0]), max(x0, 0) : min(x1, image.shape[1])]
        path = crops_dir / f"{row['annotation_id']}__{safe_name(image_id)}__{safe_name(str(row['patch_id']))}.png"
        if overwrite or not path.exists():
            write_preview_png(crop, path)
        crop_paths[str(row["annotation_id"])] = str(path)
    return crop_paths


def write_annotation_pack_outputs(
    index: pd.DataFrame,
    template: pd.DataFrame,
    summary: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, Path]:
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "annotation_index": out_dir / "annotation_patch_index.csv",
        "annotation_summary": out_dir / "annotation_summary.json",
        "annotation_template": out_dir / "annotation_template.csv",
    }
    index.to_csv(paths["annotation_index"], index=False)
    template.to_csv(paths["annotation_template"], index=False)
    paths["annotation_summary"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    return paths


def annotation_summary(index: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, Any]:
    valid_selected = index.loc[bool_column(index, "valid_for_orientation")].copy()
    oop_bins = assign_oop_bins(valid_selected["patch_oop"]) if not valid_selected.empty else pd.Series(dtype=object)
    return json_safe(
        {
            "selected_patches": int(len(index)),
            "available_candidate_patches": int(len(candidates)),
            "valid_orientation_selected": int(bool_column(index, "valid_for_orientation").sum()),
            "invalid_control_selected": int((~bool_column(index, "valid_for_orientation")).sum()),
            "oop_bin_counts": value_counts(oop_bins),
            "task_counts": value_counts(index.get("suggested_annotation_task", pd.Series(dtype=object))),
            "unique_donors": int(index["donor_id"].nunique()) if "donor_id" in index.columns else 0,
            "unique_images": int(index["image_id"].nunique()) if "image_id" in index.columns else 0,
            "max_patches_per_donor": int(index["donor_id"].value_counts().max()) if len(index) else 0,
            "max_patches_per_image": int(index["image_id"].value_counts().max()) if len(index) else 0,
            "seeded_sampling": True,
            "purpose": "manual OOP/orientation validation pack; not ML training and not statistical validation",
        }
    )


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required {label} columns: {missing}")


def bool_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    values = df[column]
    if values.dtype == object:
        return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    return values.fillna(False).astype(bool)


def value_counts(values: pd.Series) -> dict[str, int]:
    if values.empty:
        return {}
    counts = values.fillna("missing").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


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
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
