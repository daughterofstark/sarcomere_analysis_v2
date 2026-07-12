from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from PIL import Image

from .config import output_dir
from .io import load_tiff
from .zdisc_annotation import json_safe


EXPERT_TEMPLATE_COLUMNS = [
    "annotation_id",
    "patch_filename",
    "striations_visible",
    "organisation_score",
    "dominant_orientation_deg",
    "confidence_score",
    "spacing_measurable",
    "manual_sarcomere_length_um_optional",
    "notes",
]

INTERNAL_KEY_COLUMNS = [
    "annotation_id",
    "patch_filename",
    "image_id",
    "donor_id",
    "patch_id",
    "oop_bin",
    "automated_patch_oop",
    "automated_patch_orientation_deg",
    "health_status",
    "source_image_path",
]

REQUIRED_PATCH_COLUMNS = [
    "image_id",
    "donor_id",
    "patch_id",
    "y0",
    "x0",
    "y1",
    "x1",
    "patch_oop",
    "valid_for_orientation",
]

INSTRUCTION_TEXT = """# Expert Annotation Instructions

This is a blinded manual review of alpha-actinin striation organisation.

Please score only what is visible in the patch. Do not infer from neighbouring tissue, donor identity, disease group, expected biology, or automated outputs.

## Fields

- `striations_visible`: allowed values are `yes`, `unclear`, or `no`.
- `organisation_score`:
  - 1 = disorganised / no coherent striation orientation
  - 2 = weakly organised
  - 3 = moderately organised
  - 4 = strongly organised
  - 5 = highly organised
- `dominant_orientation_deg`:
  - 0 degrees = horizontal left-right striations
  - 90 degrees = vertical top-bottom striations
  - use an axial 0-180 degree convention
  - leave blank if unclear
- `confidence_score`:
  - 1 = very low confidence
  - 2 = low
  - 3 = moderate
  - 4 = high
  - 5 = very high
- `spacing_measurable`: allowed values are `yes`, `unclear`, or `no`.
- `manual_sarcomere_length_um_optional`: leave blank unless at least 3 adjacent Z-disc intervals are clearly visible.

Be conservative. Ambiguous or faint patches should be marked `unclear` or low confidence rather than forced into a confident score.
"""


def default_expert_pack_dir(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "expert_annotation_pack"


def load_expert_pack_inputs(
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
        pd.read_csv(image_path, dtype={"image_id": str, "donor_id": str}) if image_path.exists() else pd.DataFrame(),
        pd.read_csv(analysis_path, dtype={"image_id": str, "donor_id": str}) if analysis_path.exists() else pd.DataFrame(),
        pd.read_csv(manifest_path, dtype={"image_id": str, "donor_id": str}) if manifest_path.exists() else pd.DataFrame(),
    )


def export_expert_annotation_pack(
    cfg: dict[str, Any],
    patch_table: str | Path | None = None,
    image_table: str | Path | None = None,
    analysis_table: str | Path | None = None,
    manifest_table: str | Path | None = None,
    output_directory: str | Path | None = None,
    n_total: int = 75,
    n_per_bin: int | None = None,
    seed: int = 123,
    max_per_donor: int = 4,
    max_per_image: int = 3,
    write_zip: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    patches, images, analysis, manifest = load_expert_pack_inputs(
        cfg,
        patch_table=patch_table,
        image_table=image_table,
        analysis_table=analysis_table,
        manifest_table=manifest_table,
    )
    _ = images
    candidates = prepare_expert_candidates(patches, analysis=analysis, manifest=manifest)
    targets = target_counts(n_total=n_total, n_per_bin=n_per_bin)
    selected, selection_summary = select_expert_patches(
        candidates,
        targets=targets,
        seed=seed,
        max_per_donor=max_per_donor,
        max_per_image=max_per_image,
    )
    out_dir = Path(output_directory) if output_directory else default_expert_pack_dir(cfg)
    patch_dir = out_dir / "patches"
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)
    selected = assign_anonymous_ids(selected)
    export_blinded_patch_pngs(selected, patch_dir)
    template = expert_template_from_selected(selected)
    internal_key = internal_key_from_selected(selected)
    summary = expert_pack_summary(selected, candidates, targets, selection_summary, write_zip=write_zip)
    paths = write_expert_pack_outputs(out_dir, template, internal_key, summary)
    if write_zip:
        paths["zip"] = write_expert_pack_zip(out_dir, paths)
    return selected, template, internal_key, summary, paths


def prepare_expert_candidates(patches: pd.DataFrame, analysis: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    table = patches.copy(deep=True)
    require_columns(table, REQUIRED_PATCH_COLUMNS, "patch feature table")
    for column in ["image_id", "donor_id", "patch_id"]:
        table[column] = table[column].fillna("").astype(str)
    table["patch_oop"] = pd.to_numeric(table["patch_oop"], errors="coerce")
    table["patch_mean_orientation_deg"] = pd.to_numeric(table.get("patch_mean_orientation_deg", np.nan), errors="coerce")
    table["valid_for_orientation"] = bool_column(table, "valid_for_orientation")
    for column in ["y0", "x0", "y1", "x1"]:
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("Int64")
    metadata = image_metadata(analysis, manifest)
    table = table.merge(metadata, on=["image_id", "donor_id"], how="left")
    table = table.loc[table["valid_for_orientation"] & table["patch_oop"].notna() & table["image_path"].notna()].copy()
    table["oop_bin"] = assign_quantile_oop_bins(table["patch_oop"])
    table["health_status"] = health_status(table)
    table = table.rename(columns={"image_path": "source_image_path"})
    return table.reset_index(drop=True)


def image_metadata(analysis: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for source in [analysis, manifest]:
        if source.empty or "image_id" not in source.columns or "donor_id" not in source.columns:
            continue
        columns = [column for column in ["image_id", "donor_id", "image_path", "is_healthy"] if column in source.columns]
        if columns:
            frames.append(source[columns].copy())
    if not frames:
        return pd.DataFrame(columns=["image_id", "donor_id", "image_path", "is_healthy"])
    metadata = pd.concat(frames, ignore_index=True)
    metadata["image_id"] = metadata["image_id"].astype(str)
    metadata["donor_id"] = metadata["donor_id"].astype(str)
    metadata = metadata.drop_duplicates(["image_id", "donor_id"], keep="first")
    if "image_path" not in metadata.columns:
        metadata["image_path"] = np.nan
    if "is_healthy" not in metadata.columns:
        metadata["is_healthy"] = np.nan
    return metadata[["image_id", "donor_id", "image_path", "is_healthy"]]


def assign_quantile_oop_bins(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    ranks = numeric.rank(method="first")
    try:
        bins = pd.qcut(ranks, q=3, labels=["low", "medium", "high"])
    except ValueError:
        bins = pd.cut(ranks, bins=3, labels=["low", "medium", "high"], include_lowest=True)
    return pd.Series(bins.astype(str), index=values.index)


def health_status(table: pd.DataFrame) -> pd.Series:
    if "is_healthy" not in table.columns:
        return pd.Series("unknown", index=table.index)
    values = table["is_healthy"]
    if values.dtype == object:
        healthy = values.fillna("").astype(str).str.lower().isin({"true", "1", "yes", "healthy"})
    else:
        healthy = values.fillna(False).astype(bool)
    return pd.Series(np.where(healthy, "healthy", "non_healthy_or_unknown"), index=table.index)


def target_counts(n_total: int = 75, n_per_bin: int | None = None) -> dict[str, int]:
    if n_per_bin is not None:
        return {"low": int(n_per_bin), "medium": int(n_per_bin), "high": int(n_per_bin)}
    per_bin = int(n_total) // 3
    remainder = int(n_total) % 3
    return {
        "low": per_bin + (1 if remainder > 0 else 0),
        "medium": per_bin + (1 if remainder > 1 else 0),
        "high": per_bin,
    }


def select_expert_patches(
    candidates: pd.DataFrame,
    targets: dict[str, int],
    seed: int = 123,
    max_per_donor: int = 4,
    max_per_image: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_rows: list[pd.Series] = []
    selected_patch_ids: set[str] = set()
    donor_counts: dict[str, int] = {}
    image_counts: dict[str, int] = {}
    selected_by_bin = {name: 0 for name in targets}
    for offset, bin_name in enumerate(["low", "medium", "high"]):
        subset = candidates.loc[candidates["oop_bin"] == bin_name].copy()
        subset = subset.sample(frac=1.0, random_state=int(seed) + offset).reset_index(drop=True)
        for _, row in subset.iterrows():
            if selected_by_bin[bin_name] >= int(targets[bin_name]):
                break
            if can_select(row, selected_patch_ids, donor_counts, image_counts, max_per_donor, max_per_image):
                add_selection(row, selected_rows, selected_patch_ids, donor_counts, image_counts)
                selected_by_bin[bin_name] += 1
    selected = pd.DataFrame(selected_rows)
    shortfall = {bin_name: int(targets[bin_name] - selected_by_bin.get(bin_name, 0)) for bin_name in targets}
    return selected.reset_index(drop=True), {
        "target_bin_counts": {key: int(value) for key, value in targets.items()},
        "selected_bin_counts": {key: int(value) for key, value in selected_by_bin.items()},
        "bin_shortfall": shortfall,
        "max_per_donor": int(max_per_donor),
        "max_per_image": int(max_per_image),
    }


def can_select(
    row: pd.Series,
    selected_patch_ids: set[str],
    donor_counts: dict[str, int],
    image_counts: dict[str, int],
    max_per_donor: int,
    max_per_image: int,
) -> bool:
    patch_id = str(row["patch_id"])
    donor_id = str(row["donor_id"])
    image_id = str(row["image_id"])
    return (
        patch_id not in selected_patch_ids
        and donor_counts.get(donor_id, 0) < int(max_per_donor)
        and image_counts.get(image_id, 0) < int(max_per_image)
    )


def add_selection(
    row: pd.Series,
    selected_rows: list[pd.Series],
    selected_patch_ids: set[str],
    donor_counts: dict[str, int],
    image_counts: dict[str, int],
) -> None:
    selected_rows.append(row.copy())
    selected_patch_ids.add(str(row["patch_id"]))
    donor_counts[str(row["donor_id"])] = donor_counts.get(str(row["donor_id"]), 0) + 1
    image_counts[str(row["image_id"])] = image_counts.get(str(row["image_id"]), 0) + 1


def assign_anonymous_ids(selected: pd.DataFrame) -> pd.DataFrame:
    result = selected.copy(deep=True).reset_index(drop=True)
    result.insert(0, "annotation_id", [f"EXPERT_{idx + 1:04d}" for idx in range(len(result))])
    result["patch_filename"] = result["annotation_id"] + ".png"
    return result


def export_blinded_patch_pngs(selected: pd.DataFrame, patch_dir: Path) -> None:
    image_cache: dict[str, np.ndarray] = {}
    for _, row in selected.iterrows():
        image_path = str(row["source_image_path"])
        if image_path not in image_cache:
            image_cache[image_path] = load_tiff(image_path)
        image = image_cache[image_path]
        crop = crop_patch(image, row)
        write_display_png(crop, patch_dir / str(row["patch_filename"]))


def crop_patch(image: np.ndarray, row: pd.Series) -> np.ndarray:
    y0, y1 = int(row["y0"]), int(row["y1"])
    x0, x1 = int(row["x0"]), int(row["x1"])
    return np.asarray(image[max(y0, 0) : min(y1, image.shape[0]), max(x0, 0) : min(x1, image.shape[1])])


def write_display_png(crop: np.ndarray, path: Path) -> Path:
    values = np.asarray(crop, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        scaled = np.zeros(values.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, [1.0, 99.0])
        if hi <= lo:
            hi = float(np.max(finite)) if finite.size else 1.0
            lo = float(np.min(finite)) if finite.size else 0.0
        if hi <= lo:
            scaled = np.zeros(values.shape, dtype=np.uint8)
        else:
            scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
            scaled = np.asarray(scaled * 255.0, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(scaled, mode="L").save(path)
    return path


def expert_template_from_selected(selected: pd.DataFrame) -> pd.DataFrame:
    template = pd.DataFrame(columns=EXPERT_TEMPLATE_COLUMNS)
    template["annotation_id"] = selected["annotation_id"].astype(str)
    template["patch_filename"] = selected["patch_filename"].astype(str)
    for column in EXPERT_TEMPLATE_COLUMNS:
        if column not in template.columns:
            template[column] = ""
    return template[EXPERT_TEMPLATE_COLUMNS]


def internal_key_from_selected(selected: pd.DataFrame) -> pd.DataFrame:
    key = pd.DataFrame(
        {
            "annotation_id": selected["annotation_id"].astype(str),
            "patch_filename": selected["patch_filename"].astype(str),
            "image_id": selected["image_id"].astype(str),
            "donor_id": selected["donor_id"].astype(str),
            "patch_id": selected["patch_id"].astype(str),
            "oop_bin": selected["oop_bin"].astype(str),
            "automated_patch_oop": selected["patch_oop"],
            "automated_patch_orientation_deg": selected.get("patch_mean_orientation_deg", np.nan),
            "health_status": selected.get("health_status", "unknown"),
            "source_image_path": selected["source_image_path"].astype(str),
        }
    )
    return key[INTERNAL_KEY_COLUMNS]


def expert_pack_summary(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    targets: dict[str, int],
    selection_summary: dict[str, Any],
    write_zip: bool = False,
) -> dict[str, Any]:
    bin_counts = value_counts(selected["oop_bin"]) if len(selected) else {}
    donor_counts = selected["donor_id"].value_counts() if len(selected) else pd.Series(dtype=int)
    image_counts = selected["image_id"].value_counts() if len(selected) else pd.Series(dtype=int)
    return json_safe(
        {
            "mode": "blinded_expert_annotation_pack",
            "selected_patches": int(len(selected)),
            "available_candidate_patches": int(len(candidates)),
            "target_bin_counts": {key: int(value) for key, value in targets.items()},
            "oop_bin_counts": bin_counts,
            "bin_shortfall": selection_summary["bin_shortfall"],
            "unique_donors": int(selected["donor_id"].nunique()) if len(selected) else 0,
            "unique_images": int(selected["image_id"].nunique()) if len(selected) else 0,
            "max_patches_per_donor": int(donor_counts.max()) if len(donor_counts) else 0,
            "max_patches_per_image": int(image_counts.max()) if len(image_counts) else 0,
            "donor_cap": int(selection_summary["max_per_donor"]),
            "image_cap": int(selection_summary["max_per_image"]),
            "anonymous_filenames": True,
            "expert_template_blinded": True,
            "internal_key_for_project_team_only": True,
            "zip_requested": bool(write_zip),
            "zip_excludes_internal_key": bool(write_zip),
            "purpose": "blinded expert review of striation visibility, organisation, dominant orientation, and confidence; no validation statistics.",
        }
    )


def write_expert_pack_outputs(out_dir: Path, template: pd.DataFrame, internal_key: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "patch_dir": out_dir / "patches",
        "template_csv": out_dir / "expert_annotation_template.csv",
        "internal_key_csv": out_dir / "internal_blinding_key.csv",
        "instructions_md": out_dir / "annotation_instructions.md",
        "summary_json": out_dir / "expert_annotation_pack_summary.json",
        "summary_txt": out_dir / "expert_annotation_pack_summary.txt",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    template.to_csv(paths["template_csv"], index=False)
    internal_key.to_csv(paths["internal_key_csv"], index=False)
    paths["instructions_md"].write_text(INSTRUCTION_TEXT, encoding="utf-8")
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_summary_text(summary), encoding="utf-8")
    return paths


def render_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Blinded expert annotation pack summary",
        f"selected_patches: {summary['selected_patches']}",
        f"oop_bin_counts: {summary['oop_bin_counts']}",
        f"unique_donors: {summary['unique_donors']}",
        f"unique_images: {summary['unique_images']}",
        f"max_patches_per_donor: {summary['max_patches_per_donor']}",
        f"max_patches_per_image: {summary['max_patches_per_image']}",
        "Expert-facing files contain anonymous patch IDs only.",
        "Internal blinding key is for the project team only and must not be sent to the reviewer.",
    ]
    return "\n".join(lines) + "\n"


def write_expert_pack_zip(out_dir: Path, paths: dict[str, Path]) -> Path:
    zip_path = out_dir / "expert_annotation_pack_for_natalia.zip"
    allowed_files = [
        paths["template_csv"],
        paths["instructions_md"],
        paths["summary_txt"],
    ]
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        for patch in sorted((out_dir / "patches").glob("*.png")):
            archive.write(patch, arcname=str(Path("patches") / patch.name))
        for path in allowed_files:
            archive.write(path, arcname=path.name)
    return zip_path


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
    return {str(key): int(value) for key, value in values.fillna("missing").astype(str).value_counts().sort_index().items()}
