from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from .config import output_dir
from .zdisc_annotation import json_safe


DEFAULT_REVIEW_IMAGES = ["5138", "6052-CLEAR_STRIPES", "3112", "7028"]

REVIEW_SUMMARY_COLUMNS = [
    "filename",
    "expected_positive_example",
    "noted_complex_example",
    "candidate_patch_fraction",
    "valid_selected_spacing_count",
    "selected_spacing_valid_fraction",
    "selected_spacing_median_um",
    "selected_spacing_iqr_um",
    "selected_region_median_oop",
    "all_region_median_oop",
    "selected_vs_all_oop_difference",
    "review_flag",
]

PREVIEW_PATTERNS = {
    "selected_candidate_mask_overlay": "{image_id}_selected_candidate_overlay.png",
    "spacing_candidate_overlay": "{image_id}_confocal_spacing_candidate_overlay.png",
    "valid_spacing_patch_overlay": "{image_id}_confocal_valid_spacing_overlay.png",
    "spacing_um_heatmap": "{image_id}_confocal_spacing_um_heatmap.png",
    "same_grid_candidate_overlay": "{image_id}_same_grid_candidate_overlay.png",
    "same_grid_oop_heatmap": "{image_id}_same_grid_oop_heatmap.png",
}


def default_confocal_review_pack_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_review_pack"
    return {
        "root": root,
        "review_images": root / "review_images",
        "summary_csv": root / "confocal_review_summary.csv",
        "notes_md": root / "confocal_review_notes_for_natalia.md",
        "summary_json": root / "confocal_review_pack_summary.json",
        "summary_txt": root / "confocal_review_pack_summary.txt",
        "zip": root / "confocal_review_pack_for_natalia.zip",
    }


def export_confocal_review_pack(
    cfg: dict[str, Any],
    images: list[str] | None = None,
    output_directory: str | Path | None = None,
    write_zip: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    selected_images = images or DEFAULT_REVIEW_IMAGES
    paths = default_confocal_review_pack_paths(cfg, output_directory)
    inputs = load_confocal_review_inputs(cfg)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["review_images"].mkdir(parents=True, exist_ok=True)

    copied, missing = copy_review_images(cfg, selected_images, paths["review_images"])
    review_summary = build_review_summary(inputs, selected_images)
    notes = render_notes_for_natalia(selected_images)
    summary = build_review_pack_summary(selected_images, review_summary, copied, missing, write_zip)
    write_review_pack_outputs(review_summary, notes, summary, paths)
    if write_zip:
        paths["zip"] = write_review_pack_zip(paths)
        summary["zip_path"] = str(paths["zip"])
        paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
        paths["summary_txt"].write_text(render_summary_text(summary), encoding="utf-8")
    return review_summary, summary, paths


def load_confocal_review_inputs(cfg: dict[str, Any]) -> dict[str, pd.DataFrame | str]:
    root = output_dir(cfg)
    return {
        "pilot_text": read_text_if_exists(root / "confocal_pilot" / "confocal_pilot_interpretation.txt"),
        "spacing_patch": read_csv_if_exists(root / "confocal_spacing_audit" / "confocal_spacing_per_patch.csv"),
        "spacing_image": read_csv_if_exists(root / "confocal_spacing_audit" / "confocal_spacing_per_image.csv"),
        "oop_image": read_csv_if_exists(root / "confocal_same_grid_oop" / "confocal_same_grid_oop_per_image.csv"),
        "metadata": read_csv_if_exists(root / "confocal_metadata" / "confocal_metadata_calibration.csv"),
    }


def build_review_summary(inputs: dict[str, pd.DataFrame | str], images: list[str]) -> pd.DataFrame:
    spacing_image = inputs["spacing_image"]
    oop_image = inputs["oop_image"]
    if not isinstance(spacing_image, pd.DataFrame):
        spacing_image = pd.DataFrame()
    if not isinstance(oop_image, pd.DataFrame):
        oop_image = pd.DataFrame()
    spacing_image = ensure_string_id(spacing_image)
    oop_image = ensure_string_id(oop_image)
    rows = []
    for image_id in images:
        spacing = first_image_row(spacing_image, image_id)
        oop = first_image_row(oop_image, image_id)
        candidate_count = to_float(spacing.get("candidate_patch_count"))
        total = to_float(spacing.get("total_patches"))
        rows.append(
            {
                "filename": spacing.get("filename") or oop.get("filename") or f"{image_id}.tif",
                "expected_positive_example": bool_value(spacing.get("expected_positive_example", oop.get("expected_positive_example", False))),
                "noted_complex_example": bool_value(spacing.get("noted_complex_example", oop.get("noted_complex_example", False))),
                "candidate_patch_fraction": candidate_count / total if np.isfinite(candidate_count) and np.isfinite(total) and total else np.nan,
                "valid_selected_spacing_count": spacing.get("spacing_valid_patch_count_selected", np.nan),
                "selected_spacing_valid_fraction": spacing.get("spacing_valid_fraction_selected", np.nan),
                "selected_spacing_median_um": spacing.get("selected_median_spacing_um", np.nan),
                "selected_spacing_iqr_um": spacing.get("selected_iqr_spacing_um", np.nan),
                "selected_region_median_oop": oop.get("selected_region_median_oop_128", np.nan),
                "all_region_median_oop": oop.get("all_region_median_oop_128", np.nan),
                "selected_vs_all_oop_difference": oop.get("selected_vs_all_oop_difference_128", np.nan),
                "review_flag": review_flag(str(image_id), spacing, oop),
            }
        )
    return pd.DataFrame(rows, columns=REVIEW_SUMMARY_COLUMNS)


def copy_review_images(cfg: dict[str, Any], images: list[str], review_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    root = output_dir(cfg)
    source_dirs = {
        "selected_candidate_mask_overlay": root / "confocal_selective_analysis" / "previews",
        "spacing_candidate_overlay": root / "confocal_spacing_audit" / "previews",
        "valid_spacing_patch_overlay": root / "confocal_spacing_audit" / "previews",
        "spacing_um_heatmap": root / "confocal_spacing_audit" / "previews",
        "same_grid_candidate_overlay": root / "confocal_same_grid_oop" / "previews",
        "same_grid_oop_heatmap": root / "confocal_same_grid_oop" / "previews",
    }
    copied: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for image_id in images:
        for kind, pattern in PREVIEW_PATTERNS.items():
            source = source_dirs[kind] / pattern.format(image_id=image_id)
            destination = review_dir / f"{image_id}_{kind}.png"
            if source.exists():
                shutil.copy2(source, destination)
                copied.append({"image_id": image_id, "preview_type": kind, "path": str(destination)})
            else:
                missing.append({"image_id": image_id, "preview_type": kind, "expected_source": str(source)})
    return copied, missing


def render_notes_for_natalia(images: list[str]) -> str:
    image_list = ", ".join(images)
    return f"""# Confocal Review Notes For Natalia

This is a small exploratory review pack for the confocal pilot images: {image_list}.

Please inspect whether the candidate confident-striation regions and calibrated spacing overlays look biologically reasonable.

## What To Check

- For `5138` and `6052-CLEAR_STRIPES`, do the selected candidate regions overlap convincing Z-disc/striated structures?
- Do the valid spacing patches look like they are measuring real adjacent Z-disc intervals?
- Should `3112` be treated as a negative/complex example where Z-disc-like structures do not form clear striations?
- Is `7028` too broadly selected, given its high candidate fraction?

## Important Caveats

- These selected regions are candidate confident-striation regions, not final segmentation.
- Spacing estimates are calibrated per image using confocal metadata.
- This is exploratory QC/review material, not a publication figure set.
- Please do not treat the spacing values as biological conclusions until the overlays have been visually reviewed.
"""


def build_review_pack_summary(
    images: list[str],
    review_summary: pd.DataFrame,
    copied: list[dict[str, str]],
    missing: list[dict[str, str]],
    write_zip: bool,
) -> dict[str, Any]:
    return json_safe(
        {
            "mode": "confocal_review_pack",
            "images_included": images,
            "image_count": len(images),
            "review_image_files_copied": len(copied),
            "missing_preview_files": missing,
            "missing_preview_count": len(missing),
            "summary_rows": int(len(review_summary)),
            "write_zip_requested": bool(write_zip),
            "zip_path": None,
            "interpretation": [
                "Shareable visual QC pack for Natalia.",
                "Candidate regions are exploratory and not final segmentation.",
                "Spacing estimates use per-image confocal calibration.",
                "No algorithms, thresholds, widefield outputs, or existing confocal outputs were changed.",
            ],
        }
    )


def write_review_pack_outputs(
    review_summary: pd.DataFrame,
    notes: str,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    review_summary.to_csv(paths["summary_csv"], index=False)
    paths["notes_md"].write_text(notes, encoding="utf-8")
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_summary_text(summary), encoding="utf-8")


def write_review_pack_zip(paths: dict[str, Path]) -> Path:
    zip_path = paths["zip"]
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for image_path in sorted(paths["review_images"].glob("*.png")):
            archive.write(image_path, arcname=f"review_images/{image_path.name}")
        archive.write(paths["summary_csv"], arcname=paths["summary_csv"].name)
        archive.write(paths["notes_md"], arcname=paths["notes_md"].name)
        archive.write(paths["summary_txt"], arcname=paths["summary_txt"].name)
    return zip_path


def render_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal review pack",
        f"images_included: {summary['images_included']}",
        f"review_image_files_copied: {summary['review_image_files_copied']}",
        f"missing_preview_count: {summary['missing_preview_count']}",
        f"zip_path: {summary.get('zip_path')}",
        "",
        "Missing previews:",
    ]
    lines.extend(f"- {item}" for item in summary["missing_preview_files"])
    lines.append("")
    lines.extend(summary["interpretation"])
    return "\n".join(lines) + "\n"


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    dtype = {"confocal_image_id": str, "filename": str, "patch_id": str}
    return pd.read_csv(path, dtype={key: value for key, value in dtype.items() if key in pd.read_csv(path, nrows=0).columns})


def read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def ensure_string_id(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    copy = table.copy(deep=True)
    if "confocal_image_id" in copy.columns:
        copy["confocal_image_id"] = copy["confocal_image_id"].astype(str)
    return copy


def first_image_row(table: pd.DataFrame, image_id: str) -> dict[str, Any]:
    if table.empty or "confocal_image_id" not in table.columns:
        return {}
    match = table.loc[table["confocal_image_id"].astype(str) == str(image_id)]
    return match.iloc[0].to_dict() if not match.empty else {}


def review_flag(image_id: str, spacing: dict[str, Any], oop: dict[str, Any]) -> str:
    flags: list[str] = []
    if bool_value(spacing.get("expected_positive_example", oop.get("expected_positive_example", False))):
        flags.append("expected_positive_example")
    if bool_value(spacing.get("noted_complex_example", oop.get("noted_complex_example", False))):
        flags.append("noted_complex_example")
    candidate_fraction = to_float(spacing.get("spacing_valid_fraction_selected"))
    if image_id == "7028" or candidate_fraction > 0.6:
        flags.append("broad_or_high_yield_review_needed")
    if not flags:
        flags.append("review_needed")
    return ";".join(flags)


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
