from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from .config import output_dir
from .confocal_gate_refinement import DEFAULT_FOCUS_IMAGES
from .zdisc_annotation import json_safe


DEFAULT_BASELINE_VARIANT = "moderate_reference"
DEFAULT_RELAXED_VARIANT = "moderate_relaxed_combined"

GATE_REVIEW_SUMMARY_COLUMNS = [
    "filename",
    "moderate_candidate_fraction",
    "relaxed_candidate_fraction",
    "added_patch_count",
    "relaxed_classification",
    "review_flag",
    "previous_spacing_median_um",
    "previous_valid_spacing_count",
    "spacing_caveat",
]

REVIEW_IMAGE_TYPES = [
    "moderate_reference_overlay",
    "moderate_relaxed_combined_overlay",
    "added_vs_moderate_overlay",
    "moderate_spacing_context_overlay",
]


def default_gate_review_pack_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_gate_review_pack"
    return {
        "root": root,
        "review_images": root / "review_images",
        "summary_csv": root / "confocal_gate_review_summary.csv",
        "notes_md": root / "confocal_gate_review_notes_for_natalia.md",
        "summary_json": root / "confocal_gate_review_pack_summary.json",
        "summary_txt": root / "confocal_gate_review_pack_summary.txt",
        "zip": root / "confocal_gate_review_pack_for_natalia.zip",
    }


def export_confocal_gate_review_pack(
    cfg: dict[str, Any],
    baseline_variant: str = DEFAULT_BASELINE_VARIANT,
    relaxed_variant: str = DEFAULT_RELAXED_VARIANT,
    images: list[str] | None = None,
    output_directory: str | Path | None = None,
    write_zip: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    focus_images = [str(image) for image in (images or DEFAULT_FOCUS_IMAGES)]
    paths = default_gate_review_pack_paths(cfg, output_directory)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["review_images"].mkdir(parents=True, exist_ok=True)

    inputs = load_gate_review_inputs(cfg)
    review_summary = build_gate_review_summary_table(
        inputs,
        focus_images,
        baseline_variant=baseline_variant,
        relaxed_variant=relaxed_variant,
    )
    copied, missing = copy_gate_review_images(
        cfg,
        focus_images,
        paths["review_images"],
        baseline_variant=baseline_variant,
        relaxed_variant=relaxed_variant,
    )
    notes = render_gate_review_notes_for_natalia(focus_images, baseline_variant, relaxed_variant)
    summary = build_gate_review_pack_summary(
        review_summary,
        copied,
        missing,
        focus_images,
        baseline_variant=baseline_variant,
        relaxed_variant=relaxed_variant,
        write_zip=write_zip,
    )
    write_gate_review_pack_outputs(review_summary, notes, summary, paths)
    if write_zip:
        paths["zip"] = write_gate_review_pack_zip(paths)
        summary["zip_path"] = str(paths["zip"])
        summary["zip_excludes_internal_tables"] = True
        paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
        paths["summary_txt"].write_text(render_gate_review_summary_text(summary), encoding="utf-8")
    return review_summary, summary, paths


def load_gate_review_inputs(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    root = output_dir(cfg)
    return {
        "per_image": read_csv_if_exists(root / "confocal_gate_refinement" / "confocal_gate_refinement_per_image.csv"),
        "variants": read_csv_if_exists(root / "confocal_gate_refinement" / "confocal_gate_refinement_variants.csv"),
        "previous_review": read_csv_if_exists(root / "confocal_review_pack" / "confocal_review_summary.csv"),
    }


def build_gate_review_summary_table(
    inputs: dict[str, pd.DataFrame],
    images: list[str],
    baseline_variant: str = DEFAULT_BASELINE_VARIANT,
    relaxed_variant: str = DEFAULT_RELAXED_VARIANT,
) -> pd.DataFrame:
    per_image = normalize_ids(inputs.get("per_image", pd.DataFrame()))
    variants = inputs.get("variants", pd.DataFrame()).copy(deep=True)
    previous_review = normalize_ids(inputs.get("previous_review", pd.DataFrame()))
    rows: list[dict[str, Any]] = []
    relaxed_classification = variant_classification(variants, relaxed_variant)
    for image_id in images:
        baseline = first_variant_row(per_image, image_id, baseline_variant)
        relaxed = first_variant_row(per_image, image_id, relaxed_variant)
        previous = first_previous_review_row(previous_review, image_id)
        filename = (
            relaxed.get("filename")
            or baseline.get("filename")
            or previous.get("filename")
            or f"{image_id}.tif"
        )
        review_flag = relaxed.get("review_flag", "")
        rows.append(
            {
                "filename": str(filename),
                "moderate_candidate_fraction": relaxed_float(baseline.get("candidate_fraction")),
                "relaxed_candidate_fraction": relaxed_float(relaxed.get("candidate_fraction")),
                "added_patch_count": relaxed_int(relaxed.get("added_vs_moderate_count")),
                "relaxed_classification": relaxed.get("classification", relaxed_classification) or relaxed_classification,
                "review_flag": review_flag or focus_review_flag(str(image_id), relaxed_float(relaxed.get("candidate_fraction"))),
                "previous_spacing_median_um": relaxed_float(
                    previous.get("selected_spacing_median_um", baseline.get("selected_spacing_median_um"))
                ),
                "previous_valid_spacing_count": relaxed_int(
                    previous.get("valid_selected_spacing_count", baseline.get("selected_valid_spacing_count"))
                ),
                "spacing_caveat": "spacing_from_moderate_gate_not_refreshed_for_relaxed_patches",
            }
        )
    return pd.DataFrame(rows, columns=GATE_REVIEW_SUMMARY_COLUMNS)


def copy_gate_review_images(
    cfg: dict[str, Any],
    images: list[str],
    review_dir: Path,
    baseline_variant: str = DEFAULT_BASELINE_VARIANT,
    relaxed_variant: str = DEFAULT_RELAXED_VARIANT,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    root = output_dir(cfg)
    refinement_previews = root / "confocal_gate_refinement" / "previews"
    previous_review_images = root / "confocal_review_pack" / "review_images"
    copied: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for image_id in images:
        planned = [
            (
                "moderate_reference_overlay",
                refinement_previews / f"{image_id}_{baseline_variant}_gate_refinement_overlay.png",
                review_dir / f"{image_id}_moderate_reference_overlay.png",
            ),
            (
                "moderate_relaxed_combined_overlay",
                refinement_previews / f"{image_id}_{relaxed_variant}_gate_refinement_overlay.png",
                review_dir / f"{image_id}_{relaxed_variant}_overlay.png",
            ),
            (
                "added_vs_moderate_overlay",
                refinement_previews / f"{image_id}_{relaxed_variant}_gate_refinement_overlay.png",
                review_dir / f"{image_id}_added_vs_moderate_overlay.png",
            ),
            (
                "moderate_spacing_context_overlay",
                previous_review_images / f"{image_id}_valid_spacing_patch_overlay.png",
                review_dir / f"{image_id}_moderate_spacing_context_overlay.png",
            ),
        ]
        for image_type, source, destination in planned:
            if source.exists():
                shutil.copy2(source, destination)
                copied.append({"image_id": image_id, "image_type": image_type, "path": str(destination)})
            else:
                missing.append({"image_id": image_id, "image_type": image_type, "expected_source": str(source)})
    return copied, missing


def render_gate_review_notes_for_natalia(
    images: list[str],
    baseline_variant: str = DEFAULT_BASELINE_VARIANT,
    relaxed_variant: str = DEFAULT_RELAXED_VARIANT,
) -> str:
    image_list = ", ".join(images)
    return f"""# Confocal Gate Review Notes For Natalia

This pack compares the conservative `{baseline_variant}` gate with the slightly relaxed `{relaxed_variant}` gate for: {image_list}.

## What To Inspect

- In the added-vs-moderate overlays, the extra highlighted layer marks regions newly admitted by the relaxed gate.
- Are the newly added regions mostly valid visible striations?
- Is the missed middle region in `5138` better captured?
- Are shorter visible Z-disc structures in `3112` better captured?
- Does `7028` become too broad, especially in regions that looked questionable?
- Do the original moderate-gate spacing overlays remain biologically plausible as context?

## Important Caveat

Spacing has not yet been recomputed for newly added relaxed-gate patches. The spacing overlay included here is from the previous moderate-gate review only. If the relaxed gate is approved visually, the next step is a refreshed calibrated spacing audit using `{relaxed_variant}`.

## Scope

This is an exploratory visual review pack. It does not change the widefield pipeline, production algorithms, thresholds, or existing confocal outputs.
"""


def build_gate_review_pack_summary(
    review_summary: pd.DataFrame,
    copied: list[dict[str, str]],
    missing: list[dict[str, str]],
    images: list[str],
    baseline_variant: str,
    relaxed_variant: str,
    write_zip: bool,
) -> dict[str, Any]:
    return json_safe(
        {
            "mode": "confocal_gate_refinement_review_pack",
            "baseline_variant": baseline_variant,
            "relaxed_variant": relaxed_variant,
            "images_included": images,
            "image_count": len(images),
            "review_image_files_copied": len(copied),
            "missing_preview_files": missing,
            "missing_preview_count": len(missing),
            "summary_rows": int(len(review_summary)),
            "spacing_caveat_included": True,
            "write_zip_requested": bool(write_zip),
            "zip_path": None,
            "zip_excludes_internal_tables": False,
            "interpretation": [
                "Shareable review pack comparing moderate_reference to moderate_relaxed_combined.",
                "Added-vs-moderate overlays are for visual review before adopting any relaxed gate.",
                "Spacing has not been refreshed for newly added relaxed-gate patches.",
                "No algorithms, thresholds, widefield outputs, or existing confocal outputs were changed.",
            ],
        }
    )


def write_gate_review_pack_outputs(
    review_summary: pd.DataFrame,
    notes: str,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    review_summary.to_csv(paths["summary_csv"], index=False)
    paths["notes_md"].write_text(notes, encoding="utf-8")
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_gate_review_summary_text(summary), encoding="utf-8")


def write_gate_review_pack_zip(paths: dict[str, Path]) -> Path:
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


def render_gate_review_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal gate refinement review pack",
        f"baseline_variant: {summary['baseline_variant']}",
        f"relaxed_variant: {summary['relaxed_variant']}",
        f"images_included: {summary['images_included']}",
        f"review_image_files_copied: {summary['review_image_files_copied']}",
        f"missing_preview_count: {summary['missing_preview_count']}",
        f"spacing_caveat_included: {summary['spacing_caveat_included']}",
        f"zip_path: {summary.get('zip_path')}",
        "",
        "Missing previews:",
    ]
    lines.extend(f"- {item}" for item in summary["missing_preview_files"])
    lines.append("")
    lines.extend(summary["interpretation"])
    return "\n".join(lines) + "\n"


def first_variant_row(table: pd.DataFrame, image_id: str, variant_name: str) -> dict[str, Any]:
    if table.empty:
        return {}
    image_mask = table["confocal_image_id"].astype(str) == str(image_id) if "confocal_image_id" in table else False
    variant_mask = table["variant_name"].astype(str) == str(variant_name) if "variant_name" in table else False
    matches = table.loc[image_mask & variant_mask]
    return matches.iloc[0].to_dict() if not matches.empty else {}


def first_previous_review_row(table: pd.DataFrame, image_id: str) -> dict[str, Any]:
    if table.empty or "filename" not in table.columns:
        return {}
    normalized = table.copy(deep=True)
    normalized["_image_id"] = normalized["filename"].astype(str).map(filename_to_image_id)
    matches = normalized.loc[normalized["_image_id"] == str(image_id)]
    return matches.iloc[0].to_dict() if not matches.empty else {}


def variant_classification(variants: pd.DataFrame, variant_name: str) -> str:
    if variants.empty or "variant_name" not in variants or "classification" not in variants:
        return ""
    matches = variants.loc[variants["variant_name"].astype(str) == str(variant_name), "classification"]
    return str(matches.iloc[0]) if not matches.empty else ""


def focus_review_flag(image_id: str, relaxed_fraction: float | None) -> str:
    flags: list[str] = ["candidate_recovered_more_regions"]
    if image_id == "3112":
        flags.append("complex_short_zdisc_candidate")
    if image_id == "7028" or (relaxed_fraction is not None and np.isfinite(relaxed_fraction) and relaxed_fraction > 0.60):
        flags.append("broad_selection_risk")
    return ";".join(dict.fromkeys(flags))


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    header = pd.read_csv(path, nrows=0)
    dtype = {
        column: str
        for column in ["confocal_image_id", "filename", "variant_name", "patch_id"]
        if column in header.columns
    }
    return pd.read_csv(path, dtype=dtype)


def normalize_ids(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table.copy(deep=True)
    output = table.copy(deep=True)
    for column in ["confocal_image_id", "filename", "variant_name", "patch_id"]:
        if column in output.columns:
            output[column] = output[column].astype(str)
    return output


def filename_to_image_id(filename: str) -> str:
    name = Path(str(filename)).name
    return Path(name).stem


def relaxed_float(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else np.nan


def relaxed_int(value: Any) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return int(numeric) if pd.notna(numeric) else 0
