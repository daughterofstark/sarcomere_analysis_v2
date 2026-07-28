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


REVIEW_INDEX_COLUMNS = [
    "confocal_image_id",
    "filename",
    "review_group",
    "endpoint_class",
    "spacing_reportable",
    "oop_reportable",
    "review_needed",
    "selected_candidate_fraction",
    "valid_selected_spacing_patches",
    "selected_spacing_valid_fraction",
    "selected_spacing_median_um",
    "selected_spacing_iqr_um",
    "selected_median_oop",
    "selected_vs_all_oop_difference",
    "reason",
    "copied_preview_count",
    "missing_preview_count",
]

PREVIEW_PATTERNS = {
    "selected_candidate_overlay": [
        "{image_id}_selected_candidate_overlay.png",
        "selective_analysis_{image_id}_selected_candidate_overlay.png",
    ],
    "spacing_candidate_overlay": [
        "{image_id}_spacing_candidate_overlay.png",
        "spacing_audit_{image_id}_confocal_spacing_candidate_overlay.png",
    ],
    "valid_spacing_overlay": [
        "{image_id}_valid_spacing_overlay.png",
        "spacing_audit_{image_id}_confocal_valid_spacing_overlay.png",
    ],
    "spacing_um_heatmap": [
        "{image_id}_spacing_um_heatmap.png",
        "spacing_audit_{image_id}_confocal_spacing_um_heatmap.png",
    ],
    "same_grid_oop_heatmap": [
        "{image_id}_same_grid_oop_heatmap.png",
        "same_grid_oop_{image_id}_same_grid_oop_heatmap.png",
    ],
    "same_grid_candidate_overlay": [
        "{image_id}_same_grid_candidate_overlay.png",
        "same_grid_oop_{image_id}_same_grid_candidate_overlay.png",
    ],
}


def default_endpoint_review_pack_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_endpoint_review_pack"
    return {
        "root": root,
        "review_images": root / "review_images",
        "index": root / "confocal_endpoint_review_index.csv",
        "notes": root / "confocal_endpoint_review_notes.md",
        "summary_json": root / "confocal_endpoint_review_pack_summary.json",
        "summary_txt": root / "confocal_endpoint_review_pack_summary.txt",
        "zip": root / "confocal_endpoint_review_pack.zip",
    }


def export_confocal_endpoint_review_pack(
    cfg: dict[str, Any],
    endpoint_dir: str | Path | None = None,
    pipeline_dir: str | Path | None = None,
    audit_dir: str | Path | None = None,
    output_directory: str | Path | None = None,
    write_zip: bool = False,
    n_oop_only_examples: int = 5,
    seed: int = 123,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    paths = default_endpoint_review_pack_paths(cfg, output_directory)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["review_images"].mkdir(parents=True, exist_ok=True)

    root = output_dir(cfg)
    endpoint_root = Path(endpoint_dir) if endpoint_dir else root / "confocal_endpoint_report"
    pipeline_root = Path(pipeline_dir) if pipeline_dir else root / "confocal_larger_pipeline"
    audit_root = Path(audit_dir) if audit_dir else root / "confocal_larger_audit"

    endpoint = read_required_csv(endpoint_root / "confocal_endpoint_per_image.csv")
    triage = read_csv_if_exists(audit_root / "confocal_larger_image_triage.csv")
    pipeline_image = read_csv_if_exists(pipeline_root / "confocal_pipeline_per_image.csv")
    pipeline_patch = read_csv_if_exists(pipeline_root / "confocal_pipeline_per_patch.csv")

    selected = select_endpoint_review_images(endpoint, n_oop_only_examples=n_oop_only_examples, seed=seed)
    copied, missing = copy_endpoint_review_previews(
        selected,
        paths["review_images"],
        source_dirs=[audit_root / "review_previews", pipeline_root / "previews"],
    )
    index = build_review_index(selected, copied, missing)
    notes = render_endpoint_review_notes()
    summary = build_endpoint_review_summary(
        index=index,
        copied=copied,
        missing=missing,
        write_zip=write_zip,
        source_tables={
            "endpoint_rows": len(endpoint),
            "triage_rows": len(triage),
            "pipeline_image_rows": len(pipeline_image),
            "pipeline_patch_rows": len(pipeline_patch),
        },
    )
    write_endpoint_review_outputs(index, notes, summary, paths)
    if write_zip:
        paths["zip"] = write_endpoint_review_zip(paths)
        summary["zip_path"] = str(paths["zip"])
        summary["zip_excludes_raw_internal_large_tables"] = True
        paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
        paths["summary_txt"].write_text(render_endpoint_review_summary_text(summary), encoding="utf-8")
    return index, summary, paths


def select_endpoint_review_images(
    endpoint: pd.DataFrame,
    n_oop_only_examples: int = 5,
    seed: int = 123,
) -> pd.DataFrame:
    working = normalize_ids(endpoint)
    spacing = working.loc[bool_series(working.get("spacing_reportable", pd.Series(False, index=working.index)))].copy()
    spacing["review_group"] = "spacing_reportable"

    low_candidate = working.loc[working["endpoint_class"].astype(str) == "low_candidate_review_needed"].copy()
    low_candidate["review_group"] = "low_candidate_review"

    already = set(spacing["confocal_image_id"].astype(str)) | set(low_candidate["confocal_image_id"].astype(str))
    example_pool = working.loc[
        working["endpoint_class"].astype(str).isin(["oop_only_spacing_low_yield", "spacing_eligible_low_confidence"])
        & ~working["confocal_image_id"].astype(str).isin(already)
    ].copy()
    oop_examples = choose_oop_only_examples(example_pool, n_examples=n_oop_only_examples, seed=seed)
    oop_examples["review_group"] = "oop_only_examples"

    selected = pd.concat([spacing, low_candidate, oop_examples], ignore_index=True)
    if selected.empty:
        return selected
    return selected.drop_duplicates("confocal_image_id", keep="first").reset_index(drop=True)


def choose_oop_only_examples(pool: pd.DataFrame, n_examples: int = 5, seed: int = 123) -> pd.DataFrame:
    if pool.empty or n_examples <= 0:
        return pool.head(0).copy()
    working = pool.copy()
    working["_spacing_yield"] = pd.to_numeric(working.get("selected_spacing_valid_fraction", np.nan), errors="coerce")
    working["_selected_oop"] = pd.to_numeric(working.get("selected_median_oop", np.nan), errors="coerce")
    chosen_indices: list[int] = []

    def add_first(frame: pd.DataFrame) -> None:
        for idx in frame.index:
            if idx not in chosen_indices:
                chosen_indices.append(idx)
                break

    add_first(working.sort_values(["_spacing_yield", "filename"], ascending=[True, True], na_position="last"))
    median_yield = working["_spacing_yield"].dropna().median()
    if np.isfinite(median_yield):
        median_frame = working.assign(_distance=(working["_spacing_yield"] - median_yield).abs())
        add_first(median_frame.sort_values(["_distance", "filename"], ascending=[True, True], na_position="last"))
    add_first(working.sort_values(["_selected_oop", "filename"], ascending=[False, True], na_position="last"))
    add_first(working.sort_values(["_selected_oop", "filename"], ascending=[True, True], na_position="last"))

    rng = np.random.default_rng(seed)
    remaining = [idx for idx in working.index if idx not in chosen_indices]
    if remaining and len(chosen_indices) < n_examples:
        shuffled = list(rng.permutation(remaining))
        chosen_indices.extend(int(idx) for idx in shuffled[: n_examples - len(chosen_indices)])

    if len(chosen_indices) < n_examples:
        for idx in working.sort_values(["filename"]).index:
            if idx not in chosen_indices:
                chosen_indices.append(idx)
            if len(chosen_indices) >= n_examples:
                break

    return working.loc[chosen_indices[:n_examples]].drop(columns=["_spacing_yield", "_selected_oop"], errors="ignore").copy()


def copy_endpoint_review_previews(
    selected: pd.DataFrame,
    review_dir: Path,
    source_dirs: list[Path],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    copied: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for _, row in selected.iterrows():
        image_id = str(row["confocal_image_id"])
        review_group = str(row["review_group"])
        for preview_type, patterns in PREVIEW_PATTERNS.items():
            source = first_existing_preview(source_dirs, image_id, patterns)
            destination = review_dir / f"{review_group}_{image_id}_{preview_type}.png"
            if source is not None:
                shutil.copy2(source, destination)
                copied.append(
                    {
                        "confocal_image_id": image_id,
                        "filename": str(row.get("filename", "")),
                        "review_group": review_group,
                        "preview_type": preview_type,
                        "path": str(destination),
                    }
                )
            else:
                missing.append(
                    {
                        "confocal_image_id": image_id,
                        "filename": str(row.get("filename", "")),
                        "review_group": review_group,
                        "preview_type": preview_type,
                    }
                )
    return copied, missing


def first_existing_preview(source_dirs: list[Path], image_id: str, patterns: list[str]) -> Path | None:
    for source_dir in source_dirs:
        for pattern in patterns:
            path = source_dir / pattern.format(image_id=image_id)
            if path.exists():
                return path
    return None


def build_review_index(
    selected: pd.DataFrame,
    copied: list[dict[str, str]],
    missing: list[dict[str, str]],
) -> pd.DataFrame:
    copied_counts = count_by_image(copied)
    missing_counts = count_by_image(missing)
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        image_id = str(row.get("confocal_image_id", ""))
        rows.append(
            {
                "confocal_image_id": image_id,
                "filename": str(row.get("filename", "")),
                "review_group": str(row.get("review_group", "")),
                "endpoint_class": str(row.get("endpoint_class", "")),
                "spacing_reportable": bool_value(row.get("spacing_reportable", False)),
                "oop_reportable": bool_value(row.get("oop_reportable", False)),
                "review_needed": bool_value(row.get("review_needed", False)),
                "selected_candidate_fraction": safe_float(row.get("selected_candidate_fraction")),
                "valid_selected_spacing_patches": safe_int(row.get("valid_selected_spacing_patches")),
                "selected_spacing_valid_fraction": safe_float(row.get("selected_spacing_valid_fraction")),
                "selected_spacing_median_um": safe_float(row.get("selected_spacing_median_um")),
                "selected_spacing_iqr_um": safe_float(row.get("selected_spacing_iqr_um")),
                "selected_median_oop": safe_float(row.get("selected_median_oop")),
                "selected_vs_all_oop_difference": safe_float(row.get("selected_vs_all_oop_difference")),
                "reason": str(row.get("reason", "")),
                "copied_preview_count": copied_counts.get(image_id, 0),
                "missing_preview_count": missing_counts.get(image_id, 0),
            }
        )
    return pd.DataFrame(rows, columns=REVIEW_INDEX_COLUMNS)


def build_endpoint_review_summary(
    index: pd.DataFrame,
    copied: list[dict[str, str]],
    missing: list[dict[str, str]],
    write_zip: bool,
    source_tables: dict[str, int],
) -> dict[str, Any]:
    group_counts = index["review_group"].value_counts().to_dict() if not index.empty else {}
    endpoint_counts = index["endpoint_class"].value_counts().to_dict() if not index.empty else {}
    return json_safe(
        {
            "mode": "confocal_endpoint_review_pack",
            "images_included": int(len(index)),
            "review_group_counts": group_counts,
            "endpoint_class_counts": endpoint_counts,
            "spacing_reportable_images": index.loc[index["review_group"] == "spacing_reportable", "filename"].tolist()
            if not index.empty
            else [],
            "low_candidate_review_images": index.loc[index["review_group"] == "low_candidate_review", "filename"].tolist()
            if not index.empty
            else [],
            "oop_only_example_images": index.loc[index["review_group"] == "oop_only_examples", "filename"].tolist()
            if not index.empty
            else [],
            "review_image_files_copied": len(copied),
            "missing_preview_files": missing,
            "missing_preview_count": len(missing),
            "write_zip_requested": bool(write_zip),
            "zip_path": None,
            "zip_excludes_raw_internal_large_tables": False,
            "source_tables": source_tables,
            "interpretation": [
                "Visual endpoint QC pack for the larger confocal cohort.",
                "Spacing-reportable images should be checked for genuine striation and spacing overlays.",
                "Low-candidate images should be checked for missed tissue, poor image quality, or overly sparse selected regions.",
                "OOP-only examples help inspect why spacing did not become a reportable endpoint despite available OOP/coherence.",
                "No thresholds, algorithms, relaxed gate adoption, widefield outputs, or biological claims are changed by this pack.",
            ],
        }
    )


def render_endpoint_review_notes() -> str:
    return """# Confocal Endpoint Review Notes

This compact pack is for visual endpoint QC of the larger confocal cohort.

## What To Check

- For spacing-reportable images, inspect whether selected candidate regions overlap convincing striations and whether spacing overlays appear to measure genuine adjacent Z-disc intervals.
- For low-candidate review images, check whether the gate missed tissue/striations or whether the image is genuinely low quality or sparse.
- For OOP-only examples, inspect why OOP/coherence is available but spacing remains low-yield.

## Caveats

- No thresholds are changed in this review pack.
- The relaxed gate is not adopted here.
- Existing confocal pipeline, cohort audit, endpoint report, widefield outputs, and production algorithms are not modified.
- These previews are diagnostic QC artifacts, not publication figures.
- No clinical, disease, or biological claims are made here.
"""


def write_endpoint_review_outputs(index: pd.DataFrame, notes: str, summary: dict[str, Any], paths: dict[str, Path]) -> None:
    index.to_csv(paths["index"], index=False)
    paths["notes"].write_text(notes, encoding="utf-8")
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_endpoint_review_summary_text(summary), encoding="utf-8")


def write_endpoint_review_zip(paths: dict[str, Path]) -> Path:
    zip_path = paths["zip"]
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for image_path in sorted(paths["review_images"].glob("*.png")):
            archive.write(image_path, arcname=f"review_images/{image_path.name}")
        archive.write(paths["index"], arcname=paths["index"].name)
        archive.write(paths["notes"], arcname=paths["notes"].name)
        archive.write(paths["summary_txt"], arcname=paths["summary_txt"].name)
    return zip_path


def render_endpoint_review_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal endpoint review pack",
        f"images_included: {summary['images_included']}",
        f"review_group_counts: {summary['review_group_counts']}",
        f"endpoint_class_counts: {summary['endpoint_class_counts']}",
        f"review_image_files_copied: {summary['review_image_files_copied']}",
        f"missing_preview_count: {summary['missing_preview_count']}",
        f"zip_path: {summary.get('zip_path')}",
        "",
        "Spacing-reportable images:",
    ]
    lines.extend(f"- {name}" for name in summary["spacing_reportable_images"] or ["none"])
    lines.extend(["", "Low-candidate review images:"])
    lines.extend(f"- {name}" for name in summary["low_candidate_review_images"] or ["none"])
    lines.extend(["", "OOP-only example images:"])
    lines.extend(f"- {name}" for name in summary["oop_only_example_images"] or ["none"])
    lines.extend(["", "Missing previews:"])
    lines.extend(f"- {item}" for item in summary["missing_preview_files"] or ["none"])
    lines.extend(["", "Interpretation:"])
    lines.extend(f"- {item}" for item in summary["interpretation"])
    return "\n".join(lines) + "\n"


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required confocal endpoint review pack input is missing: {path}")
    return read_csv(path)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return read_csv(path) if path.exists() else pd.DataFrame()


def read_csv(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    dtype = {
        column: str
        for column in ["confocal_image_id", "filename", "patch_id", "source_path"]
        if column in header.columns
    }
    return pd.read_csv(path, dtype=dtype)


def normalize_ids(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy(deep=True)
    for column in ["confocal_image_id", "filename", "patch_id"]:
        if column in output.columns:
            output[column] = output[column].astype(str)
    return output


def bool_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        if values.dtype == object:
            return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y"})
        return values.fillna(False).astype(bool)
    return pd.Series(values).fillna(False).astype(bool)


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


def safe_float(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else np.nan


def safe_int(value: Any) -> int:
    numeric = safe_float(value)
    return int(numeric) if np.isfinite(numeric) else 0


def count_by_image(items: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        image_id = str(item.get("confocal_image_id", ""))
        counts[image_id] = counts.get(image_id, 0) + 1
    return counts
