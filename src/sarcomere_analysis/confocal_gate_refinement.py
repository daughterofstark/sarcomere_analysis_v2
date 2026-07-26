from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_dir
from .confocal_intake import load_confocal_image_2d
from .confocal_striation_sensitivity import candidate_mask_for_variant, generate_threshold_variants
from .outputs import write_preview_png
from .preprocessing import preprocess_image
from .zdisc_annotation import json_safe


DEFAULT_FOCUS_IMAGES = ["5138", "6052-CLEAR_STRIPES", "3112", "7028"]

REFINEMENT_VARIANT_COLUMNS = [
    "variant_name",
    "total_candidate_patches",
    "overall_candidate_fraction",
    "added_vs_moderate_count",
    "removed_vs_moderate_count",
    "selected_median_oop",
    "selected_median_coherence",
    "selected_valid_spacing_count",
    "selected_valid_spacing_fraction",
    "selected_median_spacing_um",
    "classification",
]

REFINEMENT_PER_IMAGE_COLUMNS = [
    "confocal_image_id",
    "filename",
    "variant_name",
    "candidate_patch_count",
    "candidate_fraction",
    "added_vs_moderate_count",
    "removed_vs_moderate_count",
    "selected_median_oop",
    "selected_median_coherence",
    "selected_valid_spacing_count",
    "selected_spacing_valid_fraction",
    "selected_spacing_median_um",
    "expected_positive_example",
    "noted_complex_example",
    "review_flag",
]


def default_gate_refinement_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_gate_refinement"
    return {
        "root": root,
        "variants": root / "confocal_gate_refinement_variants.csv",
        "per_image": root / "confocal_gate_refinement_per_image.csv",
        "summary_json": root / "confocal_gate_refinement_summary.json",
        "summary_txt": root / "confocal_gate_refinement_summary.txt",
        "previews": root / "previews",
    }


def run_confocal_gate_refinement(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
    write_previews: bool = False,
    focus_images: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    paths = default_gate_refinement_paths(cfg, output_directory)
    focus = focus_images or DEFAULT_FOCUS_IMAGES
    patch_table = load_refinement_patch_table(cfg)
    variants = refinement_variants(patch_table)
    variant_rows: list[dict[str, Any]] = []
    per_image_tables: list[pd.DataFrame] = []
    masks: dict[str, pd.Series] = {}
    moderate = patch_table["moderate_candidate"].fillna(False).astype(bool)

    for variant in variants:
        if variant["variant_name"] == "moderate_reference":
            candidate = moderate.copy()
        else:
            candidate = candidate_mask_for_variant(patch_table, variant) | moderate
        masks[variant["variant_name"]] = candidate
        variant_rows.append(summarize_refinement_variant(patch_table, candidate, moderate, variant["variant_name"]))
        per_image_tables.append(summarize_refinement_by_image(patch_table, candidate, moderate, variant["variant_name"]))

    variant_table = pd.DataFrame(variant_rows, columns=REFINEMENT_VARIANT_COLUMNS)
    per_image = pd.concat(per_image_tables, ignore_index=True) if per_image_tables else pd.DataFrame(columns=REFINEMENT_PER_IMAGE_COLUMNS)
    summary = build_gate_refinement_summary(variant_table, per_image, focus)
    preview_paths: list[str] = []
    if write_previews:
        preview_paths = write_gate_refinement_previews(cfg, patch_table, masks, focus, paths["previews"])
        summary["previews_written"] = True
        summary["preview_paths"] = preview_paths
    write_gate_refinement_outputs(variant_table, per_image, summary, paths)
    return variant_table, per_image, summary, paths


def load_refinement_patch_table(cfg: dict[str, Any]) -> pd.DataFrame:
    root = output_dir(cfg)
    mask = read_csv(root / "confocal_striation_mask" / "confocal_striation_mask_per_patch.csv")
    oop = read_csv(root / "confocal_same_grid_oop" / "confocal_same_grid_oop_per_patch.csv")
    spacing = read_csv(root / "confocal_spacing_audit" / "confocal_spacing_per_patch.csv")
    metadata = read_csv(root / "confocal_metadata" / "confocal_metadata_calibration.csv")
    table = normalize_ids(mask)
    if "candidate_striation_region" in table.columns:
        table["default_candidate"] = bool_series(table["candidate_striation_region"])
    else:
        table["default_candidate"] = False

    oop_keep = [
        column
        for column in [
            "confocal_image_id",
            "patch_id",
            "candidate_striation_region",
            "patch_oop_128",
            "patch_mean_orientation_deg_128",
            "patch_orientation_coherence_mean_128",
        ]
        if column in oop.columns
    ]
    if {"confocal_image_id", "patch_id"}.issubset(oop_keep):
        oop_subset = normalize_ids(oop[oop_keep]).drop_duplicates(["confocal_image_id", "patch_id"])
        oop_subset = oop_subset.rename(columns={"candidate_striation_region": "moderate_candidate"})
        table = table.merge(oop_subset, on=["confocal_image_id", "patch_id"], how="left")
    if "moderate_candidate" not in table.columns:
        table["moderate_candidate"] = table["default_candidate"]
    table["moderate_candidate"] = bool_series(table["moderate_candidate"])

    spacing_keep = [
        column
        for column in ["confocal_image_id", "patch_id", "spacing_valid", "spacing_estimate_um", "spacing_confidence"]
        if column in spacing.columns
    ]
    if {"confocal_image_id", "patch_id"}.issubset(spacing_keep):
        spacing_subset = normalize_ids(spacing[spacing_keep]).drop_duplicates(["confocal_image_id", "patch_id"])
        table = table.merge(spacing_subset, on=["confocal_image_id", "patch_id"], how="left")
    for column in ["spacing_valid", "spacing_estimate_um", "spacing_confidence"]:
        if column not in table.columns:
            table[column] = np.nan
    table["spacing_valid"] = bool_series(table["spacing_valid"])

    metadata_keep = [
        column
        for column in ["confocal_image_id", "source_path", "pixel_size_x_um", "pixel_size_available", "spacing_um_enabled"]
        if column in metadata.columns
    ]
    if "confocal_image_id" in metadata_keep:
        metadata_subset = normalize_ids(metadata[metadata_keep]).drop_duplicates(["confocal_image_id"])
        table = table.merge(metadata_subset, on="confocal_image_id", how="left")
    return table


def refinement_variants(patches: pd.DataFrame) -> list[dict[str, Any]]:
    generated = {variant["variant_id"]: variant for variant in generate_threshold_variants(patches)}
    moderate = dict(generated["moderate"])
    lenient = dict(generated["lenient"])
    moderate["variant_name"] = "moderate_reference"
    moderate["variant_id"] = "moderate_reference"

    def relaxed(name: str, keys: list[str], fraction_toward_lenient: float) -> dict[str, Any]:
        variant = dict(generated["moderate"])
        variant["variant_name"] = name
        variant["variant_id"] = name
        for key in keys:
            variant[key] = move_toward_lenient(float(generated["moderate"][key]), float(lenient[key]), fraction_toward_lenient)
        return variant

    return [
        moderate,
        relaxed("moderate_relaxed_coherence", ["min_orientation_coherence"], 0.30),
        relaxed("moderate_relaxed_gradient", ["min_gradient_energy"], 0.30),
        relaxed("moderate_relaxed_contrast", ["min_contrast"], 0.30),
        relaxed(
            "moderate_relaxed_combined",
            ["min_orientation_coherence", "min_gradient_energy", "min_contrast", "min_intensity_std", "min_signal_fraction"],
            0.30,
        ),
    ]


def move_toward_lenient(moderate: float, lenient: float, fraction: float) -> float:
    return float(moderate - fraction * (moderate - lenient))


def summarize_refinement_variant(
    patches: pd.DataFrame,
    candidate: pd.Series,
    moderate: pd.Series,
    variant_name: str,
) -> dict[str, Any]:
    total = int(len(patches))
    selected = patches.loc[candidate]
    spacing_valid = selected.loc[bool_series(selected.get("spacing_valid", pd.Series(False, index=selected.index)))]
    candidate_count = int(candidate.sum())
    return {
        "variant_name": variant_name,
        "total_candidate_patches": candidate_count,
        "overall_candidate_fraction": float(candidate_count / total) if total else 0.0,
        "added_vs_moderate_count": int((candidate & ~moderate).sum()),
        "removed_vs_moderate_count": int((moderate & ~candidate).sum()),
        "selected_median_oop": safe_median(selected.get("patch_oop_128", pd.Series(dtype=float))),
        "selected_median_coherence": safe_median(coherence_series(selected)),
        "selected_valid_spacing_count": int(len(spacing_valid)),
        "selected_valid_spacing_fraction": float(len(spacing_valid) / candidate_count) if candidate_count else 0.0,
        "selected_median_spacing_um": safe_median(spacing_valid.get("spacing_estimate_um", pd.Series(dtype=float))),
        "classification": classify_refinement_variant(patches, candidate, moderate, variant_name),
    }


def summarize_refinement_by_image(
    patches: pd.DataFrame,
    candidate: pd.Series,
    moderate: pd.Series,
    variant_name: str,
) -> pd.DataFrame:
    working = patches.copy(deep=True)
    working["_candidate"] = candidate
    working["_moderate"] = moderate
    rows: list[dict[str, Any]] = []
    for (image_id, filename), group in working.groupby(["confocal_image_id", "filename"], dropna=False):
        selected = group.loc[group["_candidate"]]
        spacing_valid = selected.loc[bool_series(selected.get("spacing_valid", pd.Series(False, index=selected.index)))]
        total = int(len(group))
        candidate_count = int(group["_candidate"].sum())
        added = int((group["_candidate"] & ~group["_moderate"]).sum())
        removed = int((group["_moderate"] & ~group["_candidate"]).sum())
        rows.append(
            {
                "confocal_image_id": str(image_id),
                "filename": str(filename),
                "variant_name": variant_name,
                "candidate_patch_count": candidate_count,
                "candidate_fraction": float(candidate_count / total) if total else 0.0,
                "added_vs_moderate_count": added,
                "removed_vs_moderate_count": removed,
                "selected_median_oop": safe_median(selected.get("patch_oop_128", pd.Series(dtype=float))),
                "selected_median_coherence": safe_median(coherence_series(selected)),
                "selected_valid_spacing_count": int(len(spacing_valid)),
                "selected_spacing_valid_fraction": float(len(spacing_valid) / candidate_count) if candidate_count else 0.0,
                "selected_spacing_median_um": safe_median(spacing_valid.get("spacing_estimate_um", pd.Series(dtype=float))),
                "expected_positive_example": bool(group["expected_positive_example"].fillna(False).astype(bool).any())
                if "expected_positive_example" in group
                else False,
                "noted_complex_example": bool(group["noted_complex_example"].fillna(False).astype(bool).any())
                if "noted_complex_example" in group
                else False,
                "review_flag": image_review_flag(str(image_id), variant_name, candidate_count / total if total else 0.0, added),
            }
        )
    return pd.DataFrame(rows, columns=REFINEMENT_PER_IMAGE_COLUMNS)


def classify_refinement_variant(
    patches: pd.DataFrame,
    candidate: pd.Series,
    moderate: pd.Series,
    variant_name: str,
) -> str:
    if variant_name == "moderate_reference":
        return "conservative_reference"
    working = patches.copy()
    working["_candidate"] = candidate
    fractions = working.groupby("confocal_image_id")["_candidate"].mean()
    added = int((candidate & ~moderate).sum())
    if added == 0:
        return "review_needed"
    if float(fractions.median()) > 0.60 or int((fractions > 0.75).sum()) >= 2:
        return "too_broad"
    expected = working.loc[working.get("expected_positive_example", False).fillna(False).astype(bool)]
    expected_fraction = float(expected["_candidate"].mean()) if not expected.empty else 0.0
    if expected_fraction < 0.10:
        return "too_sparse"
    return "plausible_for_review"


def image_review_flag(image_id: str, variant_name: str, candidate_fraction: float, added_count: int) -> str:
    flags: list[str] = []
    if variant_name == "moderate_reference":
        flags.append("keep_conservative_reference")
    if added_count > 0:
        flags.append("candidate_recovered_more_regions")
    if candidate_fraction > 0.60 or image_id == "7028":
        flags.append("broad_selection_risk")
    if image_id == "3112" and added_count > 0:
        flags.append("complex_short_zdisc_candidate")
    if not flags:
        flags.append("review_needed")
    return ";".join(flags)


def build_gate_refinement_summary(
    variants: pd.DataFrame,
    per_image: pd.DataFrame,
    focus_images: list[str],
) -> dict[str, Any]:
    plausible = variants.loc[variants["classification"] == "plausible_for_review"].copy()
    best = {}
    recommendation = "keep_moderate"
    if not plausible.empty:
        plausible = plausible.sort_values(["added_vs_moderate_count", "selected_valid_spacing_count"], ascending=False)
        best = plausible.iloc[0].to_dict()
        recommendation = "review_relaxed_variant"
    return json_safe(
        {
            "mode": "confocal_gate_refinement_review_guided_audit",
            "current_reference_variant": "moderate",
            "variants_tested": variants["variant_name"].tolist(),
            "natalia_feedback_summary": [
                "5138 and 6052 selected the right striated regions with no obvious incorrect picks, but missed some clear striations.",
                "3112 picked difficult striated regions well, but missed shorter visible Z-disc structures.",
                "7028 was mostly correct but has broad/review-needed regions and missed some obvious bottom-left striations.",
                "Valid spacing overlays looked reasonable and represented genuine adjacent Z-disc intervals.",
            ],
            "classification_counts": variants["classification"].value_counts().to_dict(),
            "best_plausible_relaxed_variant": best,
            "recommendation": recommendation,
            "focus_image_summaries": focus_image_summaries(per_image, focus_images),
            "spacing_caveat": (
                "Spacing summaries use calibrated spacing values already available from the prior spacing audit. "
                "Newly added relaxed-gate patches require visual review and, if adopted, a refreshed spacing audit."
            ),
            "previews_written": False,
            "preview_paths": [],
            "interpretation": [
                "Review-guided confocal gate refinement audit only.",
                "Moderate remains the conservative baseline and is not overwritten.",
                "Relaxed variants use existing patch-level features only; no new segmentation algorithm is introduced.",
                "No thresholds are changed in default config or production outputs.",
                "No biological claims are made.",
            ],
        }
    )


def focus_image_summaries(per_image: pd.DataFrame, focus_images: list[str]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for image_id in focus_images:
        mask = per_image["confocal_image_id"].astype(str) == str(image_id)
        output[image_id] = per_image.loc[mask].to_dict("records")
    return output


def write_gate_refinement_previews(
    cfg: dict[str, Any],
    patches: pd.DataFrame,
    masks: dict[str, pd.Series],
    focus_images: list[str],
    preview_dir: Path,
) -> list[str]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    moderate = patches["moderate_candidate"].fillna(False).astype(bool)
    for image_id in focus_images:
        image_rows = patches.loc[patches["confocal_image_id"].astype(str) == str(image_id)]
        if image_rows.empty or "source_path" not in image_rows.columns:
            continue
        try:
            raw, _ = load_confocal_image_2d(str(image_rows["source_path"].iloc[0]))
            image = preprocess_image(raw, cfg).image
        except Exception:
            continue
        for variant_name, candidate_mask in masks.items():
            selected = candidate_mask.loc[image_rows.index]
            added = selected & ~moderate.loc[image_rows.index]
            spacing_valid = bool_series(image_rows.get("spacing_valid", pd.Series(False, index=image_rows.index)))
            out = preview_dir / f"{image_id}_{variant_name}_gate_refinement_overlay.png"
            paths.append(str(write_refinement_overlay(image, image_rows, selected, added, spacing_valid, out)))
    return paths


def write_refinement_overlay(
    image: np.ndarray,
    patches: pd.DataFrame,
    selected: pd.Series,
    added: pd.Series,
    spacing_valid: pd.Series,
    path: Path,
) -> Path:
    rgb = np.dstack([image, image, image]).astype(np.float32)
    blend_patch_layer(rgb, patches.loc[selected], np.array([1.0, 0.15, 0.05], dtype=np.float32), 0.25)
    blend_patch_layer(rgb, patches.loc[added], np.array([0.0, 0.75, 1.0], dtype=np.float32), 0.45)
    blend_patch_layer(rgb, patches.loc[spacing_valid], np.array([0.1, 1.0, 0.25], dtype=np.float32), 0.35)
    return write_preview_png(rgb, path)


def blend_patch_layer(rgb: np.ndarray, patches: pd.DataFrame, color: np.ndarray, alpha: float) -> None:
    for _, row in patches.iterrows():
        y0, y1, x0, x1 = int(row["y0"]), int(row["y1"]), int(row["x0"]), int(row["x1"])
        rgb[y0:y1, x0:x1] = (1.0 - alpha) * rgb[y0:y1, x0:x1] + alpha * color


def write_gate_refinement_outputs(
    variants: pd.DataFrame,
    per_image: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    variants.to_csv(paths["variants"], index=False)
    per_image.to_csv(paths["per_image"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_gate_refinement_summary_text(summary), encoding="utf-8")


def render_gate_refinement_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal gate refinement audit",
        f"current_reference_variant: {summary['current_reference_variant']}",
        f"variants_tested: {summary['variants_tested']}",
        f"classification_counts: {summary['classification_counts']}",
        f"recommendation: {summary['recommendation']}",
        f"best_plausible_relaxed_variant: {summary['best_plausible_relaxed_variant']}",
        "",
        "Natalia feedback summary:",
    ]
    lines.extend(f"- {item}" for item in summary["natalia_feedback_summary"])
    lines.extend(["", "Focus image summaries:"])
    for image_id, records in summary["focus_image_summaries"].items():
        lines.append(f"- {image_id}:")
        for record in records:
            lines.append(
                f"  - {record.get('variant_name')}: fraction={record.get('candidate_fraction')}, "
                f"added={record.get('added_vs_moderate_count')}, spacing_valid={record.get('selected_valid_spacing_count')}, "
                f"median_spacing={record.get('selected_spacing_median_um')}, flag={record.get('review_flag')}"
            )
    lines.extend(["", summary["spacing_caveat"], ""])
    lines.extend(summary["interpretation"])
    return "\n".join(lines) + "\n"


def read_csv(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    dtype = {column: str for column in ["confocal_image_id", "filename", "patch_id", "source_path"] if column in header.columns}
    return pd.read_csv(path, dtype=dtype)


def normalize_ids(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy(deep=True)
    for column in ["confocal_image_id", "filename", "patch_id"]:
        if column in output.columns:
            output[column] = output[column].astype(str)
    return output


def coherence_series(table: pd.DataFrame) -> pd.Series:
    if "patch_orientation_coherence_mean_128" in table.columns:
        return pd.to_numeric(table["patch_orientation_coherence_mean_128"], errors="coerce")
    return pd.to_numeric(table.get("orientation_coherence", pd.Series(dtype=float)), errors="coerce")


def bool_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        if values.dtype == object:
            return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y"})
        return values.fillna(False).astype(bool)
    return pd.Series(values).fillna(False).astype(bool)


def safe_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(np.median(numeric.to_numpy(dtype=float)))
