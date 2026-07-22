from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import output_dir
from .outputs import write_preview_png
from .zdisc_annotation import json_safe


VARIANT_COLUMNS = [
    "variant_id",
    "classification",
    "min_gradient_energy",
    "min_orientation_coherence",
    "min_intensity_std",
    "min_contrast",
    "min_signal_fraction",
    "max_saturation_fraction",
    "total_candidate_patches",
    "overall_candidate_fraction",
    "candidate_fraction_5138",
    "candidate_fraction_6052",
    "candidate_fraction_3112",
    "images_gt_90_candidate_fraction",
    "images_lt_05_candidate_fraction",
    "median_candidate_fraction_by_image",
    "median_candidate_coherence",
    "median_candidate_gradient_energy",
    "median_candidate_intensity_std",
    "missing_feature_columns",
]

PER_IMAGE_COLUMNS = [
    "variant_id",
    "classification",
    "confocal_image_id",
    "filename",
    "total_patches",
    "candidate_patch_count",
    "candidate_patch_fraction",
    "expected_positive_example",
    "noted_complex_example",
]


def default_sensitivity_paths(cfg: dict[str, Any], output_directory: str | Path | None = None) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_striation_sensitivity"
    return {
        "root": root,
        "variants": root / "confocal_striation_sensitivity_variants.csv",
        "per_image": root / "confocal_striation_sensitivity_per_image.csv",
        "summary_json": root / "confocal_striation_sensitivity_summary.json",
        "summary_txt": root / "confocal_striation_sensitivity_summary.txt",
        "previews": root / "previews",
    }


def run_confocal_striation_sensitivity(
    cfg: dict[str, Any],
    patch_table: str | Path | None = None,
    image_table: str | Path | None = None,
    output_directory: str | Path | None = None,
    write_previews: bool = False,
    max_preview_variants: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    root = output_dir(cfg)
    patch_path = Path(patch_table) if patch_table else root / "confocal_striation_mask" / "confocal_striation_mask_per_patch.csv"
    image_path = Path(image_table) if image_table else root / "confocal_striation_mask" / "confocal_striation_mask_per_image.csv"
    patches = pd.read_csv(patch_path, dtype={"confocal_image_id": str, "filename": str, "patch_id": str})
    images = pd.read_csv(image_path, dtype={"confocal_image_id": str, "filename": str}) if image_path.exists() else pd.DataFrame()

    variants = generate_threshold_variants(patches)
    per_image_tables: list[pd.DataFrame] = []
    variant_rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_image = evaluate_variant_by_image(patches, variant)
        classification = classify_variant(variant_image)
        variant_image.insert(0, "classification", classification)
        variant_image.insert(0, "variant_id", variant["variant_id"])
        per_image_tables.append(variant_image)
        variant_rows.append(summarize_variant(patches, variant_image, variant, classification))

    variant_table = pd.DataFrame(variant_rows, columns=VARIANT_COLUMNS)
    per_image = pd.concat(per_image_tables, ignore_index=True) if per_image_tables else pd.DataFrame(columns=PER_IMAGE_COLUMNS)
    summary = build_sensitivity_summary(variant_table, per_image, images)
    paths = default_sensitivity_paths(cfg, output_directory)
    preview_paths: list[str] = []
    if write_previews:
        source_preview_dir = patch_path.parent / "previews"
        preview_paths = write_sensitivity_previews(
            patches,
            variant_table,
            paths["previews"],
            source_preview_dir,
            max_preview_variants=max_preview_variants,
        )
        summary["previews_written"] = True
        summary["preview_paths"] = preview_paths
    write_sensitivity_outputs(variant_table, per_image, summary, paths)
    return variant_table, per_image, summary, paths


def generate_threshold_variants(patches: pd.DataFrame) -> list[dict[str, Any]]:
    def q(column: str, quantile: float, fallback: float) -> float:
        if column not in patches.columns:
            return fallback
        values = pd.to_numeric(patches[column], errors="coerce").dropna()
        return fallback if values.empty else float(values.quantile(quantile))

    return [
        {
            "variant_id": "lenient",
            "min_gradient_energy": q("gradient_energy", 0.20, 0.0001),
            "min_orientation_coherence": q("orientation_coherence", 0.20, 0.15),
            "min_intensity_std": q("intensity_std", 0.20, 0.02),
            "min_contrast": q("contrast", 0.20, 0.10),
            "min_signal_fraction": 0.02,
            "max_saturation_fraction": 0.10,
        },
        {
            "variant_id": "default_current",
            "min_gradient_energy": 0.0002,
            "min_orientation_coherence": 0.20,
            "min_intensity_std": 0.03,
            "min_contrast": 0.0,
            "min_signal_fraction": 0.05,
            "max_saturation_fraction": 0.10,
        },
        {
            "variant_id": "moderate",
            "min_gradient_energy": q("gradient_energy", 0.50, 0.0005),
            "min_orientation_coherence": q("orientation_coherence", 0.50, 0.35),
            "min_intensity_std": q("intensity_std", 0.50, 0.05),
            "min_contrast": q("contrast", 0.50, 0.20),
            "min_signal_fraction": q("signal_fraction", 0.25, 0.05),
            "max_saturation_fraction": 0.10,
        },
        {
            "variant_id": "strict",
            "min_gradient_energy": q("gradient_energy", 0.70, 0.001),
            "min_orientation_coherence": q("orientation_coherence", 0.70, 0.50),
            "min_intensity_std": q("intensity_std", 0.70, 0.08),
            "min_contrast": q("contrast", 0.70, 0.30),
            "min_signal_fraction": q("signal_fraction", 0.40, 0.10),
            "max_saturation_fraction": 0.05,
        },
        {
            "variant_id": "very_strict",
            "min_gradient_energy": q("gradient_energy", 0.85, 0.002),
            "min_orientation_coherence": q("orientation_coherence", 0.85, 0.65),
            "min_intensity_std": q("intensity_std", 0.85, 0.10),
            "min_contrast": q("contrast", 0.85, 0.40),
            "min_signal_fraction": q("signal_fraction", 0.60, 0.15),
            "max_saturation_fraction": 0.02,
        },
    ]


def evaluate_variant_by_image(patches: pd.DataFrame, variant: dict[str, Any]) -> pd.DataFrame:
    candidates = candidate_mask_for_variant(patches, variant)
    working = patches.copy()
    working["_variant_candidate"] = candidates
    rows: list[dict[str, Any]] = []
    for (image_id, filename), group in working.groupby(["confocal_image_id", "filename"], dropna=False):
        total = int(len(group))
        count = int(group["_variant_candidate"].sum())
        rows.append(
            {
                "confocal_image_id": str(image_id),
                "filename": str(filename),
                "total_patches": total,
                "candidate_patch_count": count,
                "candidate_patch_fraction": float(count / total) if total else 0.0,
                "expected_positive_example": bool(group["expected_positive_example"].fillna(False).astype(bool).any())
                if "expected_positive_example" in group
                else False,
                "noted_complex_example": bool(group["noted_complex_example"].fillna(False).astype(bool).any())
                if "noted_complex_example" in group
                else False,
            }
        )
    return pd.DataFrame(rows)


def candidate_mask_for_variant(patches: pd.DataFrame, variant: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=patches.index)
    checks = {
        "gradient_energy": ("min_gradient_energy", ">="),
        "orientation_coherence": ("min_orientation_coherence", ">="),
        "intensity_std": ("min_intensity_std", ">="),
        "contrast": ("min_contrast", ">="),
        "signal_fraction": ("min_signal_fraction", ">="),
        "saturation_fraction": ("max_saturation_fraction", "<="),
    }
    for column, (threshold_key, op) in checks.items():
        if column not in patches.columns:
            continue
        values = pd.to_numeric(patches[column], errors="coerce")
        threshold = float(variant[threshold_key])
        if op == ">=":
            mask &= values >= threshold
        else:
            mask &= values <= threshold
    return mask.fillna(False).astype(bool)


def missing_feature_columns(patches: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ["gradient_energy", "orientation_coherence", "intensity_std", "contrast", "signal_fraction", "saturation_fraction"]
        if column not in patches.columns
    ]


def summarize_variant(
    patches: pd.DataFrame,
    per_image: pd.DataFrame,
    variant: dict[str, Any],
    classification: str,
) -> dict[str, Any]:
    candidates = candidate_mask_for_variant(patches, variant)
    selected = patches.loc[candidates]
    total = int(len(patches))
    return {
        **variant,
        "classification": classification,
        "total_candidate_patches": int(candidates.sum()),
        "overall_candidate_fraction": float(candidates.sum() / total) if total else 0.0,
        "candidate_fraction_5138": image_fraction_matching(per_image, "5138"),
        "candidate_fraction_6052": image_fraction_matching(per_image, "6052"),
        "candidate_fraction_3112": image_fraction_matching(per_image, "3112"),
        "images_gt_90_candidate_fraction": int((per_image["candidate_patch_fraction"] > 0.90).sum()) if not per_image.empty else 0,
        "images_lt_05_candidate_fraction": int((per_image["candidate_patch_fraction"] < 0.05).sum()) if not per_image.empty else 0,
        "median_candidate_fraction_by_image": safe_median(per_image.get("candidate_patch_fraction", pd.Series(dtype=float))),
        "median_candidate_coherence": safe_median(selected.get("orientation_coherence", pd.Series(dtype=float))),
        "median_candidate_gradient_energy": safe_median(selected.get("gradient_energy", pd.Series(dtype=float))),
        "median_candidate_intensity_std": safe_median(selected.get("intensity_std", pd.Series(dtype=float))),
        "missing_feature_columns": ";".join(missing_feature_columns(patches)),
    }


def image_fraction_matching(per_image: pd.DataFrame, token: str) -> float | None:
    if per_image.empty:
        return None
    mask = (
        per_image["confocal_image_id"].fillna("").astype(str).str.contains(token, case=False, regex=False)
        | per_image["filename"].fillna("").astype(str).str.contains(token, case=False, regex=False)
    )
    if not mask.any():
        return None
    return float(per_image.loc[mask, "candidate_patch_fraction"].max())


def classify_variant(per_image: pd.DataFrame) -> str:
    if per_image.empty:
        return "uninformative_low_yield"
    fractions = pd.to_numeric(per_image["candidate_patch_fraction"], errors="coerce").fillna(0.0)
    median_fraction = float(fractions.median())
    images_gt90 = int((fractions > 0.90).sum())
    n_images = int(len(per_image))
    pos = per_image.loc[per_image["expected_positive_example"].fillna(False).astype(bool), "candidate_patch_fraction"]
    complex_fraction = per_image.loc[per_image["noted_complex_example"].fillna(False).astype(bool), "candidate_patch_fraction"]
    max_positive = float(pos.max()) if len(pos) else 0.0
    max_complex = float(complex_fraction.max()) if len(complex_fraction) else None

    too_broad = median_fraction > 0.80 or images_gt90 >= max(3, int(np.ceil(n_images / 2)))
    too_sparse = max_positive < 0.15
    plausible = (
        not too_broad
        and not too_sparse
        and 0.10 <= median_fraction <= 0.60
        and images_gt90 <= 1
        and (max_complex is None or max_complex < max_positive)
    )
    if plausible:
        return "plausible_for_review"
    if too_broad:
        return "too_broad"
    if too_sparse or median_fraction < 0.05:
        return "too_sparse"
    return "uninformative_low_yield"


def build_sensitivity_summary(
    variants: pd.DataFrame,
    per_image: pd.DataFrame,
    image_table: pd.DataFrame,
) -> dict[str, Any]:
    plausible = variants.loc[variants["classification"] == "plausible_for_review"] if not variants.empty else pd.DataFrame()
    default_row = variants.loc[variants["variant_id"] == "default_current"].head(1)
    return json_safe(
        {
            "mode": "confocal_striation_mask_sensitivity",
            "variant_count": int(len(variants)),
            "classification_counts": variants["classification"].value_counts().to_dict() if not variants.empty else {},
            "plausible_variants": plausible["variant_id"].astype(str).tolist() if not plausible.empty else [],
            "best_plausible_variants": top_variant_records(plausible),
            "default_assessment": default_row.to_dict("records")[0] if not default_row.empty else None,
            "why_default_was_too_broad": (
                "The default/current gate was too broad because median candidate fraction or many per-image candidate fractions were near whole-image."
                if not default_row.empty and str(default_row.iloc[0]["classification"]) == "too_broad"
                else "Default/current gate was not classified as too_broad in this sensitivity run."
            ),
            "expected_positive_and_complex_behavior": expected_complex_records(per_image),
            "source_image_table_rows": int(len(image_table)) if image_table is not None else 0,
            "previews_written": False,
            "preview_paths": [],
            "warning": "Sensitivity/QC only. This is not final tuning, a validated segmentation, or a biological endpoint.",
        }
    )


def top_variant_records(variants: pd.DataFrame, n: int = 3) -> list[dict[str, Any]]:
    if variants.empty:
        return []
    ranked = variants.copy()
    ranked["_rank"] = pd.to_numeric(ranked["candidate_fraction_6052"], errors="coerce").fillna(0) + pd.to_numeric(
        ranked["candidate_fraction_5138"], errors="coerce"
    ).fillna(0)
    ranked = ranked.sort_values(["_rank", "median_candidate_fraction_by_image"], ascending=[False, True]).drop(columns=["_rank"])
    return json_safe(ranked.head(n).to_dict("records"))


def expected_complex_records(per_image: pd.DataFrame) -> list[dict[str, Any]]:
    if per_image.empty:
        return []
    mask = per_image["expected_positive_example"].fillna(False).astype(bool) | per_image["noted_complex_example"].fillna(False).astype(bool)
    return json_safe(per_image.loc[mask].to_dict("records"))


def write_sensitivity_previews(
    patches: pd.DataFrame,
    variants: pd.DataFrame,
    output_preview_dir: Path,
    source_preview_dir: Path,
    max_preview_variants: int = 3,
) -> list[str]:
    output_preview_dir.mkdir(parents=True, exist_ok=True)
    selected_variants = select_preview_variants(variants, max_preview_variants=max_preview_variants)
    selected_images = select_preview_images(patches)
    written: list[str] = []
    for variant_id in selected_variants:
        variant = variants.loc[variants["variant_id"] == variant_id].iloc[0].to_dict()
        variant_patches = patches.copy()
        variant_patches["_variant_candidate"] = candidate_mask_for_variant(variant_patches, variant)
        for image_id in selected_images:
            source = source_preview_dir / f"{image_id}_normalized.png"
            if not source.exists():
                continue
            with Image.open(source) as image:
                base = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            image_patches = variant_patches.loc[variant_patches["confocal_image_id"].astype(str) == str(image_id)]
            out = output_preview_dir / f"{variant_id}_{image_id}_candidate_overlay.png"
            write_variant_overlay(base, image_patches, out)
            written.append(str(out))
    return written


def select_preview_variants(variants: pd.DataFrame, max_preview_variants: int = 3) -> list[str]:
    plausible = variants.loc[variants["classification"] == "plausible_for_review", "variant_id"].astype(str).tolist()
    if plausible:
        return plausible[: int(max_preview_variants)]
    fallback = [variant for variant in ["moderate", "strict", "very_strict"] if variant in set(variants["variant_id"].astype(str))]
    return fallback[: int(max_preview_variants)]


def select_preview_images(patches: pd.DataFrame) -> list[str]:
    images = set(patches["confocal_image_id"].astype(str))
    selected = [
        image_id
        for token in ["5138", "6052", "3112"]
        for image_id in sorted(images)
        if token.lower() in image_id.lower()
    ]
    broad = (
        patches.groupby("confocal_image_id")["candidate_striation_region"].mean().sort_values(ascending=False)
        if "candidate_striation_region" in patches
        else pd.Series(dtype=float)
    )
    if not broad.empty:
        selected.append(str(broad.index[0]))
    deduped: list[str] = []
    for image_id in selected:
        if image_id not in deduped:
            deduped.append(image_id)
    return deduped


def write_variant_overlay(image: np.ndarray, patches: pd.DataFrame, path: str | Path) -> Path:
    rgb = np.dstack([image, image, image]).astype(np.float32)
    for _, row in patches.iterrows():
        y0, y1, x0, x1 = int(row["y0"]), int(row["y1"]), int(row["x0"]), int(row["x1"])
        candidate = bool(row.get("_variant_candidate", False))
        color = np.array([1.0, 0.1, 0.1], dtype=np.float32) if candidate else np.array([0.1, 0.4, 1.0], dtype=np.float32)
        rgb[y0:y1, x0] = color
        rgb[y0:y1, x1 - 1] = color
        rgb[y0, x0:x1] = color
        rgb[y1 - 1, x0:x1] = color
    return write_preview_png(rgb, path)


def write_sensitivity_outputs(
    variants: pd.DataFrame,
    per_image: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    variants.to_csv(paths["variants"], index=False)
    per_image.to_csv(paths["per_image"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_sensitivity_summary_text(summary), encoding="utf-8")


def render_sensitivity_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal striation-mask sensitivity audit",
        f"variant_count: {summary['variant_count']}",
        f"classification_counts: {summary['classification_counts']}",
        f"plausible_variants: {summary['plausible_variants']}",
        "",
        summary["why_default_was_too_broad"],
        "",
        "Expected positive and complex image behavior:",
    ]
    for row in summary["expected_positive_and_complex_behavior"]:
        lines.append(
            f"- {row.get('variant_id')} {row.get('filename')}: fraction={row.get('candidate_patch_fraction')}, "
            f"expected_positive={row.get('expected_positive_example')}, complex={row.get('noted_complex_example')}"
        )
    lines.extend(["", summary["warning"]])
    return "\n".join(lines) + "\n"


def safe_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(np.median(numeric))
