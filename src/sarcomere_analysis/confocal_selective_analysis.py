from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import output_dir
from .confocal_striation_sensitivity import candidate_mask_for_variant
from .outputs import write_preview_png
from .zdisc_annotation import json_safe


SELECTIVE_PATCH_COLUMNS = [
    "confocal_image_id",
    "filename",
    "patch_id",
    "y0",
    "x0",
    "y1",
    "x1",
    "center_y",
    "center_x",
    "selected_variant",
    "candidate_striation_region",
    "expected_positive_example",
    "noted_complex_example",
    "patch_oop",
    "baseline_patch_oop",
    "baseline_patch_mean_orientation_deg",
    "baseline_patch_orientation_weight_sum",
    "baseline_patch_orientation_valid_pixels",
    "baseline_join_matched",
    "baseline_coordinate_match",
    "orientation_coherence",
    "gradient_energy",
    "intensity_std",
    "contrast",
    "tissue_fraction",
    "signal_fraction",
    "saturation_fraction",
    "candidate_reason",
    "rejection_reason",
]

SELECTIVE_IMAGE_COLUMNS = [
    "confocal_image_id",
    "filename",
    "total_patches",
    "candidate_patch_count",
    "candidate_patch_fraction",
    "selected_region_median_oop",
    "selected_region_iqr_oop",
    "selected_region_median_coherence",
    "selected_region_median_gradient_energy",
    "selected_region_median_intensity_std",
    "all_region_median_oop",
    "all_region_median_coherence",
    "all_region_median_gradient_energy",
    "selected_vs_all_oop_difference",
    "expected_positive_example",
    "noted_complex_example",
    "interpretation_flag",
]


def default_selective_paths(cfg: dict[str, Any], output_directory: str | Path | None = None) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_selective_analysis"
    return {
        "root": root,
        "per_patch": root / "confocal_selective_per_patch.csv",
        "per_image": root / "confocal_selective_per_image.csv",
        "summary_json": root / "confocal_selective_summary.json",
        "summary_txt": root / "confocal_selective_summary.txt",
        "previews": root / "previews",
    }


def run_confocal_selective_analysis(
    cfg: dict[str, Any],
    selected_variant: str = "moderate",
    patch_table: str | Path | None = None,
    baseline_patch_table: str | Path | None = None,
    sensitivity_variants: str | Path | None = None,
    sensitivity_per_image: str | Path | None = None,
    output_directory: str | Path | None = None,
    min_candidate_patches: int = 10,
    write_previews: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    root = output_dir(cfg)
    patch_path = Path(patch_table) if patch_table else root / "confocal_striation_mask" / "confocal_striation_mask_per_patch.csv"
    baseline_patch_path = (
        Path(baseline_patch_table)
        if baseline_patch_table
        else root / "confocal_baseline" / "confocal_baseline_per_patch.csv"
    )
    variants_path = (
        Path(sensitivity_variants)
        if sensitivity_variants
        else root / "confocal_striation_sensitivity" / "confocal_striation_sensitivity_variants.csv"
    )
    sensitivity_image_path = (
        Path(sensitivity_per_image)
        if sensitivity_per_image
        else root / "confocal_striation_sensitivity" / "confocal_striation_sensitivity_per_image.csv"
    )

    patches = pd.read_csv(patch_path, dtype={"confocal_image_id": str, "filename": str, "patch_id": str})
    baseline_patches = (
        pd.read_csv(baseline_patch_path, dtype={"confocal_image_id": str, "filename": str, "patch_id": str})
        if baseline_patch_path.exists()
        else pd.DataFrame()
    )
    variants = pd.read_csv(variants_path)
    sensitivity_images = (
        pd.read_csv(sensitivity_image_path, dtype={"confocal_image_id": str, "filename": str})
        if sensitivity_image_path.exists()
        else pd.DataFrame()
    )
    variant = select_variant_row(variants, selected_variant)
    per_patch = build_selective_patch_table(patches, variant, selected_variant)
    per_patch, baseline_join_audit = join_baseline_patch_features(per_patch, baseline_patches)
    per_image = build_selective_image_table(per_patch, selected_variant, sensitivity_images, min_candidate_patches)
    summary = build_selective_summary(per_patch, per_image, selected_variant, variant, min_candidate_patches, baseline_join_audit)
    paths = default_selective_paths(cfg, output_directory)
    if write_previews:
        source_preview_dir = patch_path.parent / "previews"
        preview_paths = write_selective_previews(per_patch, paths["previews"], source_preview_dir)
        summary["previews_written"] = True
        summary["preview_paths"] = preview_paths
    write_selective_outputs(per_patch, per_image, summary, paths)
    return per_patch, per_image, summary, paths


def select_variant_row(variants: pd.DataFrame, selected_variant: str) -> dict[str, Any]:
    if "variant_id" not in variants.columns:
        raise ValueError("Sensitivity variants table is missing variant_id")
    match = variants.loc[variants["variant_id"].astype(str) == str(selected_variant)]
    if match.empty:
        raise ValueError(f"Selected variant not found in sensitivity variants table: {selected_variant}")
    return match.iloc[0].to_dict()


def join_baseline_patch_features(per_patch: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = per_patch.copy(deep=True)
    oop_columns = [
        column
        for column in [
            "patch_oop",
            "patch_mean_orientation_rad",
            "patch_mean_orientation_deg",
            "patch_orientation_weight_sum",
            "patch_orientation_valid_pixels",
        ]
        if column in baseline.columns
    ]
    audit: dict[str, Any] = {
        "baseline_patch_rows": int(len(baseline)),
        "selective_patch_rows": int(len(working)),
        "matched_rows": 0,
        "unmatched_rows": int(len(working)),
        "coordinate_mismatch_count": None,
        "oop_columns_found": oop_columns,
        "oop_join_usable": False,
        "oop_summary_reason": "baseline_patch_table_missing_or_empty" if baseline.empty else "no_oop_columns_found",
    }
    for column in [
        "baseline_patch_oop",
        "baseline_patch_mean_orientation_deg",
        "baseline_patch_orientation_weight_sum",
        "baseline_patch_orientation_valid_pixels",
        "baseline_join_matched",
        "baseline_coordinate_match",
    ]:
        if column not in working.columns:
            working[column] = np.nan
    working["baseline_join_matched"] = False
    working["baseline_coordinate_match"] = False
    if baseline.empty or not oop_columns:
        return working, audit

    keep_columns = ["confocal_image_id", "patch_id", *oop_columns]
    coordinate_columns = [column for column in ["y0", "x0", "y1", "x1"] if column in baseline.columns and column in working.columns]
    keep_columns.extend(column for column in coordinate_columns if column not in keep_columns)
    baseline_subset = baseline[keep_columns].drop_duplicates(["confocal_image_id", "patch_id"]).copy()
    rename = {
        "patch_oop": "baseline_patch_oop_raw",
        "patch_mean_orientation_deg": "baseline_patch_mean_orientation_deg_raw",
        "patch_orientation_weight_sum": "baseline_patch_orientation_weight_sum_raw",
        "patch_orientation_valid_pixels": "baseline_patch_orientation_valid_pixels_raw",
    }
    rename.update({column: f"baseline_{column}" for column in coordinate_columns})
    baseline_subset = baseline_subset.rename(columns=rename)
    joined = working.merge(baseline_subset, on=["confocal_image_id", "patch_id"], how="left")
    matched = joined[[column for column in joined.columns if column.endswith("_raw")]].notna().any(axis=1)
    joined["baseline_join_matched"] = matched
    audit["matched_rows"] = int(matched.sum())
    audit["unmatched_rows"] = int((~matched).sum())

    if coordinate_columns:
        coord_match = pd.Series(True, index=joined.index)
        for column in coordinate_columns:
            coord_match &= pd.to_numeric(joined[column], errors="coerce") == pd.to_numeric(joined[f"baseline_{column}"], errors="coerce")
        coord_match &= matched
        joined["baseline_coordinate_match"] = coord_match
        audit["coordinate_mismatch_count"] = int((matched & ~coord_match).sum())
    else:
        joined["baseline_coordinate_match"] = matched
        audit["coordinate_mismatch_count"] = None

    usable = bool(matched.any() and joined["baseline_coordinate_match"].all() and "patch_oop" in oop_columns)
    audit["oop_join_usable"] = usable
    if usable:
        audit["oop_summary_reason"] = "baseline_patch_oop_joined_on_matching_grid"
    elif matched.any() and audit["coordinate_mismatch_count"]:
        audit["oop_summary_reason"] = "baseline_patch_grid_coordinate_mismatch"
    elif matched.any() and "patch_oop" not in oop_columns:
        audit["oop_summary_reason"] = "baseline_oop_column_missing"
    else:
        audit["oop_summary_reason"] = "no_matching_baseline_patch_rows"

    if usable:
        joined["patch_oop"] = joined.get("baseline_patch_oop_raw", np.nan)
    joined["baseline_patch_oop"] = joined.get("baseline_patch_oop_raw", np.nan)
    joined["baseline_patch_mean_orientation_deg"] = joined.get("baseline_patch_mean_orientation_deg_raw", np.nan)
    joined["baseline_patch_orientation_weight_sum"] = joined.get("baseline_patch_orientation_weight_sum_raw", np.nan)
    joined["baseline_patch_orientation_valid_pixels"] = joined.get("baseline_patch_orientation_valid_pixels_raw", np.nan)
    drop_columns = [column for column in joined.columns if column.endswith("_raw") or column.startswith("baseline_y") or column.startswith("baseline_x")]
    return joined.drop(columns=drop_columns), audit


def build_selective_patch_table(patches: pd.DataFrame, variant: dict[str, Any], selected_variant: str) -> pd.DataFrame:
    working = patches.copy(deep=True)
    candidate = candidate_mask_for_variant(working, variant)
    working["selected_variant"] = str(selected_variant)
    working["candidate_striation_region"] = candidate
    working["candidate_reason"] = np.where(candidate, "selected_variant_candidate", "")
    working["rejection_reason"] = [
        "ok" if is_candidate else variant_rejection_reason(row, variant)
        for is_candidate, (_, row) in zip(candidate.tolist(), working.iterrows())
    ]
    for column in SELECTIVE_PATCH_COLUMNS:
        if column not in working.columns:
            working[column] = np.nan
    return working[SELECTIVE_PATCH_COLUMNS].copy()


def variant_rejection_reason(row: pd.Series, variant: dict[str, Any]) -> str:
    checks = [
        ("gradient_energy", "min_gradient_energy", "low_gradient_energy", ">="),
        ("orientation_coherence", "min_orientation_coherence", "low_orientation_coherence", ">="),
        ("intensity_std", "min_intensity_std", "low_intensity_std", ">="),
        ("contrast", "min_contrast", "low_contrast", ">="),
        ("signal_fraction", "min_signal_fraction", "low_signal_fraction", ">="),
        ("saturation_fraction", "max_saturation_fraction", "high_saturation_fraction", "<="),
    ]
    reasons: list[str] = []
    for column, threshold_key, reason, op in checks:
        if column not in row.index or threshold_key not in variant:
            continue
        try:
            value = float(row[column])
            threshold = float(variant[threshold_key])
        except (TypeError, ValueError):
            reasons.append(f"missing_{column}")
            continue
        if op == ">=" and value < threshold:
            reasons.append(reason)
        if op == "<=" and value > threshold:
            reasons.append(reason)
    return ";".join(reasons) if reasons else "not_selected"


def build_selective_image_table(
    per_patch: pd.DataFrame,
    selected_variant: str,
    sensitivity_per_image: pd.DataFrame,
    min_candidate_patches: int,
) -> pd.DataFrame:
    expected_positive_reference = expected_positive_candidate_fraction(sensitivity_per_image, selected_variant)
    rows: list[dict[str, Any]] = []
    for (image_id, filename), group in per_patch.groupby(["confocal_image_id", "filename"], dropna=False):
        candidates = group.loc[group["candidate_striation_region"].fillna(False).astype(bool)]
        total = int(len(group))
        count = int(len(candidates))
        fraction = float(count / total) if total else 0.0
        selected_median_oop = safe_median(candidates.get("patch_oop", pd.Series(dtype=float)))
        all_median_oop = safe_median(group.get("patch_oop", pd.Series(dtype=float)))
        rows.append(
            {
                "confocal_image_id": str(image_id),
                "filename": str(filename),
                "total_patches": total,
                "candidate_patch_count": count,
                "candidate_patch_fraction": fraction,
                "selected_region_median_oop": selected_median_oop,
                "selected_region_iqr_oop": safe_iqr(candidates.get("patch_oop", pd.Series(dtype=float))),
                "selected_region_median_coherence": safe_median(candidates.get("orientation_coherence", pd.Series(dtype=float))),
                "selected_region_median_gradient_energy": safe_median(candidates.get("gradient_energy", pd.Series(dtype=float))),
                "selected_region_median_intensity_std": safe_median(candidates.get("intensity_std", pd.Series(dtype=float))),
                "all_region_median_oop": all_median_oop,
                "all_region_median_coherence": safe_median(group.get("orientation_coherence", pd.Series(dtype=float))),
                "all_region_median_gradient_energy": safe_median(group.get("gradient_energy", pd.Series(dtype=float))),
                "selected_vs_all_oop_difference": difference_or_nan(selected_median_oop, all_median_oop),
                "expected_positive_example": bool(group["expected_positive_example"].fillna(False).astype(bool).any())
                if "expected_positive_example" in group
                else False,
                "noted_complex_example": bool(group["noted_complex_example"].fillna(False).astype(bool).any())
                if "noted_complex_example" in group
                else False,
                "interpretation_flag": interpretation_flags(
                    count,
                    fraction,
                    bool(group["expected_positive_example"].fillna(False).astype(bool).any())
                    if "expected_positive_example" in group
                    else False,
                    bool(group["noted_complex_example"].fillna(False).astype(bool).any())
                    if "noted_complex_example" in group
                    else False,
                    expected_positive_reference,
                    min_candidate_patches,
                ),
            }
        )
    return pd.DataFrame(rows, columns=SELECTIVE_IMAGE_COLUMNS)


def expected_positive_candidate_fraction(sensitivity_per_image: pd.DataFrame, selected_variant: str) -> float | None:
    if sensitivity_per_image.empty:
        return None
    table = sensitivity_per_image.loc[sensitivity_per_image["variant_id"].astype(str) == str(selected_variant)].copy()
    if table.empty or "expected_positive_example" not in table.columns:
        return None
    positives = table.loc[table["expected_positive_example"].fillna(False).astype(bool), "candidate_patch_fraction"]
    return safe_median(positives)


def interpretation_flags(
    candidate_count: int,
    candidate_fraction: float,
    expected_positive: bool,
    noted_complex: bool,
    expected_positive_reference: float | None,
    min_candidate_patches: int,
) -> str:
    flags: list[str] = []
    if candidate_count < int(min_candidate_patches):
        flags.append("too_few_candidates")
    if candidate_fraction > 0.80:
        flags.append("broad_candidate_fraction")
    if expected_positive and candidate_count >= int(min_candidate_patches):
        flags.append("expected_positive_has_candidates")
    if noted_complex and expected_positive_reference is not None and candidate_fraction < expected_positive_reference:
        flags.append("complex_lower_candidate_fraction")
    if not flags:
        flags.append("review_needed")
    return ";".join(flags)


def build_selective_summary(
    per_patch: pd.DataFrame,
    per_image: pd.DataFrame,
    selected_variant: str,
    variant: dict[str, Any],
    min_candidate_patches: int,
    baseline_join_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_join_audit = baseline_join_audit or {}
    return json_safe(
        {
            "mode": "confocal_selective_region_analysis",
            "selected_variant": str(selected_variant),
            "variant_thresholds": {
                key: variant.get(key)
                for key in [
                    "min_gradient_energy",
                    "min_orientation_coherence",
                    "min_intensity_std",
                    "min_contrast",
                    "min_signal_fraction",
                    "max_saturation_fraction",
                ]
                if key in variant
            },
            "total_patches": int(len(per_patch)),
            "candidate_patch_count": int(per_patch["candidate_striation_region"].fillna(False).astype(bool).sum())
            if not per_patch.empty
            else 0,
            "candidate_fraction_by_image": per_image[
                ["confocal_image_id", "filename", "candidate_patch_count", "total_patches", "candidate_patch_fraction"]
            ].to_dict("records")
            if not per_image.empty
            else [],
            "selected_region_summaries": special_image_records(per_image),
            "selected_vs_all_comparison": {
                "median_selected_region_coherence": safe_median(per_image.get("selected_region_median_coherence", pd.Series(dtype=float))),
                "median_all_region_coherence": safe_median(per_image.get("all_region_median_coherence", pd.Series(dtype=float))),
                "median_selected_region_gradient_energy": safe_median(per_image.get("selected_region_median_gradient_energy", pd.Series(dtype=float))),
                "median_all_region_gradient_energy": safe_median(per_image.get("all_region_median_gradient_energy", pd.Series(dtype=float))),
                "median_selected_region_oop": safe_median(per_image.get("selected_region_median_oop", pd.Series(dtype=float))),
                "median_all_region_oop": safe_median(per_image.get("all_region_median_oop", pd.Series(dtype=float))),
                "median_selected_vs_all_oop_difference": safe_median(per_image.get("selected_vs_all_oop_difference", pd.Series(dtype=float))),
                "oop_available": bool(pd.to_numeric(per_image.get("selected_region_median_oop", pd.Series(dtype=float)), errors="coerce").notna().any()),
            },
            "baseline_patch_join_audit": baseline_join_audit,
            "min_candidate_patches": int(min_candidate_patches),
            "spacing_status": "not_computed_in_microns_confocal_pixel_size_unknown",
            "previews_written": False,
            "preview_paths": [],
            "interpretation": [
                "Exploratory selective-region analysis only.",
                "Uses the selected sensitivity variant; no threshold search is recomputed.",
                "No spacing in microns is computed without confocal pixel calibration.",
                "Baseline patch OOP is summarized only when the baseline and selective patch grids align.",
                "Not validated by manual confocal annotation yet.",
                "6052 and 5138 retain meaningful candidate fractions under the moderate gate; 3112 remains lower; 7028 has a broad candidate fraction and needs review.",
                "No biological claims are made.",
            ],
        }
    )


def special_image_records(per_image: pd.DataFrame) -> list[dict[str, Any]]:
    if per_image.empty:
        return []
    mask = per_image["expected_positive_example"].fillna(False).astype(bool) | per_image["noted_complex_example"].fillna(False).astype(bool)
    return json_safe(per_image.loc[mask].to_dict("records"))


def write_selective_previews(per_patch: pd.DataFrame, preview_dir: Path, source_preview_dir: Path) -> list[str]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for image_id, group in per_patch.groupby("confocal_image_id", dropna=False):
        source = source_preview_dir / f"{image_id}_normalized.png"
        if not source.exists():
            continue
        with Image.open(source) as image:
            base = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        out = preview_dir / f"{image_id}_selected_candidate_overlay.png"
        write_selected_overlay(base, group, out)
        paths.append(str(out))
    return paths


def write_selected_overlay(image: np.ndarray, patches: pd.DataFrame, path: str | Path) -> Path:
    rgb = np.dstack([image, image, image]).astype(np.float32)
    alpha = 0.35
    selected_color = np.array([1.0, 0.1, 0.1], dtype=np.float32)
    for _, row in patches.iterrows():
        y0, y1, x0, x1 = int(row["y0"]), int(row["y1"]), int(row["x0"]), int(row["x1"])
        if bool(row.get("candidate_striation_region", False)):
            rgb[y0:y1, x0:x1] = (1.0 - alpha) * rgb[y0:y1, x0:x1] + alpha * selected_color
            rgb[y0:y1, x0] = selected_color
            rgb[y0:y1, x1 - 1] = selected_color
            rgb[y0, x0:x1] = selected_color
            rgb[y1 - 1, x0:x1] = selected_color
    return write_preview_png(rgb, path)


def write_selective_outputs(
    per_patch: pd.DataFrame,
    per_image: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    per_patch.to_csv(paths["per_patch"], index=False)
    per_image.to_csv(paths["per_image"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_selective_summary_text(summary), encoding="utf-8")


def render_selective_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal selective-region analysis",
        f"selected_variant: {summary['selected_variant']}",
        f"total_patches: {summary['total_patches']}",
        f"candidate_patch_count: {summary['candidate_patch_count']}",
        f"spacing_status: {summary['spacing_status']}",
        "",
        "Special image summaries:",
    ]
    for row in summary["selected_region_summaries"]:
        lines.append(
            f"- {row.get('filename')}: candidates={row.get('candidate_patch_count')}/{row.get('total_patches')} "
            f"fraction={row.get('candidate_patch_fraction')}, coherence={row.get('selected_region_median_coherence')}, "
            f"gradient={row.get('selected_region_median_gradient_energy')}, flags={row.get('interpretation_flag')}"
        )
    lines.append("")
    lines.append(f"Selected-vs-all comparison: {summary['selected_vs_all_comparison']}")
    lines.append("")
    lines.extend(summary["interpretation"])
    return "\n".join(lines) + "\n"


def safe_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(np.median(numeric))


def safe_iqr(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    q75, q25 = np.percentile(numeric, [75, 25])
    return float(q75 - q25)


def difference_or_nan(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return float("nan")
    return float(left - right)
