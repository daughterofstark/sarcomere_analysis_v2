from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import output_dir
from .zdisc_annotation import json_safe


def default_validation_status_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> dict[str, Path]:
    results_dir = Path(output_directory) if output_directory else output_dir(cfg) / "validation"
    docs_dir = Path(docs_directory) if docs_directory else Path("docs")
    return {
        "summary_json": results_dir / "validation_status_summary.json",
        "summary_txt": results_dir / "validation_status_summary.txt",
        "markdown": docs_dir / "VALIDATION_STATUS.md",
    }


def write_validation_status(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    status = build_validation_status(cfg)
    paths = default_validation_status_paths(cfg, output_directory=output_directory, docs_directory=docs_directory)
    write_validation_status_outputs(status, paths)
    return status, paths


def build_validation_status(cfg: dict[str, Any]) -> dict[str, Any]:
    root = output_dir(cfg)
    validation_dir = root / "validation"
    tables_dir = root / "tables"
    summaries = {
        "synthetic_oop": read_json(validation_dir / "synthetic_oop_validation_summary.json"),
        "manual_crop_zdisc": read_json(validation_dir / "zdisc_mask_validation_summary.json"),
        "manual_full_image_zdisc": read_json(validation_dir / "full_image_zdisc_mask_validation_summary.json"),
        "manual_full_image_patch_zdisc": read_json(validation_dir / "full_image_patch_mask_validation_summary.json"),
        "feature_assembly": read_json(tables_dir / "feature_assembly_summary.json"),
        "project_audit": read_json(root / "project_audit_summary.json"),
    }
    status = {
        "synthetic_oop_validation": synthetic_oop_status(summaries["synthetic_oop"]),
        "manual_crop_zdisc_mask_validation": manual_crop_status(summaries["manual_crop_zdisc"]),
        "manual_full_image_zdisc_mask_validation": manual_full_image_status(summaries["manual_full_image_zdisc"]),
        "manual_full_image_patch_mask_validation": manual_full_image_patch_status(summaries["manual_full_image_patch_zdisc"]),
        "spacing": spacing_status(summaries["feature_assembly"]),
        "overall_validation_decision": overall_decision(),
        "source_summary_presence": {key: bool(value) for key, value in summaries.items()},
    }
    return json_safe(status)


def synthetic_oop_status(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"status": "missing", "caveat": "Synthetic OOP validation summary was not found."}
    return {
        "status": "controlled_implementation_validated",
        "synthetic_example_count": summary.get("synthetic_examples"),
        "clean_angular_error_median_deg": summary.get("clean_case_median_angular_error_deg"),
        "clean_angular_error_max_deg": summary.get("clean_case_max_angular_error_deg"),
        "oop_monotonicity_low_gt_medium_gt_high": summary.get("oop_monotonicity_low_gt_medium_gt_high"),
        "recovered_oop_median_by_disorder_level": summary.get("recovered_oop_median_by_disorder_level"),
        "degradation_robustness_summary": summary.get("degradation_failure_modes"),
        "caveat": "Controlled synthetic recovery validates implementation behavior only; it does not prove real-tissue biological validity.",
    }


def manual_crop_status(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"status": "missing", "caveat": "Manual crop Z-disc validation summary was not found."}
    return {
        "status": "pilot_only_not_confirmatory",
        "masks": summary.get("total_annotation_masks"),
        "zdisc_labeled_count": summary.get("masks_with_zdisc_labels"),
        "orientation_pairs": summary.get("n_orientation_pairs"),
        "median_angular_error_deg": summary.get("median_axial_error_deg"),
        "oop_medians": summary.get("oop_medians_by_annotation_status"),
        "caveat": "User-drawn sparse crop masks are pilot annotations, not blinded expert validation.",
    }


def manual_full_image_status(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"status": "missing", "caveat": "Manual full-image Z-disc validation summary was not found."}
    return {
        "status": "pilot_only_not_confirmatory",
        "full_images": summary.get("total_full_image_annotations"),
        "labeled_images": summary.get("images_with_zdisc_labels"),
        "orientation_pairs": summary.get("n_orientation_pairs"),
        "median_image_level_angular_error_deg": summary.get("median_axial_error_deg"),
        "oop_medians_by_status": summary.get("oop_medians_by_annotation_status"),
        "caveat": "Sparse local manual masks are mismatched to global image-level orientation/OOP metrics.",
    }


def manual_full_image_patch_status(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"status": "missing", "caveat": "Full-image patch mask validation summary was not found."}
    return {
        "status": "pilot_only_not_confirmatory",
        "patch_rows": summary.get("total_automated_patches_in_annotated_images"),
        "zdisc_labeled_patches": summary.get("patches_with_manual_zdisc_labels"),
        "orientation_pairs": summary.get("n_orientation_pairs"),
        "median_patch_level_angular_error_deg": summary.get("median_axial_error_deg"),
        "oop_medians": summary.get("oop_medians_by_manual_patch_status"),
        "spearman_rho": nested_get(summary, ["spearman_zdisc_fraction_vs_patch_oop", "rho"]),
        "spearman_computed": nested_get(summary, ["spearman_zdisc_fraction_vs_patch_oop", "computed"]),
        "caveat": "Pilot patch-level comparison did not support OOP validation against the current sparse Z-disc masks.",
    }


def spacing_status(summary: dict[str, Any] | None) -> dict[str, Any]:
    valid_patches = summary.get("total_valid_spacing_patches") if summary else None
    return {
        "status": "exploratory_low_yield",
        "valid_spacing_patches": valid_patches,
        "primary_endpoint": False,
        "statement": "Spacing is preserved as an exploratory low-yield descriptor and is not a primary endpoint.",
    }


def overall_decision() -> dict[str, Any]:
    return {
        "oop_implementation": "validated_on_controlled_synthetic_data",
        "real_tissue_oop": "not_expert_validated_unresolved",
        "manual_zdisc_masks": "useful_pilot_annotations_not_sufficient_validation",
        "next_required_evidence": "expert_or_user_manual_organisation_orientation_annotation_not_more_zdisc_masks",
        "clinical_statistical_analysis": "downstream_and_caveated_until_validation_route_is_clarified",
        "plain_language_summary": (
            "The OOP/orientation code behaves correctly on controlled synthetic striations, but current manual "
            "Z-disc mask pilots do not validate OOP on real archival tissue. Real-tissue OOP therefore remains "
            "unresolved and needs a better expert/manual organisation-orientation validation route before strong claims."
        ),
    }


def nested_get(data: dict[str, Any], keys: list[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_validation_status_outputs(status: dict[str, Any], paths: dict[str, Path]) -> None:
    paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_json"].write_text(json.dumps(json_safe(status), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_text_status(status), encoding="utf-8")
    paths["markdown"].write_text(render_markdown_status(status), encoding="utf-8")


def render_text_status(status: dict[str, Any]) -> str:
    lines = [
        "Validation Status Summary",
        "",
        f"Synthetic OOP: {status['synthetic_oop_validation']['status']}",
        f"Manual crop Z-disc masks: {status['manual_crop_zdisc_mask_validation']['status']}",
        f"Manual full-image Z-disc masks: {status['manual_full_image_zdisc_mask_validation']['status']}",
        f"Manual full-image patch masks: {status['manual_full_image_patch_mask_validation']['status']}",
        f"Spacing: {status['spacing']['status']}",
        "",
        status["overall_validation_decision"]["plain_language_summary"],
    ]
    return "\n".join(lines) + "\n"


def render_markdown_status(status: dict[str, Any]) -> str:
    synthetic = status["synthetic_oop_validation"]
    crop = status["manual_crop_zdisc_mask_validation"]
    full_image = status["manual_full_image_zdisc_mask_validation"]
    patch = status["manual_full_image_patch_mask_validation"]
    spacing = status["spacing"]
    decision = status["overall_validation_decision"]
    lines = [
        "# Validation Status",
        "",
        "This document reconciles the current validation evidence without changing algorithms, thresholds, feature tables, masks, or outputs.",
        "",
        "## Classification",
        "",
        f"- Synthetic OOP validation: `{synthetic['status']}`",
        f"- Manual crop Z-disc mask validation: `{crop['status']}`",
        f"- Manual full-image Z-disc mask validation: `{full_image['status']}`",
        f"- Full-image patch mask validation: `{patch['status']}`",
        f"- Spacing: `{spacing['status']}`",
        "",
        "## Synthetic OOP Validation",
        "",
        f"- Synthetic examples: {format_value(synthetic.get('synthetic_example_count'))}",
        f"- Clean angular error median: {format_value(synthetic.get('clean_angular_error_median_deg'))} deg",
        f"- Clean angular error max: {format_value(synthetic.get('clean_angular_error_max_deg'))} deg",
        f"- OOP monotonicity low > medium > high: {format_value(synthetic.get('oop_monotonicity_low_gt_medium_gt_high'))}",
        f"- Recovered OOP by disorder: {format_value(synthetic.get('recovered_oop_median_by_disorder_level'))}",
        f"- Caveat: {synthetic.get('caveat')}",
        "",
        "## Manual Crop Z-Disc Masks",
        "",
        f"- Masks: {format_value(crop.get('masks'))}",
        f"- Z-disc-labeled masks: {format_value(crop.get('zdisc_labeled_count'))}",
        f"- Orientation pairs: {format_value(crop.get('orientation_pairs'))}",
        f"- Median angular error: {format_value(crop.get('median_angular_error_deg'))} deg",
        f"- OOP medians: {format_value(crop.get('oop_medians'))}",
        f"- Caveat: {crop.get('caveat')}",
        "",
        "## Manual Full-Image Z-Disc Masks",
        "",
        f"- Full images: {format_value(full_image.get('full_images'))}",
        f"- Labeled images: {format_value(full_image.get('labeled_images'))}",
        f"- Orientation pairs: {format_value(full_image.get('orientation_pairs'))}",
        f"- Median image-level angular error: {format_value(full_image.get('median_image_level_angular_error_deg'))} deg",
        f"- OOP medians: {format_value(full_image.get('oop_medians_by_status'))}",
        f"- Caveat: {full_image.get('caveat')}",
        "",
        "## Full-Image Patch Mask Validation",
        "",
        f"- Patch rows: {format_value(patch.get('patch_rows'))}",
        f"- Z-disc-labeled patches: {format_value(patch.get('zdisc_labeled_patches'))}",
        f"- Orientation pairs: {format_value(patch.get('orientation_pairs'))}",
        f"- Median patch-level angular error: {format_value(patch.get('median_patch_level_angular_error_deg'))} deg",
        f"- OOP medians: {format_value(patch.get('oop_medians'))}",
        f"- Spearman rho: {format_value(patch.get('spearman_rho'))}",
        f"- Caveat: {patch.get('caveat')}",
        "",
        "## Spacing",
        "",
        f"- Status: `{spacing['status']}`",
        f"- Valid spacing patches: {format_value(spacing.get('valid_spacing_patches'))}",
        f"- Statement: {spacing['statement']}",
        "",
        "## Overall Decision",
        "",
        f"- OOP implementation: `{decision['oop_implementation']}`",
        f"- Real-tissue OOP: `{decision['real_tissue_oop']}`",
        f"- Manual Z-disc masks: `{decision['manual_zdisc_masks']}`",
        f"- Next required evidence: `{decision['next_required_evidence']}`",
        f"- Clinical/statistical analysis: `{decision['clinical_statistical_analysis']}`",
        "",
        decision["plain_language_summary"],
        "",
    ]
    return "\n".join(lines)


def format_value(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float) and not np.isfinite(value):
        return "not available"
    return str(value)
