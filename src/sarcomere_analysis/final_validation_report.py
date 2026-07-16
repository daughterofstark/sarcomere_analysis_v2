from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import output_dir
from .zdisc_annotation import json_safe


def default_final_validation_report_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> dict[str, Path]:
    validation_dir = Path(output_directory) if output_directory else output_dir(cfg) / "validation"
    docs_dir = Path(docs_directory) if docs_directory else Path("docs")
    return {
        "json": validation_dir / "final_validation_interpretation.json",
        "txt": validation_dir / "final_validation_interpretation.txt",
        "markdown": docs_dir / "FINAL_VALIDATION_INTERPRETATION.md",
    }


def write_final_validation_report(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    report = build_final_validation_report(cfg)
    paths = default_final_validation_report_paths(cfg, output_directory=output_directory, docs_directory=docs_directory)
    write_final_validation_report_outputs(report, paths)
    return report, paths


def build_final_validation_report(cfg: dict[str, Any]) -> dict[str, Any]:
    root = output_dir(cfg)
    validation_dir = root / "validation"
    tables_dir = root / "tables"
    sources = {
        "validation_status": read_json(validation_dir / "validation_status_summary.json"),
        "synthetic_oop": read_json(validation_dir / "synthetic_oop_validation_summary.json"),
        "expert_annotation": read_json(validation_dir / "expert_annotation_validation" / "expert_annotation_validation_summary.json"),
        "expert_feature_audit": read_json(validation_dir / "expert_feature_audit" / "expert_feature_audit_summary.json"),
        "expert_crop_feature_audit": read_json(
            validation_dir / "expert_crop_feature_audit" / "expert_crop_feature_audit_summary.json"
        ),
        "full_image_patch_mask": read_json(validation_dir / "full_image_patch_mask_validation_summary.json"),
        "full_image_zdisc_mask": read_json(validation_dir / "full_image_zdisc_mask_validation_summary.json"),
        "crop_zdisc_mask": read_json(validation_dir / "zdisc_mask_validation_summary.json"),
        "feature_assembly": read_json(tables_dir / "feature_assembly_summary.json"),
        "project_audit": read_json(root / "project_audit_summary.json"),
    }
    report = {
        "mode": "final_validation_interpretation",
        "dataset_and_pipeline_status": dataset_and_pipeline_status(sources),
        "sarcgraph_zdisc_detection_interpretation": sarcgraph_interpretation(sources),
        "synthetic_oop_implementation_validation": synthetic_oop_section(sources),
        "manual_expert_validation_summary": manual_expert_section(sources),
        "region_alignment_audit": region_alignment_section(sources),
        "final_interpretation": final_interpretation_section(),
        "recommended_next_directions": recommended_next_directions(),
        "claims_allowed": claims_allowed(),
        "claims_not_allowed": claims_not_allowed(),
        "source_summary_presence": {key: value is not None for key, value in sources.items()},
    }
    return json_safe(report)


def dataset_and_pipeline_status(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    feature = sources.get("feature_assembly") or {}
    audit = sources.get("project_audit") or {}
    inventory = audit.get("core_output_inventory", {}) if isinstance(audit, dict) else {}
    test_status = audit.get("test_status", {}) if isinstance(audit, dict) else {}
    return {
        "images": feature.get("per_image_rows") or nested_get(inventory, ["features_per_image", "row_count"]),
        "donors": feature.get("donor_count") or feature.get("per_donor_rows") or nested_get(inventory, ["features_per_donor", "row_count"]),
        "patch_rows": feature.get("per_patch_rows") or nested_get(inventory, ["features_per_patch", "row_count"]),
        "image_feature_rows": feature.get("per_image_rows") or nested_get(inventory, ["features_per_image", "row_count"]),
        "donor_feature_rows": feature.get("per_donor_rows") or nested_get(inventory, ["features_per_donor", "row_count"]),
        "full_pipeline_runs_successfully": True,
        "production_pipeline_frozen": True,
        "test_status_available_from_project_audit": test_status.get("provided_status"),
        "test_status_note": "Project audit test status may be stale; rerun ../sarcgraph-env/bin/python -m pytest for current count.",
        "feature_tables_exist": {
            "per_patch": bool(feature.get("per_patch_rows") or nested_get(inventory, ["features_per_patch", "exists"])),
            "per_image": bool(feature.get("per_image_rows") or nested_get(inventory, ["features_per_image", "exists"])),
            "per_donor": bool(feature.get("per_donor_rows") or nested_get(inventory, ["features_per_donor", "exists"])),
        },
    }


def sarcgraph_interpretation(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    feature = sources.get("feature_assembly") or {}
    return {
        "interpretation": "Object-level Z-disc detection and sarcomere spacing are not the right primary route for this widefield archival dataset.",
        "reasons": [
            "Z-discs are often faint, blurred, discontinuous, or locally ambiguous.",
            "Object-based Z-disc detection and spacing are fragile on these widefield images.",
            "The corrected conservative spacing estimator found too few confident patches.",
        ],
        "spacing_status": feature.get("spacing_global_status", "exploratory_low_yield"),
        "valid_spacing_patches": feature.get("total_valid_spacing_patches"),
        "primary_endpoint_allowed": False,
        "statement": "No mean sarcomere length should be reported as a primary endpoint from this dataset.",
    }


def synthetic_oop_section(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    synthetic = sources.get("synthetic_oop") or {}
    return {
        "status": "implementation_validated_on_controlled_synthetic_data" if synthetic else "missing",
        "synthetic_examples": synthetic.get("synthetic_examples"),
        "clean_case_median_angular_error_deg": synthetic.get("clean_case_median_angular_error_deg"),
        "clean_case_max_angular_error_deg": synthetic.get("clean_case_max_angular_error_deg"),
        "oop_monotonicity_low_gt_medium_gt_high": synthetic.get("oop_monotonicity_low_gt_medium_gt_high"),
        "recovered_oop_median_by_disorder_level": synthetic.get("recovered_oop_median_by_disorder_level"),
        "degradation_summary": synthetic.get("degradation_failure_modes"),
        "interpretation": "This validates implementation behavior on controlled synthetic striated images, not biological endpoint validity in real tissue.",
    }


def manual_expert_section(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    expert = sources.get("expert_annotation") or {}
    feature_audit = sources.get("expert_feature_audit") or {}
    zdisc_crop = sources.get("crop_zdisc_mask") or {}
    zdisc_full = sources.get("full_image_zdisc_mask") or {}
    zdisc_patch = sources.get("full_image_patch_mask") or {}
    return {
        "manual_zdisc_masks": {
            "status": "pilot_only_non_confirmatory",
            "crop_masks": {
                "masks": zdisc_crop.get("total_annotation_masks"),
                "orientation_pairs": zdisc_crop.get("n_orientation_pairs"),
                "median_axial_error_deg": zdisc_crop.get("median_axial_error_deg"),
            },
            "full_image_masks": {
                "annotations": zdisc_full.get("total_full_image_annotations"),
                "orientation_pairs": zdisc_full.get("n_orientation_pairs"),
                "median_axial_error_deg": zdisc_full.get("median_axial_error_deg"),
            },
            "full_image_patch_masks": {
                "patch_rows": zdisc_patch.get("total_automated_patches_in_annotated_images"),
                "orientation_pairs": zdisc_patch.get("n_orientation_pairs"),
                "median_axial_error_deg": zdisc_patch.get("median_axial_error_deg"),
            },
            "interpretation": "Manual Z-disc masks did not confirm automated OOP as real-tissue organisation validation.",
        },
        "blinded_expert_annotations": {
            "annotation_rows": nested_get(expert, ["audit", "total_rows"]),
            "matched_rows": nested_get(expert, ["audit", "annotations_matched_to_internal_key"]),
            "visibility_completed": nested_get(expert, ["audit", "completed_striations_visible_count"]),
            "organisation_score_completed": nested_get(expert, ["audit", "completed_organisation_score_count"]),
            "confidence_completed": nested_get(expert, ["audit", "completed_confidence_score_count"]),
            "manual_sarcomere_length_completed": nested_get(expert, ["audit", "manual_sarcomere_length_completed_count"]),
            "visibility_oop_medians": nested_get(expert, ["visibility_vs_automated_oop", "oop_medians"]),
            "organisation_vs_oop_spearman": nested_get(expert, ["organisation_score_vs_automated_oop", "spearman"]),
            "confidence_filtered_spearman": nested_get(expert, ["confidence_filtered", "organisation_oop_spearman"]),
            "dominant_orientation_primary_used": nested_get(expert, ["orientation", "dominant_orientation_used_as_primary"]),
            "interpretation": "Expert visibility showed a weak directional OOP trend, but expert organisation score did not correlate with OOP.",
        },
        "expert_feature_audit": {
            "features_audited": nested_get(feature_audit, ["audit", "numeric_automated_features_considered"]),
            "top_organisation_features": (feature_audit.get("top_organisation_features_by_abs_spearman") or [])[:5],
            "oop_specific_statement": feature_audit.get("oop_specific_statement"),
            "interpretation": "No existing automated patch feature strongly tracked expert organisation score in this small single-reviewer audit.",
        },
        "spacing": {
            "manual_sarcomere_length_not_completed": True,
            "spacing_validated": False,
            "status": "exploratory_low_yield",
        },
    }


def region_alignment_section(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    crop_audit = sources.get("expert_crop_feature_audit") or {}
    previous = crop_audit.get("previous_production_patch_oop_vs_organisation") or {}
    crop = crop_audit.get("crop_oop_vs_organisation") or {}
    confidence_filtered = crop_audit.get("crop_oop_vs_organisation_confidence_filtered") or {}
    return {
        "status": "completed" if crop_audit else "missing",
        "rationale": "Natalia scored larger expert-visible crops, while the first expert audit used automated features from the internal production patch.",
        "previous_production_patch_oop_vs_organisation": previous,
        "expert_visible_crop_oop_vs_organisation": crop,
        "expert_visible_crop_oop_vs_organisation_confidence_filtered": confidence_filtered,
        "top_crop_organisation_features": (crop_audit.get("top_organisation_features_by_abs_spearman") or [])[:5],
        "top_confidence_filtered_crop_organisation_features": (
            crop_audit.get("top_confidence_filtered_organisation_features_by_abs_spearman") or []
        )[:5],
        "interpretation": (
            "Region definition affected feature relationships, but expert-visible crop OOP was inversely associated with "
            "expert organisation score and therefore still does not validate OOP as Natalia-rated organisation."
            if crop_audit
            else "Expert-visible crop feature audit not available."
        ),
    }


def final_interpretation_section() -> dict[str, str]:
    return {
        "oop_orientation_implementation": "validated_on_synthetic_controlled_data",
        "production_patch_oop_as_expert_organisation_endpoint": "not_validated",
        "expert_visible_crop_oop_as_expert_organisation_endpoint": "inversely_associated_not_validated",
        "real_tissue_oop_as_expert_organisation_endpoint": "not_validated",
        "striation_visibility": "weakly_reflected_by_oop",
        "sarcomere_spacing": "not_validated_exploratory_low_yield",
        "classical_descriptors": "may_capture_texture_anisotropy_or_visibility_but_not_natalia_biological_organisation_score",
        "automated_current_pipeline": (
            "useful_as_reproducible_image_texture_orientation_audit_not_yet_validated_biological_organisation_biomarker"
        ),
    }


def recommended_next_directions() -> list[str]:
    return [
        "If biological organisation quantification remains the goal, collect a higher-quality confocal subset.",
        "Repeat the same blinded annotation framework on confocal images.",
        "Consider larger expert/manual organisation scoring with clearer definitions.",
        "Consider supervised or semi-supervised models only after adequate labelled data exists.",
        "If the thesis/report deadline is near, frame the project as method development and negative validation on challenging archival widefield data.",
        "Emphasise reproducibility, auditability, and honest endpoint triage.",
        "Do not continue tuning OOP or spacing on the current data without new validation evidence.",
    ]


def claims_allowed() -> list[str]:
    return [
        "The pipeline processes the full dataset reproducibly.",
        "Spacing is low-yield in this dataset.",
        "Synthetic OOP validation passes on controlled striated images.",
        "Expert validation does not support OOP as a standalone organisation score.",
        "Expert-visible crop audit showed region definition affects feature relationships.",
        "Crop intensity and texture features may be worth monitoring exploratorily.",
        "Widefield archival images are challenging for object-level sarcomere analysis.",
    ]


def claims_not_allowed() -> list[str]:
    return [
        "OOP is validated as expert-rated sarcomere organisation.",
        "Disease/healthy differences are biologically meaningful based on OOP.",
        "Sarcomere length can be robustly measured from this dataset.",
        "Crop OOP validates expert organisation.",
        "The inverse crop OOP association should be interpreted biologically without further validation.",
        "Current widefield OOP supports disease/healthy biological comparisons.",
        "SarcGraph failed because of implementation error.",
    ]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(data: dict[str, Any] | None, keys: list[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def write_final_validation_report_outputs(report: dict[str, Any], paths: dict[str, Path]) -> None:
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    paths["txt"].write_text(render_text_report(report), encoding="utf-8")
    paths["markdown"].write_text(render_markdown_report(report), encoding="utf-8")


def render_text_report(report: dict[str, Any]) -> str:
    region = report["region_alignment_audit"]
    lines = [
        "Final Validation Interpretation",
        "",
        f"Images: {report['dataset_and_pipeline_status'].get('images')}",
        f"Donors: {report['dataset_and_pipeline_status'].get('donors')}",
        f"Patch rows: {report['dataset_and_pipeline_status'].get('patch_rows')}",
        "",
        "Region-alignment audit:",
        f"- previous production-patch OOP vs organisation: {region.get('previous_production_patch_oop_vs_organisation')}",
        f"- expert-visible crop OOP vs organisation: {region.get('expert_visible_crop_oop_vs_organisation')}",
        f"- confidence-filtered crop OOP vs organisation: {region.get('expert_visible_crop_oop_vs_organisation_confidence_filtered')}",
        f"- interpretation: {region.get('interpretation')}",
        "",
        "Final interpretation:",
    ]
    for key, value in report["final_interpretation"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Allowed claims:"])
    lines.extend(f"- {claim}" for claim in report["claims_allowed"])
    lines.extend(["", "Claims not allowed:"])
    lines.extend(f"- {claim}" for claim in report["claims_not_allowed"])
    return "\n".join(lines) + "\n"


def render_markdown_report(report: dict[str, Any]) -> str:
    dataset = report["dataset_and_pipeline_status"]
    sarcgraph = report["sarcgraph_zdisc_detection_interpretation"]
    synthetic = report["synthetic_oop_implementation_validation"]
    manual = report["manual_expert_validation_summary"]
    region = report["region_alignment_audit"]
    final = report["final_interpretation"]
    lines = [
        "# Final Validation Interpretation",
        "",
        "This report consolidates the validation evidence without changing algorithms, thresholds, feature tables, annotations, masks, or outputs.",
        "",
        "## 1. Dataset And Pipeline Status",
        "",
        f"- Images: {format_value(dataset.get('images'))}",
        f"- Donors: {format_value(dataset.get('donors'))}",
        f"- Patch feature rows: {format_value(dataset.get('patch_rows'))}",
        f"- Image feature rows: {format_value(dataset.get('image_feature_rows'))}",
        f"- Donor feature rows: {format_value(dataset.get('donor_feature_rows'))}",
        f"- Full pipeline runs successfully: {format_value(dataset.get('full_pipeline_runs_successfully'))}",
        f"- Production pipeline frozen: {format_value(dataset.get('production_pipeline_frozen'))}",
        f"- Test status from stored audit: {format_value(dataset.get('test_status_available_from_project_audit'))}",
        "",
        "## 2. Why Object-Level Z-Disc Detection Was Not The Primary Route",
        "",
        sarcgraph["interpretation"],
        "",
    ]
    lines.extend(f"- {reason}" for reason in sarcgraph["reasons"])
    lines.extend(
        [
            f"- Spacing status: `{sarcgraph.get('spacing_status')}`",
            f"- Valid spacing patches: {format_value(sarcgraph.get('valid_spacing_patches'))}",
            f"- Primary spacing endpoint allowed: {format_value(sarcgraph.get('primary_endpoint_allowed'))}",
            f"- {sarcgraph['statement']}",
            "",
            "## 3. What OOP/Orientation Validates",
            "",
            f"- Synthetic examples: {format_value(synthetic.get('synthetic_examples'))}",
            f"- Clean median angular error: {format_value(synthetic.get('clean_case_median_angular_error_deg'))} deg",
            f"- Clean max angular error: {format_value(synthetic.get('clean_case_max_angular_error_deg'))} deg",
            f"- OOP monotonicity low > medium > high: {format_value(synthetic.get('oop_monotonicity_low_gt_medium_gt_high'))}",
            f"- Recovered OOP by disorder: {format_value(synthetic.get('recovered_oop_median_by_disorder_level'))}",
            f"- Interpretation: {synthetic['interpretation']}",
            "",
            "## 4. What Manual/Expert Validation Showed",
            "",
            f"- Manual Z-disc masks: {manual['manual_zdisc_masks']['interpretation']}",
            f"- Expert annotation rows: {format_value(nested_get(manual, ['blinded_expert_annotations', 'annotation_rows']))}",
            f"- Expert matched rows: {format_value(nested_get(manual, ['blinded_expert_annotations', 'matched_rows']))}",
            f"- Visibility OOP medians: {format_value(nested_get(manual, ['blinded_expert_annotations', 'visibility_oop_medians']))}",
            f"- Organisation vs OOP Spearman: {format_value(nested_get(manual, ['blinded_expert_annotations', 'organisation_vs_oop_spearman']))}",
            f"- Confidence-filtered Spearman: {format_value(nested_get(manual, ['blinded_expert_annotations', 'confidence_filtered_spearman']))}",
            "- Dominant orientation column excluded because reviewer reported ambiguity.",
            "- Manual sarcomere length was not completed; spacing was not validated.",
            f"- Feature audit: {manual['expert_feature_audit']['interpretation']}",
            "",
            "## 5. Region-Alignment Audit",
            "",
            f"- Rationale: {region['rationale']}",
            f"- Previous production-patch OOP vs organisation: {format_value(region.get('previous_production_patch_oop_vs_organisation'))}",
            f"- Expert-visible crop OOP vs organisation: {format_value(region.get('expert_visible_crop_oop_vs_organisation'))}",
            f"- Confidence-filtered crop OOP vs organisation: {format_value(region.get('expert_visible_crop_oop_vs_organisation_confidence_filtered'))}",
            f"- Interpretation: {region['interpretation']}",
            "",
            "## 6. Final Interpretation",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in final.items())
    lines.extend(["", "## 7. Recommended Next Directions", ""])
    lines.extend(f"- {item}" for item in report["recommended_next_directions"])
    lines.extend(["", "## 8. Claims Allowed", ""])
    lines.extend(f"- {claim}" for claim in report["claims_allowed"])
    lines.extend(["", "## 9. Claims Not Allowed", ""])
    lines.extend(f"- {claim}" for claim in report["claims_not_allowed"])
    lines.append("")
    return "\n".join(lines)


def format_value(value: Any) -> str:
    if value is None:
        return "not available"
    return str(value)
