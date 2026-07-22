from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import output_dir
from .zdisc_annotation import json_safe


def default_confocal_pilot_report_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_pilot"
    docs_dir = Path(docs_directory) if docs_directory else Path("docs")
    return {
        "json": root / "confocal_pilot_interpretation.json",
        "txt": root / "confocal_pilot_interpretation.txt",
        "markdown": docs_dir / "CONFOCAL_PILOT_INTERPRETATION.md",
    }


def write_confocal_pilot_report(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    report = build_confocal_pilot_report(cfg)
    paths = default_confocal_pilot_report_paths(cfg, output_directory, docs_directory)
    write_confocal_pilot_report_outputs(report, paths)
    return report, paths


def build_confocal_pilot_report(cfg: dict[str, Any]) -> dict[str, Any]:
    root = output_dir(cfg)
    sources = {
        "baseline": read_json(root / "confocal_baseline" / "confocal_baseline_summary.json"),
        "mask": read_json(root / "confocal_striation_mask" / "confocal_striation_mask_summary.json"),
        "sensitivity": read_json(root / "confocal_striation_sensitivity" / "confocal_striation_sensitivity_summary.json"),
        "selective": read_json(root / "confocal_selective_analysis" / "confocal_selective_summary.json"),
        "same_grid_oop": read_json(root / "confocal_same_grid_oop" / "confocal_same_grid_oop_summary.json"),
    }
    report = {
        "mode": "confocal_pilot_interpretation",
        "confocal_dataset_intake": confocal_dataset_intake(sources),
        "baseline_transfer_audit": baseline_transfer_audit(sources),
        "selective_confident_striation_mask": selective_confident_striation_mask(sources),
        "same_grid_selected_region_oop": same_grid_selected_region_oop(sources),
        "answer_to_natalia": answer_to_natalia(),
        "calibration_and_spacing": calibration_and_spacing(sources),
        "next_recommended_steps": next_recommended_steps(),
        "claims_allowed": claims_allowed(),
        "claims_not_allowed": claims_not_allowed(),
        "final_confocal_pilot_classification": "selective_region_analysis_feasible_exploratory_needs_manual_review",
        "source_summary_presence": {key: value is not None for key, value in sources.items()},
    }
    return json_safe(report)


def confocal_dataset_intake(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    baseline = sources.get("baseline") or {}
    return {
        "image_count": baseline.get("confocal_image_count"),
        "tiffs_present_statement": "11 TIFFs present and processed in this pilot.",
        "processed_ok": baseline.get("processed_ok"),
        "processed_error": baseline.get("processed_error"),
        "filenames": baseline.get("filenames"),
        "expected_positives": baseline.get("expected_positive_examples"),
        "complex_examples": baseline.get("noted_complex_examples"),
        "interpretation": "Confocal intake succeeded without processing errors." if baseline else "Confocal baseline summary missing.",
    }


def baseline_transfer_audit(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    baseline = sources.get("baseline") or {}
    orientation = baseline.get("orientation_summary") or {}
    valid_total = orientation.get("valid_orientation_patch_count_total")
    patch_rows = baseline.get("patch_rows")
    return {
        "baseline_patch_rows": patch_rows,
        "valid_orientation_patch_count_total": valid_total,
        "valid_orientation_patch_fraction_median": orientation.get("valid_orientation_patch_fraction_median"),
        "image_oop_median": orientation.get("image_oop_median"),
        "conclusion": "widefield_qc_not_transferable_unchanged",
        "interpretation": (
            f"Existing widefield patch QC barely admitted confocal patches ({valid_total}/{patch_rows}), "
            "so widefield QC should not be transferred unchanged."
            if baseline
            else "Baseline transfer audit missing."
        ),
    }


def selective_confident_striation_mask(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    mask = sources.get("mask") or {}
    sensitivity = sources.get("sensitivity") or {}
    selective = sources.get("selective") or {}
    moderate = first_record(sensitivity.get("best_plausible_variants"), "variant_id", "moderate")
    return {
        "default_gate_assessment": nested_get(sensitivity, ["default_assessment", "classification"]) or "missing",
        "why_default_was_too_broad": sensitivity.get("why_default_was_too_broad"),
        "moderate_gate_classification": (moderate or {}).get("classification"),
        "moderate_candidate_fractions": {
            "5138": (moderate or {}).get("candidate_fraction_5138"),
            "6052_CLEAR_STRIPES": (moderate or {}).get("candidate_fraction_6052"),
            "3112": (moderate or {}).get("candidate_fraction_3112"),
            "7028": candidate_fraction_from_records(selective.get("candidate_fraction_by_image"), "7028"),
        },
        "mask_default_candidate_fraction": mask.get("candidate_patch_fraction"),
        "selected_variant": selective.get("selected_variant"),
        "selected_candidate_patch_count": selective.get("candidate_patch_count"),
        "interpretation": (
            "The default confocal gate was too broad; the moderate gate is classified as plausible_for_review and should be inspected visually."
        ),
    }


def same_grid_selected_region_oop(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    same_grid = sources.get("same_grid_oop") or {}
    selective = sources.get("selective") or {}
    oop_summary = same_grid.get("selected_vs_all_oop_summary") or {}
    selected_summaries = same_grid.get("selected_region_summaries") or []
    return {
        "same_grid_patch_rows": same_grid.get("same_grid_patch_rows"),
        "patches_processed_ok": same_grid.get("patches_processed_ok"),
        "patches_error": same_grid.get("patches_error"),
        "candidate_patch_count": same_grid.get("candidate_patch_count"),
        "selected_vs_all_oop_summary": oop_summary,
        "selected_vs_all_coherence_and_gradient_from_selective_analysis": selective.get("selected_vs_all_comparison"),
        "positive_examples": {
            "5138": first_matching_image(selected_summaries, "5138"),
            "6052_CLEAR_STRIPES": first_matching_image(selected_summaries, "6052"),
        },
        "complex_example_3112": first_matching_image(selected_summaries, "3112"),
        "review_needed_7028": first_matching_image(selected_summaries, "7028"),
        "interpretation": (
            "Same-grid OOP was computed directly on the 128 px candidate-mask grid. Selected regions had higher median OOP and coherence than all regions overall."
            if same_grid
            else "Same-grid OOP summary missing."
        ),
    }


def answer_to_natalia() -> dict[str, Any]:
    return {
        "short_answer": "yes_feasible_exploratory",
        "statement": (
            "Yes, selective confident-region analysis appears feasible on the confocal images. "
            "Analysing candidate striated regions gives cleaner OOP/coherence summaries than analysing all signal, "
            "but this remains exploratory and needs visual/manual review before biological claims."
        ),
    }


def calibration_and_spacing(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    baseline = sources.get("baseline") or {}
    same_grid = sources.get("same_grid_oop") or {}
    return {
        "confocal_pixel_size_um": "unknown",
        "baseline_spacing_status": baseline.get("spacing_calibration_status"),
        "same_grid_spacing_status": same_grid.get("spacing_status"),
        "spacing_in_microns_reported": False,
        "interpretation": (
            "No spacing in microns is reported because confocal pixel calibration is unknown. "
            "Spacing may become feasible later only if pixel calibration and clear Z-discs are available."
        ),
    }


def next_recommended_steps() -> list[str]:
    return [
        "Ask Natalia for confocal pixel size or Leica metadata if available.",
        "Manually review moderate overlays for 5138, 6052-CLEAR_STRIPES, 3112, and 7028.",
        "Optionally create a small confocal annotation pack.",
        "Only after review, consider a confocal-specific validated configuration.",
        "Do not merge confocal thresholds into the widefield default configuration.",
    ]


def claims_allowed() -> list[str]:
    return [
        "The confocal transfer pilot processed all available confocal images.",
        "The existing widefield QC gate did not transfer unchanged to confocal images.",
        "The moderate selective mask is plausible for visual review.",
        "Selected regions show higher OOP/coherence than all regions in the current same-grid audit.",
        "Selective-region analysis appears feasible but remains exploratory.",
    ]


def claims_not_allowed() -> list[str]:
    return [
        "Confocal OOP is biologically validated.",
        "The moderate mask is a final Z-disc or striation segmentation.",
        "Spacing in microns is measured from the confocal images.",
        "Disease or healthy conclusions can be drawn from this pilot.",
        "Widefield conclusions are overturned by this confocal pilot.",
    ]


def write_confocal_pilot_report_outputs(report: dict[str, Any], paths: dict[str, Path]) -> None:
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    paths["txt"].write_text(render_text_report(report), encoding="utf-8")
    paths["markdown"].write_text(render_markdown_report(report), encoding="utf-8")


def render_text_report(report: dict[str, Any]) -> str:
    intake = report["confocal_dataset_intake"]
    baseline = report["baseline_transfer_audit"]
    mask = report["selective_confident_striation_mask"]
    oop = report["same_grid_selected_region_oop"]
    lines = [
        "Confocal Pilot Interpretation",
        "",
        f"Final classification: {report['final_confocal_pilot_classification']}",
        f"Images: {intake.get('image_count')}",
        f"Processed OK/errors: {intake.get('processed_ok')}/{intake.get('processed_error')}",
        f"Expected positives: {intake.get('expected_positives')}",
        f"Complex examples: {intake.get('complex_examples')}",
        "",
        "Baseline transfer audit:",
        f"- {baseline.get('interpretation')}",
        "",
        "Selective confident-striation mask:",
        f"- default gate assessment: {mask.get('default_gate_assessment')}",
        f"- moderate gate classification: {mask.get('moderate_gate_classification')}",
        f"- moderate candidate fractions: {mask.get('moderate_candidate_fractions')}",
        "",
        "Same-grid selected-region OOP:",
        f"- patch rows: {oop.get('same_grid_patch_rows')}",
        f"- selected-vs-all summary: {oop.get('selected_vs_all_oop_summary')}",
        "",
        f"Answer to Natalia: {report['answer_to_natalia']['statement']}",
        "",
        f"Calibration/spacing: {report['calibration_and_spacing']['interpretation']}",
        "",
        "Allowed claims:",
    ]
    lines.extend(f"- {claim}" for claim in report["claims_allowed"])
    lines.extend(["", "Claims not allowed:"])
    lines.extend(f"- {claim}" for claim in report["claims_not_allowed"])
    return "\n".join(lines) + "\n"


def render_markdown_report(report: dict[str, Any]) -> str:
    intake = report["confocal_dataset_intake"]
    baseline = report["baseline_transfer_audit"]
    mask = report["selective_confident_striation_mask"]
    oop = report["same_grid_selected_region_oop"]
    calibration = report["calibration_and_spacing"]
    lines = [
        "# Confocal Pilot Interpretation",
        "",
        "This report consolidates existing confocal pilot outputs only. It does not change algorithms, thresholds, widefield outputs, or production tables.",
        "",
        "## 1. Confocal Dataset Intake",
        "",
        f"- Images processed: {format_value(intake.get('image_count'))}",
        "- Confocal files present: 11 TIFFs, not 10.",
        f"- Processed OK: {format_value(intake.get('processed_ok'))}",
        f"- Processed errors: {format_value(intake.get('processed_error'))}",
        f"- Expected positives: {format_value(intake.get('expected_positives'))}",
        f"- Complex example: {format_value(intake.get('complex_examples'))}",
        "",
        "## 2. Baseline Transfer Audit",
        "",
        f"- Baseline patch rows: {format_value(baseline.get('baseline_patch_rows'))}",
        f"- Valid orientation patch count total: {format_value(baseline.get('valid_orientation_patch_count_total'))}",
        f"- Median valid orientation patch fraction: {format_value(baseline.get('valid_orientation_patch_fraction_median'))}",
        f"- Conclusion: `{baseline.get('conclusion')}`",
        f"- {baseline.get('interpretation')}",
        "",
        "## 3. Selective Confident-Striation Mask",
        "",
        f"- Default gate assessment: `{mask.get('default_gate_assessment')}`",
        f"- Why default was too broad: {format_value(mask.get('why_default_was_too_broad'))}",
        f"- Moderate gate classification: `{mask.get('moderate_gate_classification')}`",
        f"- Candidate fractions: {format_value(mask.get('moderate_candidate_fractions'))}",
        f"- Selected variant: `{mask.get('selected_variant')}`",
        f"- Selected candidate patch count: {format_value(mask.get('selected_candidate_patch_count'))}",
        "",
        "## 4. Same-Grid Selected-Region OOP",
        "",
        f"- Same-grid patch rows: {format_value(oop.get('same_grid_patch_rows'))}",
        f"- Patches processed OK/errors: {format_value(oop.get('patches_processed_ok'))}/{format_value(oop.get('patches_error'))}",
        f"- Candidate patch count: {format_value(oop.get('candidate_patch_count'))}",
        f"- Selected-vs-all OOP summary: {format_value(oop.get('selected_vs_all_oop_summary'))}",
        f"- Selective coherence/gradient summary: {format_value(oop.get('selected_vs_all_coherence_and_gradient_from_selective_analysis'))}",
        f"- 5138: {format_value(nested_get(oop, ['positive_examples', '5138']))}",
        f"- 6052-CLEAR_STRIPES: {format_value(nested_get(oop, ['positive_examples', '6052_CLEAR_STRIPES']))}",
        f"- 3112: {format_value(oop.get('complex_example_3112'))}",
        f"- 7028: {format_value(oop.get('review_needed_7028'))}",
        "",
        "## 5. Answer To Natalia",
        "",
        report["answer_to_natalia"]["statement"],
        "",
        "## 6. Calibration And Spacing",
        "",
        f"- Confocal pixel size: `{calibration.get('confocal_pixel_size_um')}`",
        f"- Spacing in microns reported: {format_value(calibration.get('spacing_in_microns_reported'))}",
        f"- {calibration.get('interpretation')}",
        "",
        "## 7. Next Recommended Steps",
        "",
    ]
    lines.extend(f"- {item}" for item in report["next_recommended_steps"])
    lines.extend(["", "## 8. Allowed Claims", ""])
    lines.extend(f"- {claim}" for claim in report["claims_allowed"])
    lines.extend(["", "## 9. Claims Not Allowed", ""])
    lines.extend(f"- {claim}" for claim in report["claims_not_allowed"])
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def first_record(records: Any, key: str, value: str) -> dict[str, Any] | None:
    if not isinstance(records, list):
        return None
    for record in records:
        if isinstance(record, dict) and str(record.get(key)) == value:
            return record
    return None


def first_matching_image(records: Any, needle: str) -> dict[str, Any] | None:
    if not isinstance(records, list):
        return None
    for record in records:
        if not isinstance(record, dict):
            continue
        image_id = str(record.get("confocal_image_id", ""))
        filename = str(record.get("filename", ""))
        if needle in image_id or needle in filename:
            return record
    return None


def candidate_fraction_from_records(records: Any, image_id: str) -> float | None:
    record = first_matching_image(records, image_id)
    if record is None:
        return None
    value = record.get("candidate_patch_fraction")
    return None if value is None else float(value)


def nested_get(data: dict[str, Any] | None, keys: list[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def format_value(value: Any) -> str:
    if value is None:
        return "not available"
    return str(value)
