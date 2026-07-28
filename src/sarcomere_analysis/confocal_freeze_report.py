from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import output_dir
from .zdisc_annotation import json_safe


PRIMARY_GATE = "moderate"
RELAXED_GATE_STATUS = "moderate_relaxed_combined_sensitivity_only_not_primary"

MANUAL_SPOT_CHECK_VERDICTS = [
    {
        "filename": "EB98A_1.tif",
        "verdict": "pass",
        "caveat": "not perfect",
        "review_status": "reviewed_in_chat",
    },
    {"filename": "8A793.tif", "verdict": "pass", "caveat": "", "review_status": "reviewed_in_chat"},
    {
        "filename": "94217.tif",
        "verdict": "pass",
        "caveat": "broad-selection caveat",
        "review_status": "reviewed_in_chat",
    },
    {"filename": "B23E3_1.tif", "verdict": "pass", "caveat": "", "review_status": "reviewed_in_chat"},
    {"filename": "E0ABF_1.tif", "verdict": "pass", "caveat": "", "review_status": "reviewed_in_chat"},
    {"filename": "E0ABF_2.tif", "verdict": "pass", "caveat": "", "review_status": "reviewed_in_chat"},
    {
        "filename": "E0ABF.tif",
        "verdict": "algorithmically_spacing_reportable_pending_visual_confirmation",
        "caveat": "not manually reviewed in-chat",
        "review_status": "pending_visual_confirmation",
    },
]


def default_confocal_freeze_report_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_freeze_report"
    docs_root = Path(docs_directory) if docs_directory else Path("docs")
    return {
        "root": root,
        "json": root / "confocal_freeze_summary.json",
        "txt": root / "confocal_freeze_summary.txt",
        "markdown": docs_root / "CONFOCAL_FREEZE_REPORT.md",
    }


def write_confocal_freeze_report(
    cfg: dict[str, Any],
    pipeline_dir: str | Path,
    audit_dir: str | Path,
    endpoint_dir: str | Path,
    review_pack_dir: str | Path,
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    report = build_confocal_freeze_report(
        pipeline_dir=Path(pipeline_dir),
        audit_dir=Path(audit_dir),
        endpoint_dir=Path(endpoint_dir),
        review_pack_dir=Path(review_pack_dir),
    )
    paths = default_confocal_freeze_report_paths(cfg, output_directory, docs_directory)
    write_confocal_freeze_outputs(report, paths)
    return report, paths


def build_confocal_freeze_report(
    pipeline_dir: Path,
    audit_dir: Path,
    endpoint_dir: Path,
    review_pack_dir: Path,
) -> dict[str, Any]:
    pipeline_summary = read_required_json(pipeline_dir / "confocal_pipeline_summary.json")
    audit_summary = read_required_json(audit_dir / "confocal_larger_audit_summary.json")
    endpoint_summary = read_required_json(endpoint_dir / "confocal_endpoint_summary.json")
    endpoint_per_image = read_required_csv(endpoint_dir / "confocal_endpoint_per_image.csv")
    review_summary = read_required_json(review_pack_dir / "confocal_endpoint_review_pack_summary.json")
    review_index = read_csv_if_exists(review_pack_dir / "confocal_endpoint_review_index.csv")

    report = {
        "mode": "final_confocal_freeze_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_gate": PRIMARY_GATE,
        "relaxed_gate": RELAXED_GATE_STATUS,
        "widefield_calibration_used": bool(pipeline_summary.get("widefield_calibration_used", False)),
        "larger_dataset": larger_dataset_section(pipeline_summary, audit_summary),
        "endpoint_result": endpoint_result_section(endpoint_summary, endpoint_per_image),
        "endpoint_review_pack": endpoint_review_pack_section(review_summary, review_index),
        "manual_visual_spot_check": manual_spot_check_section(endpoint_per_image),
        "final_frozen_interpretation": final_frozen_interpretation(),
        "allowed_downstream_use": allowed_downstream_use(),
        "not_allowed_claims": not_allowed_claims(),
        "next_scientific_action": (
            "Expert review/confirmation or additional labels/images should come before clinical/disease statistics, "
            "ML, relaxed-gate adoption, or additional unsupervised threshold tuning."
        ),
        "source_summary_presence": {
            "pipeline_summary": True,
            "audit_summary": True,
            "endpoint_summary": True,
            "endpoint_per_image": True,
            "review_pack_summary": True,
            "review_pack_index": not review_index.empty,
        },
        "no_change_statement": (
            "This freeze report is documentation/reporting only. It does not rerun analysis, change thresholds, "
            "adopt the relaxed gate, modify widefield outputs, or alter production algorithms."
        ),
    }
    return json_safe(report)


def larger_dataset_section(pipeline_summary: dict[str, Any], audit_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "images": pipeline_summary.get("images_processed"),
        "errors": pipeline_summary.get("errors"),
        "calibrated_images": pipeline_summary.get("calibrated_images"),
        "total_patches": pipeline_summary.get("total_patches"),
        "selected_candidate_patches": pipeline_summary.get("selected_candidate_patches"),
        "valid_selected_spacing_patches": pipeline_summary.get("valid_selected_spacing_patches"),
        "selected_spacing_valid_fraction": audit_summary.get(
            "selected_spacing_valid_fraction", safe_fraction(pipeline_summary, "valid_selected_spacing_patches", "selected_candidate_patches")
        ),
        "median_selected_oop": pipeline_summary.get("median_selected_oop"),
        "median_selected_spacing_um": pipeline_summary.get("median_selected_spacing_um"),
        "interpretation": (
            "The larger 42-image confocal pipeline scales technically, but spacing yield is much lower than the 11-image pilot."
        ),
    }


def endpoint_result_section(endpoint_summary: dict[str, Any], endpoint_per_image: pd.DataFrame) -> dict[str, Any]:
    spacing_reportable = endpoint_per_image.loc[bool_series(endpoint_per_image.get("spacing_reportable", pd.Series(dtype=bool)))]
    return {
        "oop_reportable_images": endpoint_summary.get("oop_reportable_image_count"),
        "spacing_reportable_images": endpoint_summary.get("spacing_reportable_image_count"),
        "failed_or_unusable": (endpoint_summary.get("endpoint_class_counts") or {}).get("failed_or_unusable", 0),
        "endpoint_class_counts": normalize_endpoint_counts(endpoint_summary.get("endpoint_class_counts") or {}),
        "spacing_reportable_image_list": spacing_reportable["filename"].astype(str).tolist(),
        "interpretation": (
            "OOP/coherence is broadly available. Spacing is not universal and is reportable only for spacing-eligible selected regions."
        ),
    }


def endpoint_review_pack_section(review_summary: dict[str, Any], review_index: pd.DataFrame) -> dict[str, Any]:
    return {
        "images_included": review_summary.get("images_included"),
        "review_group_counts": review_summary.get("review_group_counts"),
        "review_image_files_copied": review_summary.get("review_image_files_copied"),
        "missing_preview_count": review_summary.get("missing_preview_count"),
        "zip_path": review_summary.get("zip_path"),
        "review_index_rows": int(len(review_index)),
        "interpretation": "Endpoint review pack exported for visual QC before downstream statistics or additional interpretation.",
    }


def manual_spot_check_section(endpoint_per_image: pd.DataFrame) -> dict[str, Any]:
    reviewed = [item for item in MANUAL_SPOT_CHECK_VERDICTS if item["review_status"] == "reviewed_in_chat"]
    accepted = [item for item in reviewed if item["verdict"] == "pass"]
    spacing_reportable = endpoint_per_image.loc[bool_series(endpoint_per_image.get("spacing_reportable", pd.Series(dtype=bool)))]
    algorithmic_spacing = spacing_reportable["filename"].astype(str).tolist()
    return {
        "reviewed_in_chat_count": len(reviewed),
        "reviewed_pass_count": len(accepted),
        "reviewed_pass_denominator": len(reviewed),
        "algorithmically_spacing_reportable_count": int(len(spacing_reportable)),
        "algorithmically_spacing_reportable_images": algorithmic_spacing,
        "verdicts": MANUAL_SPOT_CHECK_VERDICTS,
        "pending_visual_confirmation": ["E0ABF.tif"],
        "caveats": [
            "E0ABF.tif was not manually reviewed in-chat and remains algorithmic/pending visual confirmation.",
            "94217.tif passed with a broad-selection caveat.",
            "EB98A_1.tif passed with a not-perfect caveat.",
        ],
        "interpretation": (
            "Six of seven spacing-reportable images were spot-checked in-chat and the six reviewed images were acceptable "
            "as selected-region spacing-reportable. E0ABF.tif remains pending visual confirmation."
        ),
    }


def final_frozen_interpretation() -> list[str]:
    return [
        "Widefield spacing remains low-yield/negative.",
        "Widefield OOP is not validated as expert organisation.",
        "Confocal pipeline scales technically.",
        "Confocal OOP/coherence is broadly reportable.",
        "Confocal spacing is not universal.",
        "Confocal spacing is reportable only as a selected-region/subset endpoint for spacing-eligible images.",
        "Do not run clinical/disease statistics yet.",
        "Do not use ML yet.",
        "Next scientific action would be expert review/confirmation or additional labels/images, not more unsupervised tuning.",
    ]


def allowed_downstream_use() -> list[str]:
    return [
        "Report confocal OOP/coherence broadly with caveats.",
        "Report spacing only for spacing-reportable selected regions.",
        "Use endpoint flags in downstream tables.",
        "Share the endpoint review pack for expert QC.",
    ]


def not_allowed_claims() -> list[str]:
    return [
        "Whole-cohort spacing claims.",
        "Disease or clinical inference.",
        "Relaxed gate as primary.",
        "Widefield spacing claims.",
        "Claiming OOP is validated biological organisation.",
        "ML claims.",
    ]


def write_confocal_freeze_outputs(report: dict[str, Any], paths: dict[str, Path]) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["markdown"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    text = render_freeze_text(report)
    paths["txt"].write_text(text, encoding="utf-8")
    paths["markdown"].write_text(render_freeze_markdown(report), encoding="utf-8")


def render_freeze_text(report: dict[str, Any]) -> str:
    lines = [
        "Final confocal freeze report",
        f"generated_at: {report['generated_at']}",
        f"primary_gate: {report['primary_gate']}",
        f"relaxed_gate: {report['relaxed_gate']}",
        f"widefield_calibration_used: {report['widefield_calibration_used']}",
        "",
        "Larger dataset:",
    ]
    lines.extend(format_mapping(report["larger_dataset"]))
    lines.extend(["", "Endpoint result:"])
    lines.extend(format_mapping(report["endpoint_result"]))
    lines.extend(["", "Manual visual spot-check:"])
    lines.extend(format_mapping(report["manual_visual_spot_check"]))
    lines.extend(["", "Final frozen interpretation:"])
    lines.extend(f"- {item}" for item in report["final_frozen_interpretation"])
    lines.extend(["", "Allowed downstream use:"])
    lines.extend(f"- {item}" for item in report["allowed_downstream_use"])
    lines.extend(["", "Not allowed claims:"])
    lines.extend(f"- {item}" for item in report["not_allowed_claims"])
    lines.extend(["", report["no_change_statement"]])
    return "\n".join(lines) + "\n"


def render_freeze_markdown(report: dict[str, Any]) -> str:
    larger = report["larger_dataset"]
    endpoint = report["endpoint_result"]
    manual = report["manual_visual_spot_check"]
    lines = [
        "# Confocal Freeze Report",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Frozen Decision",
        "",
        f"- Primary gate: `{report['primary_gate']}`",
        f"- Relaxed gate: `{report['relaxed_gate']}`",
        f"- Widefield calibration used: `{report['widefield_calibration_used']}`",
        "",
        "## Larger Dataset",
        "",
        f"- Images: {larger.get('images')}",
        f"- Errors: {larger.get('errors')}",
        f"- Calibrated images: {larger.get('calibrated_images')}",
        f"- Total patches: {larger.get('total_patches')}",
        f"- Selected candidate patches: {larger.get('selected_candidate_patches')}",
        f"- Valid selected spacing patches: {larger.get('valid_selected_spacing_patches')}",
        f"- Selected spacing valid fraction: {larger.get('selected_spacing_valid_fraction')}",
        f"- Median selected OOP: {larger.get('median_selected_oop')}",
        f"- Median selected spacing: {larger.get('median_selected_spacing_um')} um",
        "",
        "## Endpoint Result",
        "",
        f"- OOP reportable: {endpoint.get('oop_reportable_images')}/42",
        f"- Spacing reportable: {endpoint.get('spacing_reportable_images')}/42",
        f"- Failed/unusable: {endpoint.get('failed_or_unusable')}",
        f"- Endpoint class counts: `{endpoint.get('endpoint_class_counts')}`",
        "",
        "Spacing-reportable images:",
    ]
    lines.extend(f"- `{name}`" for name in endpoint.get("spacing_reportable_image_list", []))
    lines.extend(
        [
            "",
            "## Manual Visual Spot-Check",
            "",
            f"- Reviewed in-chat: {manual.get('reviewed_in_chat_count')}/7",
            f"- Reviewed acceptable: {manual.get('reviewed_pass_count')}/{manual.get('reviewed_pass_denominator')}",
            "- `E0ABF.tif` was not manually reviewed in-chat and remains algorithmic/pending visual confirmation.",
            "- `94217.tif` passed with a broad-selection caveat.",
            "- `EB98A_1.tif` passed with a not-perfect caveat.",
            "",
            "## Final Frozen Interpretation",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["final_frozen_interpretation"])
    lines.extend(["", "## Allowed Downstream Use", ""])
    lines.extend(f"- {item}" for item in report["allowed_downstream_use"])
    lines.extend(["", "## Not Allowed Claims", ""])
    lines.extend(f"- {item}" for item in report["not_allowed_claims"])
    lines.extend(
        [
            "",
            "## No-Change Statement",
            "",
            report["no_change_statement"],
            "",
        ]
    )
    return "\n".join(lines)


def format_mapping(mapping: dict[str, Any]) -> list[str]:
    return [f"- {key}: {value}" for key, value in mapping.items()]


def normalize_endpoint_counts(counts: dict[str, Any]) -> dict[str, int]:
    required = [
        "spacing_eligible_moderate",
        "spacing_eligible_low_confidence",
        "oop_only_spacing_low_yield",
        "low_candidate_review_needed",
        "failed_or_unusable",
    ]
    return {key: int(counts.get(key, 0)) for key in required}


def read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required confocal freeze report input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required confocal freeze report input is missing: {path}")
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


def bool_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        if values.dtype == object:
            return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y"})
        return values.fillna(False).astype(bool)
    return pd.Series(values).fillna(False).astype(bool)


def safe_fraction(mapping: dict[str, Any], numerator_key: str, denominator_key: str) -> float | None:
    try:
        numerator = float(mapping.get(numerator_key))
        denominator = float(mapping.get(denominator_key))
    except (TypeError, ValueError):
        return None
    return float(numerator / denominator) if denominator else None
