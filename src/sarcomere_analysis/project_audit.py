from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

from .config import manifest_csv_path, output_dir


CORE_OUTPUT_KEYS = [
    "manifest",
    "per_patch_metrics",
    "per_image_metrics",
    "features_per_patch",
    "features_per_image",
    "features_per_donor",
    "enriched_manifest",
    "donor_metadata",
    "analysis_per_image",
    "analysis_per_donor",
    "pipeline_run_summary",
]


@dataclass(frozen=True)
class AuditPaths:
    project_root: Path
    results_root: Path
    docs_root: Path
    config_path: Path


def build_project_audit(
    cfg: dict[str, Any],
    config_path: str | Path,
    project_root: str | Path,
    docs_dir: str | Path | None = None,
    test_status: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    paths = AuditPaths(
        project_root=Path(project_root),
        results_root=output_dir(cfg),
        docs_root=Path(docs_dir) if docs_dir else Path(project_root) / "docs",
        config_path=Path(config_path),
    )
    core_paths = core_output_paths(cfg)
    core_inventory = {key: file_inventory(path) for key, path in core_paths.items()}
    optional_inventory = optional_output_inventory(paths.results_root)
    summary = {
        "generated_at": utc_now_iso(),
        "repository_state": repository_state(paths.project_root, paths.config_path),
        "test_status": {
            "provided_status": test_status,
            "test_command": "../sarcgraph-env/bin/python -m pytest",
            "tests_run_by_audit": False,
        },
        "core_output_inventory": core_inventory,
        "optional_output_inventory": optional_inventory,
        "scientific_decision_summary": scientific_decision_summary(),
        "pipeline_stage_summary": pipeline_stage_summary(),
        "reproducible_commands": reproducible_commands(),
        "safety_checks": run_safety_checks(cfg, core_paths, strict=strict),
    }
    return json_safe(summary)


def core_output_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    tables = output_dir(cfg) / "tables"
    return {
        "manifest": manifest_csv_path(cfg),
        "per_patch_metrics": tables / "per_patch_metrics.csv",
        "per_image_metrics": tables / "per_image_metrics.csv",
        "features_per_patch": tables / "features_per_patch.csv",
        "features_per_image": tables / "features_per_image.csv",
        "features_per_donor": tables / "features_per_donor.csv",
        "enriched_manifest": tables / "enriched_manifest.csv",
        "donor_metadata": tables / "donor_metadata.csv",
        "analysis_per_image": tables / "analysis_per_image.csv",
        "analysis_per_donor": tables / "analysis_per_donor.csv",
        "pipeline_run_summary": output_dir(cfg) / "pipeline_run_summary.json",
    }


def file_inventory(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "suffix": path.suffix,
        "size_bytes": None,
        "modified_time": None,
        "row_count": None,
    }
    if not path.exists():
        return info
    stat = path.stat()
    info["size_bytes"] = int(stat.st_size)
    info["modified_time"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    if path.suffix.lower() == ".csv":
        info["row_count"] = csv_row_count(path)
    return info


def optional_output_inventory(results_root: Path) -> dict[str, Any]:
    diagnostics = results_root / "diagnostics"
    review = diagnostics / "spacing_candidate_review"
    previews = results_root / "previews"
    validation = results_root / "validation"
    gallery = results_root / "qc_gallery" / "index.html"
    return {
        "diagnostics": directory_inventory(diagnostics),
        "spacing_candidate_review": directory_inventory(review, pattern="*.png"),
        "qc_gallery_index": file_inventory(gallery),
        "previews": directory_inventory(previews, pattern="*.png"),
        "validation": directory_inventory(validation),
    }


def directory_inventory(path: Path, pattern: str = "*") -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "file_count": 0, "total_size_bytes": 0}
    files = [item for item in path.glob(pattern) if item.is_file()]
    return {
        "path": str(path),
        "exists": True,
        "file_count": int(len(files)),
        "total_size_bytes": int(sum(item.stat().st_size for item in files)),
    }


def repository_state(project_root: Path, config_path: Path) -> dict[str, Any]:
    return {
        "project_path": str(project_root),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "package_version": package_version(),
        "git": git_state(project_root),
        "config_path": str(config_path),
        "config_hash_sha256": sha256_file(config_path) if config_path.exists() else None,
    }


def package_version() -> str | None:
    try:
        return importlib.metadata.version("sarcomere-analysis")
    except importlib.metadata.PackageNotFoundError:
        return None


def git_state(project_root: Path) -> dict[str, Any]:
    if not (project_root / ".git").exists():
        return {"is_git_repo": False, "commit": None, "dirty": None, "status": None}
    commit = run_git(project_root, ["git", "rev-parse", "HEAD"])
    status = run_git(project_root, ["git", "status", "--short"])
    return {
        "is_git_repo": True,
        "commit": commit.strip() if commit is not None else None,
        "dirty": bool(status.strip()) if status is not None else None,
        "status": status,
    }


def run_git(project_root: Path, command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, cwd=project_root, text=True, capture_output=True, check=False)
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def run_safety_checks(cfg: dict[str, Any], core_paths: dict[str, Path], strict: bool = True) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    missing_required = [key for key in CORE_OUTPUT_KEYS if not core_paths[key].exists()]
    checks["missing_required_core_outputs"] = missing_required

    manifest = read_csv(core_paths["manifest"])
    donor_metadata = read_csv(core_paths["donor_metadata"])
    analysis_image = read_csv(core_paths["analysis_per_image"])
    analysis_donor = read_csv(core_paths["analysis_per_donor"])
    feature_summary = read_json(output_dir(cfg) / "tables" / "feature_assembly_summary.json")
    analysis_summary = read_json(output_dir(cfg) / "tables" / "analysis_table_summary.json")

    checks["analysis_per_image_matches_manifest_rows"] = len(analysis_image) == len(manifest) if not manifest.empty else False
    checks["analysis_per_donor_matches_donor_metadata_rows"] = (
        len(analysis_donor) == len(donor_metadata) if not donor_metadata.empty else False
    )
    checks["donor_id_string_preserved"] = donor_id_string_preserved([manifest, donor_metadata, analysis_image, analysis_donor])
    checks["spacing_status"] = spacing_status_from_summaries(feature_summary, analysis_summary)
    checks["spacing_status_present"] = bool(checks["spacing_status"])
    checks["passed"] = (
        not missing_required
        and checks["analysis_per_image_matches_manifest_rows"]
        and checks["analysis_per_donor_matches_donor_metadata_rows"]
        and checks["donor_id_string_preserved"]
        and checks["spacing_status_present"]
    )
    if strict and not checks["passed"]:
        raise ValueError(f"Project audit safety checks failed: {checks}")
    return checks


def donor_id_string_preserved(tables: list[pd.DataFrame]) -> bool:
    for table in tables:
        if table.empty or "donor_id" not in table.columns:
            return False
        values = table["donor_id"].dropna()
        if values.empty:
            return False
        if not values.map(lambda value: isinstance(value, str)).all():
            return False
        if values.astype(str).str.contains(r"\.0$", regex=True).any():
            return False
    return True


def spacing_status_from_summaries(feature_summary: dict[str, Any], analysis_summary: dict[str, Any]) -> str | None:
    return analysis_summary.get("spacing_global_status") or feature_summary.get("spacing_global_status")


def scientific_decision_summary() -> dict[str, Any]:
    return {
        "primary_feature_family": "OOP/orientation",
        "spacing_status": "exploratory_low_yield",
        "spacing_policy": "Spacing is preserved but should not be used as a primary endpoint unless future validation changes this.",
        "biological_claims_made": False,
        "independent_biological_unit": "donor_id",
        "healthy_vs_diseased_status": "exploratory_grouping_only; healthy_donor_count=4",
        "manual_fiji_validation_data_ingested": False,
    }


def pipeline_stage_summary() -> dict[str, list[str]]:
    return {
        "completed": [
            "scaffold/config/calibration",
            "IO/manifest",
            "preprocessing",
            "tissue masking/QC/patch grid",
            "orientation/OOP",
            "spacing scaffold + diagnostics",
            "batch metrics",
            "feature assembly",
            "metadata/enriched manifest",
            "analysis-ready tables",
            "validation intake scaffold",
            "pipeline orchestration",
        ],
        "not_yet_implemented": [
            "real FIJI validation statistics",
            "Bland-Altman/correlation",
            "benchmark tools",
            "synthetic degradation",
            "clinical/mixed-model stats",
            "publication figures",
            "JOSS packaging",
            "cell segmentation/ML",
        ],
    }


def reproducible_commands() -> dict[str, str]:
    return {
        "full_tests": "../sarcgraph-env/bin/python -m pytest",
        "dry_run_pipeline": "../sarcgraph-env/bin/python scripts/run_classical_pipeline.py --config configs/default.yaml --dry-run",
        "skip_existing_pipeline": "../sarcgraph-env/bin/python scripts/run_classical_pipeline.py --config configs/default.yaml --skip-existing",
        "full_table_pipeline": "../sarcgraph-env/bin/python scripts/run_classical_pipeline.py --config configs/default.yaml",
        "optional_previews": "../sarcgraph-env/bin/python scripts/run_classical_pipeline.py --config configs/default.yaml --with-previews",
        "optional_validation_template": "../sarcgraph-env/bin/python scripts/run_classical_pipeline.py --config configs/default.yaml --with-validation-template",
        "feature_assembly": "../sarcgraph-env/bin/python scripts/assemble_features.py --config configs/default.yaml",
        "analysis_table_build": "../sarcgraph-env/bin/python scripts/build_analysis_tables.py --config configs/default.yaml",
    }


def write_project_audit_outputs(
    audit: dict[str, Any],
    results_root: str | Path,
    docs_root: str | Path,
) -> dict[str, Path]:
    results = Path(results_root)
    docs = Path(docs_root)
    results.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": results / "project_audit_summary.json",
        "summary_txt": results / "project_audit_summary.txt",
        "handoff_md": docs / "PROJECT_STATUS_HANDOFF.md",
        "inventory_md": docs / "OUTPUT_INVENTORY.md",
    }
    paths["summary_json"].write_text(json.dumps(json_safe(audit), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(project_audit_text(audit), encoding="utf-8")
    paths["handoff_md"].write_text(project_handoff_markdown(audit), encoding="utf-8")
    paths["inventory_md"].write_text(output_inventory_markdown(audit), encoding="utf-8")
    return paths


def project_audit_text(audit: dict[str, Any]) -> str:
    checks = audit["safety_checks"]
    rows = {key: item.get("row_count") for key, item in audit["core_output_inventory"].items()}
    return "\n".join(
        [
            "Project Audit Summary",
            f"generated_at: {audit['generated_at']}",
            f"project_path: {audit['repository_state']['project_path']}",
            f"test_status: {audit['test_status']['provided_status']}",
            f"row_counts: {rows}",
            f"spacing_status: {checks.get('spacing_status')}",
            f"safety_checks_passed: {checks.get('passed')}",
        ]
    ) + "\n"


def project_handoff_markdown(audit: dict[str, Any]) -> str:
    decisions = audit["scientific_decision_summary"]
    stages = audit["pipeline_stage_summary"]
    commands = audit["reproducible_commands"]
    checks = audit["safety_checks"]
    rows = {key: item.get("row_count") for key, item in audit["core_output_inventory"].items()}
    completed = "\n".join(f"- {item}" for item in stages["completed"])
    missing = "\n".join(f"- {item}" for item in stages["not_yet_implemented"])
    command_lines = "\n".join(f"- `{name}`: `{command}`" for name, command in commands.items())
    return f"""# Project Status Handoff

Generated: `{audit['generated_at']}`

Project path: `{audit['repository_state']['project_path']}`

Test status recorded: `{audit['test_status']['provided_status']}`

## Current Scientific Decisions

- Primary feature family: `{decisions['primary_feature_family']}`
- Spacing status: `{decisions['spacing_status']}`
- Spacing policy: {decisions['spacing_policy']}
- Biological claims made: `{decisions['biological_claims_made']}`
- Independent biological unit: `{decisions['independent_biological_unit']}`
- Healthy-vs-diseased status: {decisions['healthy_vs_diseased_status']}
- Real FIJI/manual validation data ingested: `{decisions['manual_fiji_validation_data_ingested']}`

## Row Counts

```json
{json.dumps(rows, indent=2)}
```

## Safety Checks

```json
{json.dumps(checks, indent=2)}
```

## Completed Modules

{completed}

## Not Yet Implemented

{missing}

## Reproducible Commands

{command_lines}

## Boundary

This handoff records project state only. It does not add algorithms, statistics, validation analysis, figures, benchmark tools, clinical models, ML, segmentation, or spacing changes.
"""


def output_inventory_markdown(audit: dict[str, Any]) -> str:
    lines = ["# Output Inventory", "", "## Core Outputs", ""]
    lines.append("| Key | Exists | Rows | Size bytes | Modified | Path |")
    lines.append("|---|---:|---:|---:|---|---|")
    for key, item in audit["core_output_inventory"].items():
        lines.append(
            f"| `{key}` | {item['exists']} | {item.get('row_count')} | {item.get('size_bytes')} | "
            f"{item.get('modified_time')} | `{item['path']}` |"
        )
    lines.extend(["", "## Optional / Diagnostic Outputs", ""])
    lines.append("| Key | Exists | File count | Size bytes | Path |")
    lines.append("|---|---:|---:|---:|---|")
    for key, item in audit["optional_output_inventory"].items():
        lines.append(
            f"| `{key}` | {item['exists']} | {item.get('file_count')} | {item.get('total_size_bytes', item.get('size_bytes'))} | `{item['path']}` |"
        )
    lines.append("")
    lines.append("Raw TIFFs are not inventoried or copied by this audit.")
    lines.append("")
    return "\n".join(lines)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"image_id": str, "donor_id": str, "region_id": str, "patch_id": str})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
