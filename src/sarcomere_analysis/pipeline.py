from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import manifest_csv_path, output_dir
from .io import load_tiff
from .masking import compute_tissue_mask
from .orientation import compute_orientation_analysis
from .outputs import (
    ensure_output_dirs,
    preview_paths,
    write_heatmap,
    write_image_metrics,
    write_mask_overlay,
    write_patch_metrics,
    write_preview_png,
)
from .preprocessing import preprocess_image
from .provenance import collect_run_provenance, write_run_provenance
from .qc import compute_patch_qc
from .spacing import compute_spacing_analysis


@dataclass(frozen=True)
class WriteOptions:
    tables: bool = False
    preview: bool = False
    provenance: bool = False
    overwrite_previews: bool = True


@dataclass(frozen=True)
class RunResult:
    image_id: str
    donor_id: str | None
    image_path: Path
    patch_metrics: pd.DataFrame
    image_metrics: dict[str, object]
    preprocessing_metadata: dict[str, object]
    tissue_mask_metadata: dict[str, object]
    output_paths: dict[str, str]
    runtime_seconds: float

    @property
    def total_patches(self) -> int:
        return int(self.image_metrics["total_patches"])

    @property
    def valid_orientation_patches(self) -> int:
        return int(self.image_metrics["valid_orientation_patches"])

    @property
    def valid_spacing_patches(self) -> int:
        return int(self.image_metrics["n_spacing_valid_patches"])


def run_single_image(
    image_path: str | Path,
    image_id: str,
    donor_id: str | None,
    cfg: dict[str, Any],
    write_options: WriteOptions | None = None,
    config_path: str | Path | None = None,
) -> RunResult:
    started = perf_counter()
    options = write_options or WriteOptions()
    ensure_output_dirs(cfg)

    image_path = Path(image_path)
    raw = load_tiff(image_path)
    preprocessing_result = preprocess_image(raw, cfg)
    mask_result = compute_tissue_mask(preprocessing_result.image, cfg)
    patch_qc = compute_patch_qc(preprocessing_result.image, mask_result.mask, image_id, cfg)
    orientation_result = compute_orientation_analysis(preprocessing_result.image, mask_result.mask, patch_qc, cfg)
    spacing_result = compute_spacing_analysis(preprocessing_result.image, orientation_result.patch_metrics, cfg)

    patch_metrics = spacing_result.patch_metrics.copy()
    if "donor_id" not in patch_metrics.columns:
        patch_metrics.insert(1, "donor_id", donor_id)

    tissue_fraction = float(np.mean(mask_result.mask))
    image_metrics = build_image_metrics(
        image_id,
        donor_id,
        tissue_fraction,
        patch_metrics,
        orientation_result.image_metrics,
        spacing_result.image_metrics,
    )

    output_paths: dict[str, str] = {}
    if options.preview:
        output_paths.update(
            write_standard_previews(
                image_id,
                preprocessing_result.image,
                mask_result.mask,
                orientation_result,
                patch_metrics,
                cfg,
                overwrite=options.overwrite_previews,
            )
        )

    if options.tables:
        output_paths["per_patch_metrics"] = str(write_patch_metrics(patch_metrics, image_id, cfg))
        output_paths["per_image_metrics"] = str(write_image_metrics(image_metrics, image_id, cfg))

    if options.provenance:
        provenance_metadata = {
            "config_path": str(config_path) if config_path is not None else None,
            "input_image_shape": raw.shape,
            "input_image_dtype": str(raw.dtype),
            "preprocessing_metadata": preprocessing_result.metadata,
            "tissue_mask_metadata": mask_result.metadata,
            "counts": {
                "total_patches": int(len(patch_metrics)),
                "valid_orientation_patches": int(patch_metrics["valid_for_orientation"].sum()) if not patch_metrics.empty else 0,
                "valid_spacing_patches": int(patch_metrics["valid_for_spacing_final"].sum()) if not patch_metrics.empty else 0,
            },
            "output_file_paths": dict(output_paths),
        }
        provenance = collect_run_provenance(cfg, image_path, image_id, provenance_metadata)
        provenance_path = write_run_provenance(provenance, image_id, cfg)
        output_paths["run_provenance"] = str(provenance_path)
        provenance["output_file_paths"] = dict(output_paths)
        write_run_provenance(provenance, image_id, cfg)

    runtime_seconds = perf_counter() - started
    return RunResult(
        image_id=image_id,
        donor_id=donor_id,
        image_path=image_path,
        patch_metrics=patch_metrics,
        image_metrics=image_metrics,
        preprocessing_metadata=preprocessing_result.metadata,
        tissue_mask_metadata=mask_result.metadata,
        output_paths=output_paths,
        runtime_seconds=runtime_seconds,
    )


def build_image_metrics(
    image_id: str,
    donor_id: str | None,
    tissue_fraction: float,
    patch_metrics: pd.DataFrame,
    orientation_metrics: dict[str, object],
    spacing_metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "donor_id": donor_id,
        "tissue_fraction": tissue_fraction,
        "total_patches": int(len(patch_metrics)),
        "valid_orientation_patches": int(patch_metrics["valid_for_orientation"].sum()) if not patch_metrics.empty else 0,
        **orientation_metrics,
        **spacing_metrics,
    }


def write_standard_previews(
    image_id: str,
    image: np.ndarray,
    tissue_mask: np.ndarray,
    orientation_result,
    patch_metrics: pd.DataFrame,
    cfg: dict[str, Any],
    overwrite: bool = True,
) -> dict[str, str]:
    paths = preview_paths(image_id, cfg)
    written: dict[str, str] = {}
    if overwrite or not paths["tissue_mask_overlay"].exists():
        written["tissue_mask_overlay"] = str(write_mask_overlay(image, tissue_mask, paths["tissue_mask_overlay"]))
    if overwrite or not paths["orientation"].exists():
        written["orientation"] = str(write_preview_png((orientation_result.orientation_map + np.pi / 2.0) / np.pi, paths["orientation"]))
    if overwrite or not paths["coherence"].exists():
        written["coherence"] = str(write_preview_png(orientation_result.coherence_map, paths["coherence"]))
    if overwrite or not paths["oop_heatmap"].exists():
        written["oop_heatmap"] = str(write_heatmap("patch_oop", patch_metrics, image.shape, paths["oop_heatmap"], cfg))
    spacing_column = "patch_spacing_um" if patch_metrics["patch_spacing_um"].notna().any() else "patch_spacing_confidence"
    if overwrite or not paths["spacing_heatmap"].exists():
        written["spacing_heatmap"] = str(write_heatmap(spacing_column, patch_metrics, image.shape, paths["spacing_heatmap"], cfg))
    return written


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: list[str]
    expected_outputs: list[Path]
    optional: bool = False


@dataclass(frozen=True)
class StepExecution:
    name: str
    status: str
    runtime_seconds: float
    command: list[str]
    expected_outputs: list[str]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""


CommandRunner = Callable[[list[str], Path], subprocess.CompletedProcess]


def build_classical_pipeline_plan(
    cfg: dict[str, Any],
    config_path: str | Path,
    project_root: str | Path,
    python_executable: str | Path | None = None,
    with_previews: bool = False,
    with_spacing_diagnostics: bool = False,
    with_validation_template: bool = False,
    continue_on_error: bool = False,
    force: bool = False,
) -> list[PipelineStep]:
    _ = force
    py = str(python_executable or sys.executable)
    config_arg = str(config_path)
    root = Path(project_root)
    tables = output_dir(cfg) / "tables"
    diagnostics = output_dir(cfg) / "diagnostics"
    validation = output_dir(cfg) / "validation"
    qc_gallery = output_dir(cfg) / "qc_gallery"
    batch_command = [
        py,
        "scripts/run_batch_metrics.py",
        "--config",
        config_arg,
        "--write-tables",
        "--write-provenance",
    ]
    if continue_on_error:
        batch_command.append("--continue-on-error")
    preview_command = [
        py,
        "scripts/generate_qc_previews.py",
        "--config",
        config_arg,
    ]
    if force:
        preview_command.append("--overwrite")
    if continue_on_error:
        preview_command.append("--continue-on-error")
    review_pack_command = [
        py,
        "scripts/export_spacing_review_pack.py",
        "--config",
        config_arg,
        "--max-per-class",
        "10",
    ]
    if force:
        review_pack_command.append("--overwrite")
    steps = [
        PipelineStep(
            "build_manifest",
            [py, "scripts/build_manifest.py", "--config", config_arg],
            [manifest_csv_path(cfg)],
        ),
        PipelineStep(
            "run_batch_metrics",
            batch_command,
            [tables / "per_patch_metrics.csv", tables / "per_image_metrics.csv", tables / "batch_run_summary.csv"],
        ),
        PipelineStep(
            "audit_batch_outputs",
            [py, "scripts/audit_batch_outputs.py", "--config", config_arg],
            [tables / "batch_audit_summary.json", tables / "batch_audit_summary.txt"],
        ),
        PipelineStep(
            "assemble_feature_tables",
            [py, "scripts/assemble_features.py", "--config", config_arg],
            [
                tables / "features_per_patch.csv",
                tables / "features_per_image.csv",
                tables / "features_per_donor.csv",
                tables / "feature_assembly_summary.json",
            ],
        ),
        PipelineStep(
            "enrich_manifest",
            [py, "scripts/enrich_manifest.py", "--config", config_arg],
            [tables / "enriched_manifest.csv", tables / "donor_metadata.csv", tables / "metadata_join_summary.json"],
        ),
        PipelineStep(
            "build_analysis_tables",
            [py, "scripts/build_analysis_tables.py", "--config", config_arg],
            [tables / "analysis_per_image.csv", tables / "analysis_per_donor.csv", tables / "analysis_table_summary.json"],
        ),
    ]
    if with_previews:
        steps.extend(
            [
                PipelineStep(
                    "generate_qc_previews",
                    preview_command,
                    [tables / "qc_preview_generation_summary.csv"],
                    optional=True,
                ),
                PipelineStep(
                    "build_qc_gallery",
                    [py, "scripts/build_qc_gallery.py", "--config", config_arg, "--write-index", "--write-html"],
                    [tables / "qc_gallery_index.csv", qc_gallery / "index.html"],
                    optional=True,
                ),
            ]
        )
    if with_spacing_diagnostics:
        steps.extend(
            [
                PipelineStep(
                    "diagnose_spacing",
                    [py, "scripts/diagnose_spacing.py", "--config", config_arg, "--write-summary"],
                    [diagnostics / "spacing_diagnostic_summary.csv", diagnostics / "spacing_diagnostic_by_image.csv"],
                    optional=True,
                ),
                PipelineStep(
                    "triage_spacing_failures",
                    [py, "scripts/triage_spacing_failures.py", "--config", config_arg],
                    [diagnostics / "spacing_failure_summary.json", diagnostics / "spacing_failure_by_image.csv"],
                    optional=True,
                ),
                PipelineStep(
                    "diagnose_spacing_candidates",
                    [py, "scripts/diagnose_spacing_candidates.py", "--config", config_arg, "--all", "--compare-main-table"],
                    [diagnostics / "spacing_candidates.csv", diagnostics / "spacing_candidates_summary.json"],
                    optional=True,
                ),
                PipelineStep(
                    "report_spacing_sensitivity",
                    [py, "scripts/report_spacing_sensitivity.py", "--config", config_arg],
                    [diagnostics / "spacing_sensitivity_variants.csv", diagnostics / "spacing_sensitivity_summary.json"],
                    optional=True,
                ),
                PipelineStep(
                    "export_spacing_review_pack",
                    review_pack_command,
                    [diagnostics / "spacing_candidate_review" / "review_index.csv"],
                    optional=True,
                ),
            ]
        )
    if with_validation_template:
        steps.append(
            PipelineStep(
                "prepare_validation_template",
                [py, "scripts/prepare_validation_template.py", "--config", config_arg],
                [root / "templates" / "manual_validation_template.csv"],
                optional=True,
            )
        )
    return steps


def run_classical_pipeline(
    cfg: dict[str, Any],
    config_path: str | Path,
    project_root: str | Path,
    with_previews: bool = False,
    with_spacing_diagnostics: bool = False,
    with_validation_template: bool = False,
    continue_on_error: bool = False,
    dry_run: bool = False,
    force: bool = False,
    skip_existing: bool = False,
    command_runner: CommandRunner | None = None,
    python_executable: str | Path | None = None,
) -> tuple[dict[str, Any], list[PipelineStep]]:
    project = Path(project_root)
    steps = build_classical_pipeline_plan(
        cfg,
        config_path,
        project,
        python_executable=python_executable,
        with_previews=with_previews,
        with_spacing_diagnostics=with_spacing_diagnostics,
        with_validation_template=with_validation_template,
        continue_on_error=continue_on_error,
        force=force,
    )
    started_at = utc_now_iso()
    if dry_run:
        summary = {
            "started_at": started_at,
            "finished_at": None,
            "config_path": str(config_path),
            "dry_run": True,
            "steps": [planned_step_summary(step, skip_existing=skip_existing, force=force) for step in steps],
            "row_counts": {},
            "spacing_global_status": None,
        }
        return summary, steps

    runner = command_runner or default_command_runner
    executions: list[StepExecution] = []
    for step in steps:
        if skip_existing and not force and outputs_exist(step.expected_outputs):
            executions.append(
                StepExecution(
                    name=step.name,
                    status="skipped_existing",
                    runtime_seconds=0.0,
                    command=step.command,
                    expected_outputs=[str(path) for path in step.expected_outputs],
                )
            )
            continue
        started = perf_counter()
        try:
            completed = runner(step.command, project)
            runtime = perf_counter() - started
            status = "ok" if completed.returncode == 0 else "error"
            execution = StepExecution(
                name=step.name,
                status=status,
                runtime_seconds=runtime,
                command=step.command,
                expected_outputs=[str(path) for path in step.expected_outputs],
                returncode=int(completed.returncode),
                stdout=completed.stdout if isinstance(completed.stdout, str) else "",
                stderr=completed.stderr if isinstance(completed.stderr, str) else "",
                error_message="" if status == "ok" else (completed.stderr or completed.stdout or f"returncode {completed.returncode}"),
            )
        except Exception as exc:
            execution = StepExecution(
                name=step.name,
                status="error",
                runtime_seconds=perf_counter() - started,
                command=step.command,
                expected_outputs=[str(path) for path in step.expected_outputs],
                error_message=str(exc),
            )
        executions.append(execution)
        if execution.status == "error" and not continue_on_error:
            break

    summary = {
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "config_path": str(config_path),
        "dry_run": False,
        "steps": [step_execution_to_dict(item) for item in executions],
        "row_counts": collect_pipeline_row_counts(cfg),
        "spacing_global_status": collect_spacing_global_status(cfg),
        "key_output_paths": key_pipeline_output_paths(cfg),
        "caution": (
            "Classical pipeline orchestration only. No validation statistics, clinical models, plots, "
            "benchmarking, ML, cell segmentation, or spacing algorithm changes were implemented."
        ),
    }
    write_pipeline_run_summary(summary, cfg)
    return summary, steps


def default_command_runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def planned_step_summary(step: PipelineStep, skip_existing: bool, force: bool) -> dict[str, Any]:
    exists = outputs_exist(step.expected_outputs)
    planned_status = "would_skip_existing" if skip_existing and exists and not force else "would_run"
    return {
        "name": step.name,
        "status": planned_status,
        "optional": step.optional,
        "command": step.command,
        "expected_outputs": [str(path) for path in step.expected_outputs],
    }


def outputs_exist(paths: list[Path]) -> bool:
    return bool(paths) and all(path.exists() for path in paths)


def collect_pipeline_row_counts(cfg: dict[str, Any]) -> dict[str, int | None]:
    tables = output_dir(cfg) / "tables"
    return {
        "manifest_rows": csv_row_count(manifest_csv_path(cfg)),
        "per_patch_rows": csv_row_count(tables / "per_patch_metrics.csv"),
        "per_image_rows": csv_row_count(tables / "per_image_metrics.csv"),
        "per_donor_rows": csv_row_count(tables / "features_per_donor.csv"),
        "analysis_per_image_rows": csv_row_count(tables / "analysis_per_image.csv"),
        "analysis_per_donor_rows": csv_row_count(tables / "analysis_per_donor.csv"),
    }


def collect_spacing_global_status(cfg: dict[str, Any]) -> str | None:
    for path in [
        output_dir(cfg) / "tables" / "analysis_table_summary.json",
        output_dir(cfg) / "tables" / "feature_assembly_summary.json",
    ]:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        status = data.get("spacing_global_status")
        if status:
            return str(status)
    return None


def key_pipeline_output_paths(cfg: dict[str, Any]) -> dict[str, str]:
    tables = output_dir(cfg) / "tables"
    return {
        "manifest": str(manifest_csv_path(cfg)),
        "per_patch_metrics": str(tables / "per_patch_metrics.csv"),
        "per_image_metrics": str(tables / "per_image_metrics.csv"),
        "batch_audit_summary": str(tables / "batch_audit_summary.json"),
        "features_per_image": str(tables / "features_per_image.csv"),
        "features_per_donor": str(tables / "features_per_donor.csv"),
        "enriched_manifest": str(tables / "enriched_manifest.csv"),
        "donor_metadata": str(tables / "donor_metadata.csv"),
        "analysis_per_image": str(tables / "analysis_per_image.csv"),
        "analysis_per_donor": str(tables / "analysis_per_donor.csv"),
        "pipeline_run_summary": str(output_dir(cfg) / "pipeline_run_summary.json"),
    }


def csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            line_count = sum(1 for _ in handle)
    except OSError:
        return None
    return max(line_count - 1, 0)


def write_pipeline_run_summary(summary: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Path]:
    root = output_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "pipeline_run_summary.json"
    txt_path = root / "pipeline_run_summary.txt"
    json_path.write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    txt_path.write_text(pipeline_summary_text(summary), encoding="utf-8")
    return {"summary_json": json_path, "summary_txt": txt_path}


def pipeline_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Classical Pipeline Run Summary",
        f"started_at: {summary.get('started_at')}",
        f"finished_at: {summary.get('finished_at')}",
        f"config_path: {summary.get('config_path')}",
        f"dry_run: {summary.get('dry_run')}",
        f"spacing_global_status: {summary.get('spacing_global_status')}",
        f"row_counts: {summary.get('row_counts')}",
        "steps:",
    ]
    for step in summary.get("steps", []):
        lines.append(
            f"  - {step.get('name')}: {step.get('status')} "
            f"runtime={step.get('runtime_seconds', 0.0):.3f}s"
        )
    lines.append(str(summary.get("caution", "")))
    return "\n".join(lines) + "\n"


def step_execution_to_dict(execution: StepExecution) -> dict[str, Any]:
    return {
        "name": execution.name,
        "status": execution.status,
        "runtime_seconds": execution.runtime_seconds,
        "command": execution.command,
        "expected_outputs": execution.expected_outputs,
        "returncode": execution.returncode,
        "stdout": truncate_text(execution.stdout),
        "stderr": truncate_text(execution.stderr),
        "error_message": truncate_text(execution.error_message),
    }


def truncate_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>"


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
