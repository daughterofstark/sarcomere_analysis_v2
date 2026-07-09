from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from sarcomere_analysis.pipeline import (
    build_classical_pipeline_plan,
    collect_pipeline_row_counts,
    run_classical_pipeline,
    write_pipeline_run_summary,
)


def orchestration_config(tmp_path: Path) -> dict:
    return {
        "paths": {"raw_tiff_dir": str(tmp_path / "raw"), "output_dir": str(tmp_path / "results")},
        "outputs": {"manifest_csv": str(tmp_path / "results" / "tables" / "manifest.csv")},
        "calibration": {
            "pixel_size_um": 0.1299,
            "expected_sarcomere_spacing_um": {"min": 1.5, "max": 2.4},
        },
        "filename_pattern": {"regex": r"^(?P<donor_id>\d+\.\d+)-(?P<region_id>\d+)$"},
        "run": {"include_extensions": [".tif", ".tiff"], "recursive": False},
    }


def successful_runner(commands: list[list[str]]) -> callable:
    def runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
        _ = cwd
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    return runner


def failing_runner(fail_step_script: str, commands: list[list[str]]) -> callable:
    def runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
        _ = cwd
        commands.append(command)
        if fail_step_script in {Path(part).name for part in command}:
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="boom")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    return runner


def test_dry_run_lists_expected_steps_in_order(tmp_path: Path) -> None:
    cfg = orchestration_config(tmp_path)
    summary, _ = run_classical_pipeline(cfg, "config.yaml", tmp_path, dry_run=True)
    names = [step["name"] for step in summary["steps"]]
    assert names == [
        "build_manifest",
        "run_batch_metrics",
        "audit_batch_outputs",
        "assemble_feature_tables",
        "enrich_manifest",
        "build_analysis_tables",
    ]
    assert all(step["status"] == "would_run" for step in summary["steps"])


def test_default_plan_excludes_previews_and_spacing_diagnostics(tmp_path: Path) -> None:
    plan = build_classical_pipeline_plan(orchestration_config(tmp_path), "config.yaml", tmp_path, python_executable="python")
    names = [step.name for step in plan]
    assert "generate_qc_previews" not in names
    assert "diagnose_spacing" not in names


def test_flags_include_optional_steps(tmp_path: Path) -> None:
    plan = build_classical_pipeline_plan(
        orchestration_config(tmp_path),
        "config.yaml",
        tmp_path,
        python_executable="python",
        with_previews=True,
        with_spacing_diagnostics=True,
        with_validation_template=True,
    )
    names = [step.name for step in plan]
    assert "generate_qc_previews" in names
    assert "diagnose_spacing_candidates" in names
    assert "prepare_validation_template" in names


def test_failed_step_stops_pipeline_by_default(tmp_path: Path) -> None:
    cfg = orchestration_config(tmp_path)
    commands: list[list[str]] = []
    summary, _ = run_classical_pipeline(
        cfg,
        "config.yaml",
        tmp_path,
        command_runner=failing_runner("run_batch_metrics.py", commands),
    )
    assert [step["name"] for step in summary["steps"]] == ["build_manifest", "run_batch_metrics"]
    assert summary["steps"][-1]["status"] == "error"


def test_continue_on_error_records_failure_and_continues(tmp_path: Path) -> None:
    cfg = orchestration_config(tmp_path)
    commands: list[list[str]] = []
    summary, _ = run_classical_pipeline(
        cfg,
        "config.yaml",
        tmp_path,
        continue_on_error=True,
        command_runner=failing_runner("run_batch_metrics.py", commands),
    )
    assert len(summary["steps"]) == 6
    assert summary["steps"][1]["status"] == "error"
    assert summary["steps"][-1]["name"] == "build_analysis_tables"


def test_run_summary_json_is_serializable(tmp_path: Path) -> None:
    cfg = orchestration_config(tmp_path)
    commands: list[list[str]] = []
    summary, _ = run_classical_pipeline(cfg, "config.yaml", tmp_path, command_runner=successful_runner(commands))
    paths = write_pipeline_run_summary(summary, cfg)
    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded["steps"][0]["name"] == "build_manifest"


def test_output_row_count_collector_handles_missing_files_safely(tmp_path: Path) -> None:
    counts = collect_pipeline_row_counts(orchestration_config(tmp_path))
    assert counts["manifest_rows"] is None
    assert counts["analysis_per_donor_rows"] is None


def test_no_raw_tiff_copy_operation_is_planned(tmp_path: Path) -> None:
    plan = build_classical_pipeline_plan(
        orchestration_config(tmp_path),
        "config.yaml",
        tmp_path,
        python_executable="python",
        with_previews=True,
        with_spacing_diagnostics=True,
        with_validation_template=True,
    )
    forbidden = {"cp", "copy", "rsync"}
    for step in plan:
        assert not forbidden.intersection(Path(part).name for part in step.command)


def test_pipeline_plan_order_is_stable(tmp_path: Path) -> None:
    plan = build_classical_pipeline_plan(orchestration_config(tmp_path), "config.yaml", tmp_path, python_executable="python")
    assert [step.name for step in plan[:3]] == ["build_manifest", "run_batch_metrics", "audit_batch_outputs"]
    assert [step.name for step in plan[-2:]] == ["enrich_manifest", "build_analysis_tables"]


def test_skip_existing_marks_steps_skipped(tmp_path: Path) -> None:
    cfg = orchestration_config(tmp_path)
    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    for name in [
        "manifest.csv",
        "per_patch_metrics.csv",
        "per_image_metrics.csv",
        "batch_run_summary.csv",
        "batch_audit_summary.json",
        "batch_audit_summary.txt",
        "features_per_patch.csv",
        "features_per_image.csv",
        "features_per_donor.csv",
        "feature_assembly_summary.json",
        "enriched_manifest.csv",
        "donor_metadata.csv",
        "metadata_join_summary.json",
        "analysis_per_image.csv",
        "analysis_per_donor.csv",
        "analysis_table_summary.json",
    ]:
        (tables / name).write_text("header\n", encoding="utf-8")
    summary, _ = run_classical_pipeline(cfg, "config.yaml", tmp_path, skip_existing=True)
    assert all(step["status"] == "skipped_existing" for step in summary["steps"])


def test_cli_help_works() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_classical_pipeline.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--with-previews" in result.stdout


def test_row_count_collector_counts_existing_csvs(tmp_path: Path) -> None:
    cfg = orchestration_config(tmp_path)
    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame([{"image_id": "x"}, {"image_id": "y"}]).to_csv(tables / "analysis_per_image.csv", index=False)
    counts = collect_pipeline_row_counts(cfg)
    assert counts["analysis_per_image_rows"] == 2
