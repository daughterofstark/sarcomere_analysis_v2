from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from sarcomere_analysis.schemas import PATCH_METRICS_COLUMNS
from sarcomere_analysis.spacing.diagnostics import (
    SPACING_DIAGNOSTIC_COLUMNS,
    summarize_spacing_diagnostics,
)
from scripts.diagnose_spacing import run_spacing_diagnostics
from test_step_8_cli_batch import script_path, write_step8_config, write_synthetic_tiffs


def read_config(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def synthetic_diagnostics() -> pd.DataFrame:
    rows = [
        diagnostic_row("2.001-1", "2.001", True, True, "accepted", "ok", 10.0, 1.0, 0.30),
        diagnostic_row("2.001-1", "2.001", True, False, "confidence", "low_periodicity_confidence", 16.0, 1.6, 0.05),
        diagnostic_row("3.110-1", "3.110", False, False, "failed_patch_qc", "low_tissue_fraction;failed_patch_qc", np.nan, np.nan, 0.0),
        diagnostic_row("3.110-1", "3.110", True, False, "missing_orientation", "missing_orientation", np.nan, np.nan, 0.0),
    ]
    return pd.DataFrame(rows, columns=SPACING_DIAGNOSTIC_COLUMNS)


def diagnostic_row(
    image_id: str,
    donor_id: str,
    valid_for_spacing: bool,
    valid_final: bool,
    stage: str,
    reason: str,
    lag_px: float,
    lag_um: float,
    confidence: float,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "donor_id": donor_id,
        "patch_id": f"{image_id}_p",
        "y0": 0,
        "x0": 0,
        "y1": 32,
        "x1": 32,
        "valid_for_spacing": valid_for_spacing,
        "valid_for_spacing_final": valid_final,
        "patch_spacing_um": lag_um if valid_final else np.nan,
        "patch_spacing_px": lag_px if valid_final else np.nan,
        "patch_spacing_confidence": confidence,
        "expected_spacing_min_px": 10.0,
        "expected_spacing_max_px": 16.0,
        "selected_lag_px": lag_px,
        "selected_lag_um": lag_um,
        "peak_score": confidence + 0.5 if np.isfinite(confidence) else np.nan,
        "peak_rank_or_index": 0,
        "autocorr_peak_value": confidence + 0.5 if np.isfinite(confidence) else np.nan,
        "autocorr_baseline_value": 0.5,
        "confidence_threshold": 0.15,
        "spacing_rejection_stage": stage,
        "spacing_invalid_reason": reason,
    }


def test_diagnostic_summary_handles_all_nan_spacing_safely() -> None:
    diagnostics = synthetic_diagnostics()
    diagnostics["valid_for_spacing_final"] = False
    diagnostics["patch_spacing_um"] = np.nan
    diagnostics["patch_spacing_px"] = np.nan
    summary, by_image = summarize_spacing_diagnostics(diagnostics)

    assert int(summary.loc[0, "accepted_spacing_count"]) == 0
    assert np.isnan(summary.loc[0, "accepted_selected_lag_um_median"])
    assert len(by_image) == 2


def test_accepted_spacing_near_lower_bound_is_counted() -> None:
    summary, _ = summarize_spacing_diagnostics(synthetic_diagnostics())

    assert int(summary.loc[0, "accepted_spacing_count"]) == 1
    assert int(summary.loc[0, "accepted_near_lower_bound_count"]) == 1
    assert float(summary.loc[0, "accepted_near_lower_bound_fraction"]) == 1.0


def test_missing_orientation_and_failed_patch_qc_are_counted_separately() -> None:
    summary, _ = summarize_spacing_diagnostics(synthetic_diagnostics())

    assert int(summary.loc[0, "missing_orientation_count"]) == 1
    assert int(summary.loc[0, "failed_patch_qc_count"]) == 1
    assert int(summary.loc[0, "low_periodicity_confidence_count"]) == 1


def test_main_patch_metric_schema_is_not_extended_with_diagnostics() -> None:
    diagnostic_only = {
        "expected_spacing_min_px",
        "selected_lag_px",
        "spacing_rejection_stage",
        "autocorr_baseline_value",
    }
    assert diagnostic_only.isdisjoint(set(PATCH_METRICS_COLUMNS))


def test_diagnostic_outputs_preserve_string_ids_and_paths(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=1)
    config_path = write_step8_config(tmp_path, raw_dir)
    cfg = read_config(config_path)
    diagnostics, paths, _ = run_spacing_diagnostics(
        cfg,
        limit=1,
        output_dir_override=str(tmp_path / "results" / "diagnostics"),
        write_patch_diagnostics=True,
        write_summary=True,
    )

    assert diagnostics.loc[0, "image_id"] == "2.001-1"
    assert diagnostics.loc[0, "donor_id"] == "2.001"
    assert paths["summary"].is_relative_to(tmp_path / "results" / "diagnostics")
    assert paths["by_image"].is_relative_to(tmp_path / "results" / "diagnostics")
    assert (tmp_path / "results" / "diagnostics" / "2.001-1_spacing_patch_diagnostics.csv").exists()


def test_diagnostic_script_works_on_synthetic_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=1)
    config_path = write_step8_config(tmp_path, raw_dir)
    output_dir = tmp_path / "results" / "diagnostics"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path("diagnose_spacing.py")),
            "--config",
            str(config_path),
            "--limit",
            "1",
            "--output-dir",
            str(output_dir),
            "--write-summary",
            "--write-patch-diagnostics",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "diagnostic_patch_rows:" in completed.stdout
    assert (output_dir / "spacing_diagnostic_summary.csv").exists()
    assert (output_dir / "spacing_diagnostic_by_image.csv").exists()


def test_autocorrelation_debug_plot_is_written(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=1)
    config_path = write_step8_config(tmp_path, raw_dir)
    output_dir = tmp_path / "results" / "diagnostics"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path("diagnose_spacing.py")),
            "--config",
            str(config_path),
            "--limit",
            "1",
            "--output-dir",
            str(output_dir),
            "--write-summary",
            "--debug-image-id",
            "2.001-1",
            "--debug-patch-id",
            "2.001-1_p00000",
            "--write-autocorr-debug",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote autocorr_debug:" in completed.stdout
    assert (output_dir / "2.001-1_patch_2.001-1_p00000_autocorr_debug.png").exists()
