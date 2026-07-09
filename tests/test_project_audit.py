from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sarcomere_analysis.project_audit import (
    build_project_audit,
    core_output_paths,
    output_inventory_markdown,
    write_project_audit_outputs,
)


def audit_config(tmp_path: Path) -> dict:
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


def write_minimal_outputs(tmp_path: Path, donor_id: str = "2.007", mismatch: bool = False) -> dict:
    cfg = audit_config(tmp_path)
    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    n_image_rows = 2 if not mismatch else 1
    pd.DataFrame(
        [
            {"image_id": "2.007-1", "donor_id": donor_id, "region_id": "1"},
            {"image_id": "2.007-2", "donor_id": donor_id, "region_id": "2"},
        ]
    ).to_csv(tables / "manifest.csv", index=False)
    pd.DataFrame([{"image_id": "2.007-1", "donor_id": donor_id, "patch_id": "p0"}]).to_csv(tables / "per_patch_metrics.csv", index=False)
    pd.DataFrame([{"image_id": "2.007-1", "donor_id": donor_id}]).to_csv(tables / "per_image_metrics.csv", index=False)
    pd.DataFrame([{"image_id": "2.007-1", "donor_id": donor_id, "patch_id": "p0"}]).to_csv(tables / "features_per_patch.csv", index=False)
    pd.DataFrame([{"image_id": f"2.007-{idx+1}", "donor_id": donor_id} for idx in range(2)]).to_csv(tables / "features_per_image.csv", index=False)
    pd.DataFrame([{"donor_id": donor_id}]).to_csv(tables / "features_per_donor.csv", index=False)
    pd.DataFrame([{"image_id": f"2.007-{idx+1}", "donor_id": donor_id} for idx in range(2)]).to_csv(tables / "enriched_manifest.csv", index=False)
    pd.DataFrame([{"donor_id": donor_id, "is_healthy": False, "n_images": 2}]).to_csv(tables / "donor_metadata.csv", index=False)
    pd.DataFrame([{"image_id": f"2.007-{idx+1}", "donor_id": donor_id} for idx in range(n_image_rows)]).to_csv(
        tables / "analysis_per_image.csv",
        index=False,
    )
    pd.DataFrame([{"donor_id": donor_id}]).to_csv(tables / "analysis_per_donor.csv", index=False)
    (tables / "feature_assembly_summary.json").write_text(json.dumps({"spacing_global_status": "exploratory_low_yield"}), encoding="utf-8")
    (tables / "analysis_table_summary.json").write_text(json.dumps({"spacing_global_status": "exploratory_low_yield"}), encoding="utf-8")
    (tmp_path / "results" / "pipeline_run_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config: synthetic\n", encoding="utf-8")
    return {"cfg": cfg, "config_path": config_path}


def test_audit_collects_row_counts_correctly(tmp_path: Path) -> None:
    setup = write_minimal_outputs(tmp_path)
    audit = build_project_audit(setup["cfg"], setup["config_path"], tmp_path, test_status="synthetic passed")
    assert audit["core_output_inventory"]["manifest"]["row_count"] == 2
    assert audit["core_output_inventory"]["analysis_per_image"]["row_count"] == 2


def test_missing_optional_diagnostics_do_not_fail(tmp_path: Path) -> None:
    setup = write_minimal_outputs(tmp_path)
    audit = build_project_audit(setup["cfg"], setup["config_path"], tmp_path)
    assert not audit["optional_output_inventory"]["diagnostics"]["exists"]
    assert audit["safety_checks"]["passed"]


def test_missing_required_core_table_fails_clearly(tmp_path: Path) -> None:
    setup = write_minimal_outputs(tmp_path)
    (tmp_path / "results" / "tables" / "analysis_per_image.csv").unlink()
    with pytest.raises(ValueError, match="missing_required_core_outputs"):
        build_project_audit(setup["cfg"], setup["config_path"], tmp_path)


def test_donor_id_string_preservation_check_works(tmp_path: Path) -> None:
    setup = write_minimal_outputs(tmp_path, donor_id="2.0")
    with pytest.raises(ValueError, match="donor_id_string_preserved"):
        build_project_audit(setup["cfg"], setup["config_path"], tmp_path)


def test_row_count_mismatch_is_detected(tmp_path: Path) -> None:
    setup = write_minimal_outputs(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="analysis_per_image_matches_manifest_rows"):
        build_project_audit(setup["cfg"], setup["config_path"], tmp_path)


def test_json_summary_is_serializable(tmp_path: Path) -> None:
    setup = write_minimal_outputs(tmp_path)
    audit = build_project_audit(setup["cfg"], setup["config_path"], tmp_path)
    json.dumps(audit)


def test_output_inventory_includes_required_paths(tmp_path: Path) -> None:
    setup = write_minimal_outputs(tmp_path)
    audit = build_project_audit(setup["cfg"], setup["config_path"], tmp_path)
    paths = core_output_paths(setup["cfg"])
    assert set(paths) <= set(audit["core_output_inventory"])
    assert audit["core_output_inventory"]["manifest"]["path"].endswith("manifest.csv")


def test_scientific_decision_summary_includes_spacing_low_yield(tmp_path: Path) -> None:
    setup = write_minimal_outputs(tmp_path)
    audit = build_project_audit(setup["cfg"], setup["config_path"], tmp_path)
    assert audit["scientific_decision_summary"]["spacing_status"] == "exploratory_low_yield"
    assert audit["safety_checks"]["spacing_status"] == "exploratory_low_yield"


def test_script_can_write_markdown_handoff(tmp_path: Path) -> None:
    setup = write_minimal_outputs(tmp_path)
    audit = build_project_audit(setup["cfg"], setup["config_path"], tmp_path, docs_dir=tmp_path / "docs")
    paths = write_project_audit_outputs(audit, tmp_path / "results", tmp_path / "docs")
    assert paths["handoff_md"].exists()
    assert paths["inventory_md"].exists()
    assert "Project Status Handoff" in paths["handoff_md"].read_text(encoding="utf-8")


def test_no_raw_tiff_copying_is_attempted(tmp_path: Path) -> None:
    setup = write_minimal_outputs(tmp_path)
    audit = build_project_audit(setup["cfg"], setup["config_path"], tmp_path)
    inventory = output_inventory_markdown(audit)
    assert "Raw TIFFs are not inventoried or copied" in inventory
    assert "cp " not in inventory
