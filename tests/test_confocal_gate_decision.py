from __future__ import annotations

import json
from pathlib import Path

from sarcomere_analysis.confocal_gate_decision import (
    build_confocal_gate_decision,
    render_confocal_gate_decision_markdown,
    render_confocal_gate_decision_text,
    write_confocal_gate_decision,
)


def decision_config(tmp_path: Path) -> dict:
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


def test_summary_json_serializable() -> None:
    decision = build_confocal_gate_decision()

    json.dumps(decision)


def test_decision_text_includes_primary_gate_moderate() -> None:
    text = render_confocal_gate_decision_text(build_confocal_gate_decision())

    assert "primary_gate: moderate" in text


def test_decision_text_states_relaxed_not_adopted_globally() -> None:
    decision = build_confocal_gate_decision()
    text = render_confocal_gate_decision_text(decision)
    markdown = render_confocal_gate_decision_markdown(decision)

    assert "not adopted globally" in text
    assert "not adopted globally" in markdown
    assert "moderate_relaxed_combined should replace moderate globally" in text


def test_decision_text_states_spacing_remains_based_on_moderate() -> None:
    text = render_confocal_gate_decision_text(build_confocal_gate_decision())

    assert "calibrated spacing audit remains based on the moderate gate" in text
    assert "Relaxed-gate spacing results exist" in text


def test_write_confocal_gate_decision_outputs(tmp_path: Path) -> None:
    decision, paths = write_confocal_gate_decision(
        decision_config(tmp_path),
        output_directory=tmp_path / "results" / "confocal_gate_refinement",
        docs_directory=tmp_path / "docs",
    )

    assert paths["json"].exists()
    assert paths["txt"].exists()
    assert paths["markdown"].exists()
    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert loaded["primary_gate"] == "moderate"
    assert decision["secondary_sensitivity_gate"] == "moderate_relaxed_combined"


def test_writing_decision_does_not_modify_existing_outputs(tmp_path: Path) -> None:
    source = tmp_path / "results" / "confocal_gate_refinement" / "confocal_gate_refinement_per_image.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("filename,variant_name\n5138.tif,moderate_reference\n", encoding="utf-8")
    before = source.read_bytes()

    write_confocal_gate_decision(
        decision_config(tmp_path),
        output_directory=tmp_path / "results" / "confocal_gate_refinement",
        docs_directory=tmp_path / "docs",
    )

    assert source.read_bytes() == before
