from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from PIL import Image

from sarcomere_analysis.confocal_endpoint_review_pack import (
    REVIEW_INDEX_COLUMNS,
    export_confocal_endpoint_review_pack,
    select_endpoint_review_images,
)


def review_config(tmp_path: Path) -> dict:
    return {"paths": {"output_dir": str(tmp_path / "results")}}


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(path)


def endpoint_rows() -> pd.DataFrame:
    rows = [
        row("spacing_a", "spacing_a.tif", "spacing_eligible_moderate", True, True, False, 0.4, 20, 0.2, 2.0, 0.8),
        row("spacing_b", "spacing_b.tif", "spacing_eligible_moderate", True, True, False, 0.3, 12, 0.12, 1.8, 0.7),
        row("low_a", "low_a.tif", "low_candidate_review_needed", False, False, True, 0.04, 0, 0.0, None, 0.5),
        row("low_b", "low_b.tif", "low_candidate_review_needed", False, False, True, 0.03, 1, 0.03, 1.7, 0.4),
        row("oop_low", "oop_low.tif", "oop_only_spacing_low_yield", False, True, False, 0.3, 1, 0.01, 1.9, 0.3),
        row("oop_mid", "oop_mid.tif", "spacing_eligible_low_confidence", False, True, True, 0.3, 8, 0.08, 2.0, 0.6),
        row("oop_high", "oop_high.tif", "spacing_eligible_low_confidence", False, True, True, 0.3, 7, 0.07, 2.1, 0.9),
        row("oop_other", "oop_other.tif", "oop_only_spacing_low_yield", False, True, False, 0.3, 2, 0.02, 2.2, 0.2),
        row("oop_extra", "oop_extra.tif", "spacing_eligible_low_confidence", False, True, True, 0.3, 6, 0.06, 2.3, 0.75),
        row("oop_random", "oop_random.tif", "spacing_eligible_low_confidence", False, True, True, 0.3, 5, 0.05, 2.4, 0.65),
    ]
    return pd.DataFrame(rows)


def row(
    image_id: str,
    filename: str,
    endpoint_class: str,
    spacing_reportable: bool,
    oop_reportable: bool,
    review_needed: bool,
    candidate_fraction: float,
    spacing_count: int,
    spacing_fraction: float,
    spacing_median: float | None,
    oop: float,
) -> dict:
    return {
        "confocal_image_id": image_id,
        "filename": filename,
        "calibrated": True,
        "total_patches": 100,
        "selected_candidate_patches": int(candidate_fraction * 100),
        "selected_candidate_fraction": candidate_fraction,
        "selected_median_oop": oop,
        "selected_vs_all_oop_difference": 0.1,
        "selected_spacing_valid_fraction": spacing_fraction,
        "valid_selected_spacing_patches": spacing_count,
        "selected_spacing_median_um": spacing_median,
        "selected_spacing_iqr_um": 0.2,
        "endpoint_class": endpoint_class,
        "spacing_reportable": spacing_reportable,
        "oop_reportable": oop_reportable,
        "review_needed": review_needed,
        "reason": endpoint_class,
    }


def write_inputs(tmp_path: Path, include_previews: bool = True) -> tuple[Path, Path, Path]:
    root = tmp_path / "results"
    endpoint = root / "confocal_endpoint_report"
    audit = root / "confocal_larger_audit"
    pipeline = root / "confocal_larger_pipeline"
    endpoint.mkdir(parents=True)
    audit.mkdir(parents=True)
    pipeline.mkdir(parents=True)
    endpoint_rows().to_csv(endpoint / "confocal_endpoint_per_image.csv", index=False)
    pd.DataFrame({"confocal_image_id": ["spacing_a"], "filename": ["spacing_a.tif"]}).to_csv(
        audit / "confocal_larger_image_triage.csv", index=False
    )
    pd.DataFrame({"confocal_image_id": ["spacing_a"], "filename": ["spacing_a.tif"]}).to_csv(
        pipeline / "confocal_pipeline_per_image.csv", index=False
    )
    pd.DataFrame({"confocal_image_id": ["spacing_a"], "patch_id": ["p1"]}).to_csv(
        pipeline / "confocal_pipeline_per_patch.csv", index=False
    )
    if include_previews:
        for image_id in ["spacing_a", "spacing_b", "low_a", "low_b", "oop_low", "oop_mid", "oop_high"]:
            write_png(audit / "review_previews" / f"{image_id}_selected_candidate_overlay.png")
            write_png(audit / "review_previews" / f"{image_id}_spacing_um_heatmap.png")
            write_png(pipeline / "previews" / f"spacing_audit_{image_id}_confocal_valid_spacing_overlay.png")
    return endpoint, audit, pipeline


def test_includes_all_spacing_reportable_images() -> None:
    selected = select_endpoint_review_images(endpoint_rows(), n_oop_only_examples=3)

    spacing = selected.loc[selected["review_group"] == "spacing_reportable", "filename"].tolist()
    assert spacing == ["spacing_a.tif", "spacing_b.tif"]


def test_includes_all_low_candidate_review_images() -> None:
    selected = select_endpoint_review_images(endpoint_rows(), n_oop_only_examples=3)

    low = selected.loc[selected["review_group"] == "low_candidate_review", "filename"].tolist()
    assert low == ["low_a.tif", "low_b.tif"]


def test_selects_deterministic_oop_only_examples() -> None:
    first = select_endpoint_review_images(endpoint_rows(), n_oop_only_examples=5, seed=123)
    second = select_endpoint_review_images(endpoint_rows(), n_oop_only_examples=5, seed=123)

    assert first.loc[first["review_group"] == "oop_only_examples", "filename"].tolist() == second.loc[
        second["review_group"] == "oop_only_examples", "filename"
    ].tolist()


def test_summary_index_has_required_columns(tmp_path: Path) -> None:
    endpoint, audit, pipeline = write_inputs(tmp_path)

    index, _, paths = export_confocal_endpoint_review_pack(
        review_config(tmp_path),
        endpoint_dir=endpoint,
        audit_dir=audit,
        pipeline_dir=pipeline,
        output_directory=tmp_path / "out",
    )

    assert list(index.columns) == REVIEW_INDEX_COLUMNS
    assert list(pd.read_csv(paths["index"]).columns) == REVIEW_INDEX_COLUMNS


def test_missing_previews_handled_gracefully(tmp_path: Path) -> None:
    endpoint, audit, pipeline = write_inputs(tmp_path, include_previews=False)

    _, summary, _ = export_confocal_endpoint_review_pack(
        review_config(tmp_path),
        endpoint_dir=endpoint,
        audit_dir=audit,
        pipeline_dir=pipeline,
        output_directory=tmp_path / "out",
    )

    assert summary["review_image_files_copied"] == 0
    assert summary["missing_preview_count"] > 0


def test_zip_excludes_raw_internal_large_tables(tmp_path: Path) -> None:
    endpoint, audit, pipeline = write_inputs(tmp_path)

    _, _, paths = export_confocal_endpoint_review_pack(
        review_config(tmp_path),
        endpoint_dir=endpoint,
        audit_dir=audit,
        pipeline_dir=pipeline,
        output_directory=tmp_path / "out",
        write_zip=True,
    )

    with ZipFile(paths["zip"]) as archive:
        names = archive.namelist()
    assert "confocal_endpoint_review_index.csv" in names
    assert "confocal_endpoint_review_notes.md" in names
    assert "confocal_endpoint_review_pack_summary.txt" in names
    assert any(name.startswith("review_images/") and name.endswith(".png") for name in names)
    assert not any("confocal_pipeline_per_patch" in name for name in names)
    assert not any("confocal_larger_image_triage" in name for name in names)
    assert not any("raw" in name.lower() for name in names)


def test_summary_json_serializable(tmp_path: Path) -> None:
    endpoint, audit, pipeline = write_inputs(tmp_path)

    _, summary, paths = export_confocal_endpoint_review_pack(
        review_config(tmp_path),
        endpoint_dir=endpoint,
        audit_dir=audit,
        pipeline_dir=pipeline,
        output_directory=tmp_path / "out",
    )

    json.dumps(summary)
    assert json.loads(paths["summary_json"].read_text(encoding="utf-8"))["mode"] == "confocal_endpoint_review_pack"


def test_existing_outputs_are_not_modified(tmp_path: Path) -> None:
    endpoint, audit, pipeline = write_inputs(tmp_path)
    source = endpoint / "confocal_endpoint_per_image.csv"
    before = source.read_bytes()

    export_confocal_endpoint_review_pack(
        review_config(tmp_path),
        endpoint_dir=endpoint,
        audit_dir=audit,
        pipeline_dir=pipeline,
        output_directory=tmp_path / "out",
    )

    assert source.read_bytes() == before
