from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sarcomere_analysis.confocal_analysis_dataset import (
    ANALYSIS_COLUMNS,
    REVIEW_TEMPLATE_COLUMNS,
    build_analysis_per_image,
    build_confocal_analysis_dataset,
    build_manual_review_template,
)


def analysis_config(tmp_path: Path) -> dict:
    return {"paths": {"output_dir": str(tmp_path / "results")}}


def endpoint_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            endpoint_row("spacing", "spacing.tif", True, True, False, "spacing_eligible_moderate", 0.3, 12, 0.12, 2.0),
            endpoint_row(
                "lowyield", "lowyield.tif", True, False, True, "spacing_eligible_low_confidence", 0.4, 8, 0.08, 2.1
            ),
            endpoint_row("ooponly", "ooponly.tif", True, False, False, "oop_only_spacing_low_yield", 0.5, 2, 0.02, 1.9),
        ]
    )


def endpoint_row(
    image_id: str,
    filename: str,
    oop_reportable: bool,
    spacing_reportable: bool,
    review_needed: bool,
    endpoint_class: str,
    candidate_fraction: float,
    valid_spacing: int,
    spacing_fraction: float,
    spacing_median: float,
) -> dict:
    return {
        "confocal_image_id": image_id,
        "filename": filename,
        "calibrated": True,
        "total_patches": 100,
        "selected_candidate_patches": int(candidate_fraction * 100),
        "selected_candidate_fraction": candidate_fraction,
        "selected_median_oop": 0.7,
        "selected_vs_all_oop_difference": 0.1,
        "selected_spacing_valid_fraction": spacing_fraction,
        "valid_selected_spacing_patches": valid_spacing,
        "selected_spacing_median_um": spacing_median,
        "selected_spacing_iqr_um": 0.2,
        "endpoint_class": endpoint_class,
        "spacing_reportable": spacing_reportable,
        "oop_reportable": oop_reportable,
        "review_needed": review_needed,
        "reason": endpoint_class,
    }


def pipeline_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "confocal_image_id": "spacing",
                "filename": "spacing.tif",
                "pixel_size_x_um": 0.08,
                "pixel_size_y_um": 0.08,
                "pixel_size_available": True,
                "all_region_median_oop": 0.5,
                "selected_median_coherence": 0.8,
                "selected_spacing_range_um": "1.6-2.3",
                "interpretation_flag": "ok",
            },
            {
                "confocal_image_id": "lowyield",
                "filename": "lowyield.tif",
                "pixel_size_x_um": 0.08,
                "pixel_size_y_um": 0.08,
                "pixel_size_available": True,
                "all_region_median_oop": 0.4,
                "selected_median_coherence": 0.7,
                "selected_spacing_range_um": "1.7-2.2",
                "interpretation_flag": "low",
            },
            {
                "confocal_image_id": "ooponly",
                "filename": "ooponly.tif",
                "pixel_size_x_um": 0.08,
                "pixel_size_y_um": 0.08,
                "pixel_size_available": True,
                "all_region_median_oop": 0.3,
                "selected_median_coherence": 0.6,
                "selected_spacing_range_um": "1.8-2.1",
                "interpretation_flag": "oop",
            },
        ]
    )


def triage_table() -> pd.DataFrame:
    return pipeline_table()[["confocal_image_id", "filename", "selected_median_coherence", "selected_spacing_range_um"]].copy()


def freeze_summary() -> dict:
    return {
        "final_frozen_interpretation": ["Confocal OOP/coherence is broadly reportable.", "Confocal spacing is not universal."],
        "manual_visual_spot_check": {
            "verdicts": [
                {
                    "filename": "spacing.tif",
                    "review_status": "reviewed_in_chat",
                    "verdict": "pass",
                    "caveat": "synthetic caveat",
                }
            ]
        },
    }


def review_index() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"confocal_image_id": "spacing", "filename": "spacing.tif", "review_group": "spacing_reportable"},
            {"confocal_image_id": "lowyield", "filename": "lowyield.tif", "review_group": "oop_only_examples"},
        ]
    )


def write_inputs(root: Path, include_review_index: bool = True) -> tuple[Path, Path, Path, Path, Path]:
    endpoint = root / "endpoint"
    pipeline = root / "pipeline"
    audit = root / "audit"
    freeze = root / "freeze"
    review = root / "review"
    for path in [endpoint, pipeline, audit, freeze, review]:
        path.mkdir(parents=True, exist_ok=True)
    endpoint_table().to_csv(endpoint / "confocal_endpoint_per_image.csv", index=False)
    pipeline_table().to_csv(pipeline / "confocal_pipeline_per_image.csv", index=False)
    triage_table().to_csv(audit / "confocal_larger_image_triage.csv", index=False)
    (freeze / "confocal_freeze_summary.json").write_text(json.dumps(freeze_summary()), encoding="utf-8")
    if include_review_index:
        review_index().to_csv(review / "confocal_endpoint_review_index.csv", index=False)
    return endpoint, pipeline, audit, freeze, review


def test_output_contains_one_row_per_image() -> None:
    table = build_analysis_per_image(endpoint_table(), pipeline_table(), triage_table(), freeze_summary(), review_index())

    assert len(table) == 3
    assert list(table.columns) == ANALYSIS_COLUMNS


def test_spacing_allowed_flag_matches_spacing_reportable() -> None:
    table = build_analysis_per_image(endpoint_table(), pipeline_table(), triage_table(), freeze_summary(), review_index())

    assert table["spacing_value_allowed_for_downstream"].tolist() == table["spacing_reportable"].tolist()


def test_oop_allowed_flag_matches_oop_reportable() -> None:
    table = build_analysis_per_image(endpoint_table(), pipeline_table(), triage_table(), freeze_summary(), review_index())

    assert table["oop_value_allowed_for_downstream"].tolist() == table["oop_reportable"].tolist()


def test_non_reportable_spacing_values_are_retained_but_warned() -> None:
    table = build_analysis_per_image(endpoint_table(), pipeline_table(), triage_table(), freeze_summary(), review_index())
    lowyield = table.loc[table["filename"] == "lowyield.tif"].iloc[0]

    assert lowyield["selected_spacing_median_um"] == 2.1
    assert lowyield["spacing_downstream_warning"] == "not_reportable_endpoint_low_yield"


def test_review_template_includes_all_spacing_reportable_images() -> None:
    table = build_analysis_per_image(endpoint_table(), pipeline_table(), triage_table(), freeze_summary(), review_index())
    template = build_manual_review_template(table, review_index())

    assert "spacing.tif" in template["filename"].tolist()


def test_review_template_allowed_value_columns_exist() -> None:
    table = build_analysis_per_image(endpoint_table(), pipeline_table(), triage_table(), freeze_summary(), review_index())
    template = build_manual_review_template(table, review_index())

    assert list(template.columns) == REVIEW_TEMPLATE_COLUMNS
    for column in [
        "selected_regions_valid",
        "valid_spacing_patches_valid",
        "image_suitable_for_spacing",
        "image_suitable_for_oop",
        "reviewer_confidence",
    ]:
        assert column in template.columns


def test_summary_json_serializable(tmp_path: Path) -> None:
    endpoint, pipeline, audit, freeze, review = write_inputs(tmp_path)

    _, _, summary, paths = build_confocal_analysis_dataset(
        analysis_config(tmp_path),
        endpoint_dir=endpoint,
        pipeline_dir=pipeline,
        audit_dir=audit,
        freeze_dir=freeze,
        review_pack_dir=review,
        output_directory=tmp_path / "out",
        docs_directory=tmp_path / "docs",
    )

    json.dumps(summary)
    assert paths["summary_json"].exists()
    assert paths["markdown"].exists()


def test_missing_optional_review_index_handled_gracefully(tmp_path: Path) -> None:
    endpoint, pipeline, audit, freeze, review = write_inputs(tmp_path, include_review_index=False)

    _, template, summary, _ = build_confocal_analysis_dataset(
        analysis_config(tmp_path),
        endpoint_dir=endpoint,
        pipeline_dir=pipeline,
        audit_dir=audit,
        freeze_dir=freeze,
        review_pack_dir=review,
        output_directory=tmp_path / "out",
        docs_directory=tmp_path / "docs",
    )

    assert summary["review_index_present"] is False
    assert len(template) >= 1


def test_existing_inputs_not_modified(tmp_path: Path) -> None:
    endpoint, pipeline, audit, freeze, review = write_inputs(tmp_path)
    source = endpoint / "confocal_endpoint_per_image.csv"
    before = source.read_bytes()

    build_confocal_analysis_dataset(
        analysis_config(tmp_path),
        endpoint_dir=endpoint,
        pipeline_dir=pipeline,
        audit_dir=audit,
        freeze_dir=freeze,
        review_pack_dir=review,
        output_directory=tmp_path / "out",
        docs_directory=tmp_path / "docs",
    )

    assert source.read_bytes() == before
