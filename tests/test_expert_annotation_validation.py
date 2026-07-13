from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sarcomere_analysis.expert_annotation_validation import (
    normalize_expert_annotations,
    validate_expert_annotations,
)


def validation_config(tmp_path: Path) -> dict:
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


def synthetic_annotations(rows: int = 12) -> pd.DataFrame:
    values = []
    for index in range(rows):
        score = (index % 5) + 1
        values.append(
            {
                "annotation_id": f"EXPERT_{index + 1:04d}",
                "patch_filename": f"EXPERT_{index + 1:04d}.png",
                "striations_visible": ["no", "unclear", "yes"][index % 3],
                "organisation_score": score,
                "dominant_orientation_deg*": 10 * index,
                "confidence_score": [2, 3, 4, 5][index % 4],
                "spacing_measurable": ["no", "unclear", "yes"][index % 3],
                "manual_sarcomere_length_um_optional": np.nan,
                "notes": "",
                "Unnamed: 9": np.nan,
                "*dominant_orientation_deg: myofibril clockwise (+ve) or anticlockwise (-ve) from horizontal axis": np.nan,
            }
        )
    return pd.DataFrame(values)


def synthetic_internal_key(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "annotation_id": f"EXPERT_{index + 1:04d}",
                "patch_filename": f"EXPERT_{index + 1:04d}.png",
                "image_id": f"2.007-{(index % 3) + 1}",
                "donor_id": "2.007",
                "patch_id": f"2.007-1_p{index:05d}",
                "oop_bin": ["low", "medium", "high"][index % 3],
                "automated_patch_oop": index / max(rows - 1, 1),
                "automated_patch_orientation_deg": 15.0,
                "health_status": "unknown",
            }
            for index in range(rows)
        ]
    )


def write_inputs(tmp_path: Path, annotations: pd.DataFrame | None = None, key: pd.DataFrame | None = None) -> tuple[Path, Path, Path]:
    pack_dir = tmp_path / "results" / "expert_annotation_pack"
    tables_dir = tmp_path / "results" / "tables"
    pack_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)
    annotation_path = pack_dir / "expert_annotation_template_NG.csv"
    key_path = pack_dir / "internal_blinding_key.csv"
    patch_path = tables_dir / "features_per_patch.csv"
    (annotations if annotations is not None else synthetic_annotations()).to_csv(annotation_path, index=False)
    (key if key is not None else synthetic_internal_key()).to_csv(key_path, index=False)
    pd.DataFrame({"image_id": ["2.007-1"], "donor_id": ["2.007"], "patch_id": ["p"], "patch_oop": [0.5]}).to_csv(patch_path, index=False)
    return annotation_path, key_path, patch_path


def test_column_normalization_handles_dominant_orientation_star() -> None:
    normalized, _ = normalize_expert_annotations(synthetic_annotations(1))

    assert "expert_dominant_orientation_deg_raw" in normalized.columns
    assert normalized.loc[0, "expert_dominant_orientation_deg_raw"] == 0


def test_unnamed_empty_columns_are_ignored() -> None:
    _, audit = normalize_expert_annotations(synthetic_annotations(1))

    assert "unnamed:_9" not in audit["normalized_columns"]


def test_extra_explanatory_columns_are_ignored() -> None:
    normalized, _ = normalize_expert_annotations(synthetic_annotations(1))

    assert len(normalized) == 1
    assert "dominant_orientation_deg:_myofibril_clockwise_(+ve)_or_anticlockwise_(-ve)_from_horizontal_axis" not in normalized.columns


def test_all_annotations_join_by_annotation_id(tmp_path: Path) -> None:
    annotation_path, key_path, _ = write_inputs(tmp_path)
    _, matched, summary, _ = validate_expert_annotations(validation_config(tmp_path), annotations=annotation_path, internal_key=key_path)

    assert len(matched) == 12
    assert summary["audit"]["annotations_matched_to_internal_key"] == 12


def test_duplicate_annotation_ids_are_reported(tmp_path: Path) -> None:
    annotations = synthetic_annotations(3)
    annotations.loc[1, "annotation_id"] = annotations.loc[0, "annotation_id"]
    annotation_path, key_path, _ = write_inputs(tmp_path, annotations=annotations)

    _, _, summary, _ = validate_expert_annotations(validation_config(tmp_path), annotations=annotation_path, internal_key=key_path)

    assert summary["audit"]["duplicate_annotation_ids"] == ["EXPERT_0001"]


def test_unmatched_annotation_ids_are_reported(tmp_path: Path) -> None:
    annotations = synthetic_annotations(3)
    annotations.loc[2, "annotation_id"] = "EXPERT_9999"
    annotation_path, key_path, _ = write_inputs(tmp_path, annotations=annotations, key=synthetic_internal_key(3))

    _, _, summary, _ = validate_expert_annotations(validation_config(tmp_path), annotations=annotation_path, internal_key=key_path)

    assert summary["audit"]["unmatched_annotation_ids"] == ["EXPERT_9999"]


def test_invalid_categorical_values_are_reported_not_fatal(tmp_path: Path) -> None:
    annotations = synthetic_annotations(2)
    annotations.loc[0, "striations_visible"] = "maybe"
    annotations.loc[1, "spacing_measurable"] = "sometimes"
    annotation_path, key_path, _ = write_inputs(tmp_path, annotations=annotations, key=synthetic_internal_key(2))

    _, _, summary, _ = validate_expert_annotations(validation_config(tmp_path), annotations=annotation_path, internal_key=key_path)

    assert summary["audit"]["invalid_values_by_field"]["striations_visible"] == 1
    assert summary["audit"]["invalid_values_by_field"]["spacing_measurable"] == 1


def test_missing_manual_sarcomere_length_does_not_fail(tmp_path: Path) -> None:
    annotation_path, key_path, _ = write_inputs(tmp_path)
    _, _, summary, _ = validate_expert_annotations(validation_config(tmp_path), annotations=annotation_path, internal_key=key_path)

    assert summary["audit"]["manual_sarcomere_length_completed_count"] == 0
    assert summary["spacing"]["spacing_validation_status"] == "not_validated_from_this_file"


def test_dominant_orientation_marked_unusable_by_default() -> None:
    normalized, _ = normalize_expert_annotations(synthetic_annotations(3))

    assert normalized["expert_orientation_usable_primary"].eq(False).all()


def test_oop_medians_by_visibility_are_computed(tmp_path: Path) -> None:
    annotation_path, key_path, _ = write_inputs(tmp_path)
    _, _, summary, _ = validate_expert_annotations(validation_config(tmp_path), annotations=annotation_path, internal_key=key_path)

    medians = summary["visibility_vs_automated_oop"]["oop_medians"]
    assert set(medians) == {"yes", "unclear", "no"}
    assert medians["yes"] is not None


def test_oop_medians_by_organisation_score_are_computed(tmp_path: Path) -> None:
    annotation_path, key_path, _ = write_inputs(tmp_path)
    _, _, summary, _ = validate_expert_annotations(validation_config(tmp_path), annotations=annotation_path, internal_key=key_path)

    medians = summary["organisation_score_vs_automated_oop"]["oop_medians"]
    assert medians["1"] is not None
    assert medians["5"] is not None


def test_spearman_skips_safely_if_n_too_small(tmp_path: Path) -> None:
    annotation_path, key_path, _ = write_inputs(tmp_path, annotations=synthetic_annotations(5), key=synthetic_internal_key(5))
    _, _, summary, _ = validate_expert_annotations(validation_config(tmp_path), annotations=annotation_path, internal_key=key_path, min_n_correlation=10)

    assert summary["organisation_score_vs_automated_oop"]["spearman"]["computed"] is False
    assert summary["organisation_score_vs_automated_oop"]["spearman"]["reason"] == "too_few_rows"


def test_confidence_filtered_analysis_works(tmp_path: Path) -> None:
    annotation_path, key_path, _ = write_inputs(tmp_path)
    _, _, summary, _ = validate_expert_annotations(validation_config(tmp_path), annotations=annotation_path, internal_key=key_path, min_confidence=3)

    assert summary["confidence_filtered"]["min_confidence"] == 3
    assert summary["confidence_filtered"]["row_count"] > 0


def test_summary_json_is_serializable(tmp_path: Path) -> None:
    annotation_path, key_path, _ = write_inputs(tmp_path)
    _, _, _, paths = validate_expert_annotations(validation_config(tmp_path), annotations=annotation_path, internal_key=key_path)

    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded["audit"]["total_rows"] == 12


def test_original_input_csv_is_not_modified(tmp_path: Path) -> None:
    annotation_path, key_path, _ = write_inputs(tmp_path)
    before = annotation_path.read_bytes()

    validate_expert_annotations(validation_config(tmp_path), annotations=annotation_path, internal_key=key_path)

    assert annotation_path.read_bytes() == before


def test_production_tables_are_not_modified(tmp_path: Path) -> None:
    annotation_path, key_path, patch_path = write_inputs(tmp_path)
    before = patch_path.read_bytes()

    validate_expert_annotations(validation_config(tmp_path), annotations=annotation_path, internal_key=key_path)

    assert patch_path.read_bytes() == before
