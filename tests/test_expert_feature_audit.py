from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sarcomere_analysis.expert_feature_audit import (
    audit_expert_feature_relationships,
    build_expert_feature_table,
    organisation_feature_summary,
    spearman_pair,
    visibility_feature_summary,
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


def synthetic_matched(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "annotation_id": f"EXPERT_{idx + 1:04d}",
                "patch_filename": f"EXPERT_{idx + 1:04d}.png",
                "image_id": f"2.007-{(idx % 2) + 1}",
                "donor_id": "2.007",
                "patch_id": f"patch_{idx}",
                "oop_bin": ["low", "medium", "high"][idx % 3],
                "automated_patch_oop": idx / max(n - 1, 1),
                "automated_patch_orientation_deg": idx * 5,
                "striations_visible": ["no", "unclear", "yes"][idx % 3],
                "organisation_score": (idx % 5) + 1,
                "confidence_score": [2, 3, 4, 5][idx % 4],
                "spacing_measurable": ["no", "unclear", "yes"][idx % 3],
                "expert_orientation_usable_primary": False,
                "validation_match_status": "matched",
            }
            for idx in range(n)
        ]
    )


def synthetic_patch_features(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": f"2.007-{(idx % 2) + 1}",
                "donor_id": "2.007",
                "patch_id": f"patch_{idx}",
                "patch_oop": idx / max(n - 1, 1),
                "rms_contrast": float(idx),
                "gradient_energy": float(n - idx),
                "valid_for_orientation": idx % 2 == 0,
                "manual_leakage_column": idx,
                "health_status": "healthy",
                "invalid_reason": "ok",
            }
            for idx in range(n)
        ]
    )


def write_inputs(tmp_path: Path, matched: pd.DataFrame | None = None, patches: pd.DataFrame | None = None) -> tuple[Path, Path]:
    validation_dir = tmp_path / "results" / "validation" / "expert_annotation_validation"
    tables_dir = tmp_path / "results" / "tables"
    validation_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)
    matched_path = validation_dir / "expert_annotation_validation_matched.csv"
    patch_path = tables_dir / "features_per_patch.csv"
    (matched if matched is not None else synthetic_matched()).to_csv(matched_path, index=False)
    (patches if patches is not None else synthetic_patch_features()).to_csv(patch_path, index=False)
    return matched_path, patch_path


def test_joins_expert_matched_rows_to_patch_features() -> None:
    table, features, _ = build_expert_feature_table(synthetic_matched(), synthetic_patch_features())

    assert len(table) == 12
    assert "rms_contrast" in features


def test_identifier_leakage_manual_columns_excluded() -> None:
    _, features, excluded = build_expert_feature_table(synthetic_matched(), synthetic_patch_features())

    assert "manual_leakage_column" not in features
    assert "health_status" not in features
    assert any("manual_leakage_column" == column for column in excluded)


def test_boolean_numeric_flags_converted_safely() -> None:
    table, features, _ = build_expert_feature_table(synthetic_matched(), synthetic_patch_features())

    assert "valid_for_orientation" in features
    assert set(table["valid_for_orientation"].dropna().unique()).issubset({0, 1})


def test_visibility_medians_computed_correctly() -> None:
    table, features, _ = build_expert_feature_table(synthetic_matched(), synthetic_patch_features())
    summary = visibility_feature_summary(table, features)
    row = summary.loc[summary["feature"] == "rms_contrast"].iloc[0]

    assert row["median_yes"] == 6.5
    assert row["median_no"] == 4.5


def test_yes_minus_no_separation_computed_correctly() -> None:
    table, features, _ = build_expert_feature_table(synthetic_matched(), synthetic_patch_features())
    summary = visibility_feature_summary(table, features)
    row = summary.loc[summary["feature"] == "rms_contrast"].iloc[0]

    assert row["yes_minus_no"] == 2.0
    assert row["abs_yes_minus_no"] == 2.0


def test_spearman_skips_safely_when_n_too_small() -> None:
    stats = spearman_pair(synthetic_matched(5), "organisation_score", "automated_patch_oop", min_n=10)

    assert stats["computed"] is False
    assert stats["reason"] == "too_few_rows"


def test_confidence_filtered_spearman_works() -> None:
    table, features, _ = build_expert_feature_table(synthetic_matched(), synthetic_patch_features())
    summary = organisation_feature_summary(table, features, min_n=3, min_confidence=3)
    row = summary.loc[summary["feature"] == "rms_contrast"].iloc[0]

    assert row["n_confidence_filtered"] >= 3


def test_missingness_reported(tmp_path: Path) -> None:
    patches = synthetic_patch_features()
    patches.loc[0:2, "rms_contrast"] = np.nan
    matched_path, patch_path = write_inputs(tmp_path, patches=patches)

    _, _, _, _, summary, _ = audit_expert_feature_relationships(audit_config(tmp_path), matched=matched_path, patch_features=patch_path)

    assert summary["audit"]["missingness_per_feature"]["rms_contrast"] == 3


def test_oop_prior_result_carried_into_summary(tmp_path: Path) -> None:
    matched_path, patch_path = write_inputs(tmp_path)
    _, _, _, _, summary, _ = audit_expert_feature_relationships(audit_config(tmp_path), matched=matched_path, patch_features=patch_path, min_n=3)

    assert summary["oop_specific_statement"]["patch_oop_result"] is not None
    assert "near-zero correlation" in summary["oop_specific_statement"]["statement"]


def test_output_json_serializable(tmp_path: Path) -> None:
    matched_path, patch_path = write_inputs(tmp_path)
    _, _, _, _, _, paths = audit_expert_feature_relationships(audit_config(tmp_path), matched=matched_path, patch_features=patch_path)

    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded["mode"] == "expert_annotation_feature_audit"


def test_production_tables_are_not_modified(tmp_path: Path) -> None:
    matched_path, patch_path = write_inputs(tmp_path)
    before = patch_path.read_bytes()

    audit_expert_feature_relationships(audit_config(tmp_path), matched=matched_path, patch_features=patch_path)

    assert patch_path.read_bytes() == before
