from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sarcomere_analysis.synthetic_oop import (
    SYNTHETIC_RESULT_COLUMNS,
    generate_synthetic_striated_image,
    run_orientation_on_synthetic,
    validate_synthetic_oop,
)
from sarcomere_analysis.validation_zdisc_masks import axial_angular_error_deg


def synthetic_config(tmp_path: Path) -> dict:
    return {
        "paths": {"raw_tiff_dir": str(tmp_path / "raw"), "output_dir": str(tmp_path / "results")},
        "outputs": {"manifest_csv": str(tmp_path / "results" / "tables" / "manifest.csv")},
        "calibration": {
            "pixel_size_um": 0.1299,
            "expected_sarcomere_spacing_um": {"min": 1.5, "max": 2.4},
        },
        "filename_pattern": {"regex": r"^(?P<donor_id>\d+\.\d+)-(?P<region_id>\d+)$"},
        "run": {"include_extensions": [".tif", ".tiff"], "recursive": False},
        "orientation": {
            "tensor_sigma_px": 1.0,
            "weight_mode": "energy_x_coherence",
            "min_orientation_weight_sum": 1.0e-8,
            "min_orientation_valid_pixels": 8,
            "heterogeneity_method": "std",
            "eps": 1.0e-12,
        },
    }


def test_generator_returns_expected_shape() -> None:
    image = generate_synthetic_striated_image(size=64, orientation_deg=30, seed=1)

    assert image.shape == (64, 64)
    assert image.dtype == np.float32


def test_generator_is_deterministic_with_seed() -> None:
    first = generate_synthetic_striated_image(size=64, orientation_deg=30, disorder_level="medium", noise_sigma=0.1, seed=10)
    second = generate_synthetic_striated_image(size=64, orientation_deg=30, disorder_level="medium", noise_sigma=0.1, seed=10)

    np.testing.assert_allclose(first, second)


def test_clean_horizontal_pattern_recovers_finite_oop_in_unit_interval(tmp_path: Path) -> None:
    image = generate_synthetic_striated_image(size=64, orientation_deg=0, disorder_level="low", noise_sigma=0, seed=2)
    metrics = run_orientation_on_synthetic(image, "synthetic", synthetic_config(tmp_path))

    assert np.isfinite(metrics["image_oop"])
    assert 0.0 <= metrics["image_oop"] <= 1.0


def test_clean_high_order_pattern_has_higher_oop_than_disordered_pattern(tmp_path: Path) -> None:
    cfg = synthetic_config(tmp_path)
    clean = generate_synthetic_striated_image(size=96, orientation_deg=30, disorder_level="low", noise_sigma=0, seed=3)
    disordered = generate_synthetic_striated_image(size=96, orientation_deg=30, disorder_level="high", noise_sigma=0.18, seed=3)

    clean_oop = run_orientation_on_synthetic(clean, "clean", cfg)["image_oop"]
    disordered_oop = run_orientation_on_synthetic(disordered, "disordered", cfg)["image_oop"]

    assert clean_oop > disordered_oop


def test_axial_angular_error_handles_wraparound() -> None:
    assert axial_angular_error_deg(175, 5) == 10.0


def test_validation_result_csv_has_required_columns(tmp_path: Path) -> None:
    results, _, paths = validate_synthetic_oop(synthetic_config(tmp_path), seed=4, size=64)

    assert paths["results_csv"].exists()
    assert list(results.columns) == SYNTHETIC_RESULT_COLUMNS


def test_summary_json_is_serializable(tmp_path: Path) -> None:
    _, summary, paths = validate_synthetic_oop(synthetic_config(tmp_path), seed=5, size=64)

    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert loaded["synthetic_examples"] == summary["synthetic_examples"]


def test_production_feature_analysis_tables_are_not_modified(tmp_path: Path) -> None:
    cfg = synthetic_config(tmp_path)
    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    feature_path = tables / "features_per_image.csv"
    analysis_path = tables / "analysis_per_image.csv"
    feature_path.write_text("image_id,donor_id,image_oop\n2.007-1,2.007,0.5\n", encoding="utf-8")
    analysis_path.write_text("image_id,donor_id,image_oop\n2.007-1,2.007,0.5\n", encoding="utf-8")
    feature_before = feature_path.read_bytes()
    analysis_before = analysis_path.read_bytes()

    validate_synthetic_oop(cfg, seed=6, size=64)

    assert feature_path.read_bytes() == feature_before
    assert analysis_path.read_bytes() == analysis_before
