from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sarcomere_analysis.confocal_striation_mask import (
    STRIATION_PATCH_COLUMNS,
    candidate_decision,
    run_confocal_striation_mask_audit,
)


def mask_config(tmp_path: Path) -> dict:
    return {
        "paths": {"raw_tiff_dir": str(tmp_path / "raw"), "output_dir": str(tmp_path / "results")},
        "outputs": {"manifest_csv": str(tmp_path / "results" / "tables" / "manifest.csv")},
        "calibration": {
            "pixel_size_um": 0.1299,
            "expected_sarcomere_spacing_um": {"min": 1.5, "max": 2.4},
        },
        "filename_pattern": {"regex": r"^(?P<donor_id>\d+\.\d+)-(?P<region_id>\d+)$"},
        "run": {"include_extensions": [".tif", ".tiff"], "recursive": False},
        "orientation": {"min_orientation_valid_pixels": 8, "min_orientation_weight_sum": 0.0, "tensor_sigma_px": 1.0},
        "masking": {"min_object_size_px": 4},
    }


def write_image(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(values, 0, 255).astype(np.uint8)).save(path)


def stripe_image(size: int = 96, period: float = 12.0) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    values = 128 + 90 * np.sin(2 * np.pi * x / period)
    return values.astype(np.uint8)


def manifest_for(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        name = path.stem
        rows.append(
            {
                "confocal_image_id": name,
                "filename": path.name,
                "source_path": str(path),
                "extension": path.suffix.lower(),
                "image_shape_y": 96,
                "image_shape_x": 96,
                "dtype": "uint8",
                "inferred_sample_id": name,
                "expected_positive_example": "6052" in name or "5138" in name,
                "noted_complex_example": "3112" in name,
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def run_on_manifest(tmp_path: Path, manifest: pd.DataFrame, **kwargs):
    manifest_path = tmp_path / "confocal_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return run_confocal_striation_mask_audit(
        mask_config(tmp_path),
        manifest_path,
        output_directory=tmp_path / "out",
        patch_size=32,
        stride=32,
        min_gradient_energy=kwargs.pop("min_gradient_energy", 0.000001),
        min_orientation_coherence=kwargs.pop("min_orientation_coherence", 0.05),
        min_intensity_std=kwargs.pop("min_intensity_std", 0.005),
        max_saturation_fraction=kwargs.pop("max_saturation_fraction", 0.10),
        **kwargs,
    )


def test_high_contrast_oriented_stripe_image_produces_candidate_regions(tmp_path: Path) -> None:
    path = tmp_path / "6052_stripes.png"
    write_image(path, stripe_image())
    per_patch, per_image, _, _ = run_on_manifest(tmp_path, manifest_for([path]))

    assert int(per_patch["candidate_striation_region"].sum()) > 0
    assert int(per_image.loc[0, "candidate_patch_count"]) > 0


def test_flat_image_produces_no_or_few_candidate_regions(tmp_path: Path) -> None:
    path = tmp_path / "flat.png"
    write_image(path, np.full((96, 96), 128, dtype=np.uint8))
    per_patch, per_image, _, _ = run_on_manifest(tmp_path, manifest_for([path]))

    assert int(per_patch["candidate_striation_region"].sum()) == 0
    assert int(per_image.loc[0, "candidate_patch_count"]) == 0


def test_saturated_regions_can_be_rejected() -> None:
    metrics = {
        "intensity_mean": 0.9,
        "signal_fraction": 0.9,
        "gradient_energy": 0.01,
        "orientation_coherence": 0.8,
        "intensity_std": 0.2,
        "saturation_fraction": 0.5,
    }
    params = {
        "min_signal_fraction": 0.05,
        "min_gradient_energy": 0.0001,
        "min_orientation_coherence": 0.2,
        "min_intensity_std": 0.03,
        "max_saturation_fraction": 0.1,
    }

    decision = candidate_decision(metrics, params)

    assert decision["candidate_striation_region"] is False
    assert "high_saturation_fraction" in decision["rejection_reason"]


def test_output_mask_table_has_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "6052_stripes.png"
    write_image(path, stripe_image())
    per_patch, _, _, _ = run_on_manifest(tmp_path, manifest_for([path]))

    for column in STRIATION_PATCH_COLUMNS:
        assert column in per_patch.columns


def test_per_image_summary_counts_candidates_correctly(tmp_path: Path) -> None:
    path = tmp_path / "6052_stripes.png"
    write_image(path, stripe_image())
    per_patch, per_image, _, _ = run_on_manifest(tmp_path, manifest_for([path]))

    expected = int(per_patch["candidate_striation_region"].sum())
    assert int(per_image.loc[0, "candidate_patch_count"]) == expected


def test_expected_positive_flags_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "5138_good.png"
    write_image(path, stripe_image())
    per_patch, per_image, summary, _ = run_on_manifest(tmp_path, manifest_for([path]))

    assert bool(per_patch["expected_positive_example"].iloc[0])
    assert bool(per_image.loc[0, "expected_positive_example"])
    assert summary["expected_positive_examples"][0]["filename"] == "5138_good.png"


def test_3112_complex_flag_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "3112_complex.png"
    write_image(path, stripe_image())
    per_patch, per_image, summary, _ = run_on_manifest(tmp_path, manifest_for([path]))

    assert bool(per_patch["noted_complex_example"].iloc[0])
    assert bool(per_image.loc[0, "noted_complex_example"])
    assert summary["noted_complex_examples"][0]["filename"] == "3112_complex.png"


def test_preview_writing_can_be_disabled_by_default(tmp_path: Path) -> None:
    path = tmp_path / "6052_stripes.png"
    write_image(path, stripe_image())
    _, _, summary, paths = run_on_manifest(tmp_path, manifest_for([path]))

    assert summary["previews_written"] is False
    assert summary["preview_paths"] == []
    assert not paths["previews"].exists()


def test_summary_json_serializable(tmp_path: Path) -> None:
    path = tmp_path / "6052_stripes.png"
    write_image(path, stripe_image())
    _, _, summary, paths = run_on_manifest(tmp_path, manifest_for([path]))

    json.dumps(summary)
    assert paths["summary_json"].exists()


def test_existing_widefield_and_baseline_outputs_are_not_modified(tmp_path: Path) -> None:
    path = tmp_path / "6052_stripes.png"
    write_image(path, stripe_image())
    widefield = tmp_path / "results" / "tables" / "features_per_image.csv"
    baseline = tmp_path / "results" / "confocal_baseline" / "confocal_baseline_per_image.csv"
    widefield.parent.mkdir(parents=True, exist_ok=True)
    baseline.parent.mkdir(parents=True, exist_ok=True)
    widefield.write_text("image_id,image_oop\n2.007-1,0.1\n", encoding="utf-8")
    baseline.write_text("confocal_image_id,image_oop\n6052,0.5\n", encoding="utf-8")
    widefield_before = widefield.read_bytes()
    baseline_before = baseline.read_bytes()

    run_on_manifest(tmp_path, manifest_for([path]))

    assert widefield.read_bytes() == widefield_before
    assert baseline.read_bytes() == baseline_before
