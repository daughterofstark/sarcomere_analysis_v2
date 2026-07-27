from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tifffile

from sarcomere_analysis.confocal_pipeline import (
    CONFOCAL_PIPELINE_IMAGE_COLUMNS,
    CONFOCAL_PIPELINE_PATCH_COLUMNS,
    PRIMARY_CONFOCAL_GATE,
    build_pipeline_image_table,
    run_confocal_pipeline,
)


def pipeline_config(tmp_path: Path) -> dict:
    return {
        "paths": {"raw_tiff_dir": "/path/to/local/widefield/raw", "output_dir": str(tmp_path / "results")},
        "outputs": {"manifest_csv": str(tmp_path / "results" / "tables" / "manifest.csv")},
        "calibration": {
            "pixel_size_um": 0.1299,
            "expected_sarcomere_spacing_um": {"min": 1.5, "max": 2.4},
        },
        "filename_pattern": {"regex": r"^(?P<donor_id>\d+\.\d+)-(?P<region_id>\d+)$"},
        "run": {"include_extensions": [".tif", ".tiff"], "recursive": False},
        "preprocessing": {
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "background_method": "gaussian",
            "background_sigma_px": 12.0,
            "enable_denoise": False,
            "denoise_sigma_px": 0.75,
            "output_dtype": "float32",
            "cache_preprocessed": False,
        },
        "masking": {
            "tissue_method": "percentile",
            "tissue_percentile": 5.0,
            "min_object_size_px": 4,
            "fill_holes": False,
        },
        "patches": {"patch_size_px": 64, "stride_px": 64, "margin_px": 0},
        "qc": {
            "min_tissue_fraction": 0.01,
            "min_contrast": 0.001,
            "min_gradient_energy": 0.0,
            "near_zero_threshold": 0.0,
            "saturation_threshold": 1.1,
        },
        "orientation": {
            "tensor_sigma_px": 1.0,
            "weight_mode": "energy_x_coherence",
            "min_orientation_weight_sum": 0.0,
            "min_orientation_valid_pixels": 4,
            "heterogeneity_method": "std",
            "eps": 1.0e-12,
        },
        "spacing": {
            "method": "autocorrelation",
            "fallback_to_fft": False,
            "min_spacing_um": 1.5,
            "max_spacing_um": 2.4,
            "min_periodicity_confidence": 0.15,
            "autocorrelation": {
                "profile_bin_px": 1.0,
                "min_profile_length_px": 16,
                "peak_baseline_percentile": 50.0,
            },
            "fft": {"min_profile_length_px": 16, "peak_prominence_ratio": 4.0},
        },
        "confocal_striation_mask": {
            "patch_size_px": 64,
            "stride_px": 64,
            "min_signal_fraction": 0.01,
            "min_gradient_energy": 0.0,
            "min_orientation_coherence": 0.0,
            "min_intensity_std": 0.0,
            "max_saturation_fraction": 1.0,
        },
    }


def stripe_image(size: int = 128, period: float = 16.0) -> np.ndarray:
    yy, xx = np.indices((size, size))
    values = 0.5 + 0.35 * np.sin(2 * np.pi * xx / period)
    values += 0.05 * np.sin(2 * np.pi * yy / (period * 2))
    return np.clip(values * 65535, 0, 65535).astype(np.uint16)


def write_calibrated_tiff(path: Path, pixel_size_um: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        stripe_image(),
        imagej=True,
        resolution=(1.0 / pixel_size_um, 1.0 / pixel_size_um),
        metadata={"unit": "um"},
    )


def write_uncalibrated_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((stripe_image() / 256).astype(np.uint8)).save(path)


def test_pipeline_writes_required_output_files(tmp_path: Path) -> None:
    confocal = tmp_path / "confocal"
    write_calibrated_tiff(confocal / "6052-CLEAR_STRIPES.tif", pixel_size_um=0.1)

    _, per_patch, per_image, summary, paths = run_confocal_pipeline(
        pipeline_config(tmp_path),
        confocal_root=confocal,
        output_directory=tmp_path / "out",
    )

    assert paths["manifest"].exists()
    assert paths["per_patch"].exists()
    assert paths["per_image"].exists()
    assert paths["summary_json"].exists()
    assert paths["summary_txt"].exists()
    assert list(per_patch.columns) == CONFOCAL_PIPELINE_PATCH_COLUMNS
    assert list(per_image.columns) == CONFOCAL_PIPELINE_IMAGE_COLUMNS
    assert summary["images_processed"] == 1


def test_per_image_calibration_is_preserved(tmp_path: Path) -> None:
    confocal = tmp_path / "confocal"
    write_calibrated_tiff(confocal / "5138.tif", pixel_size_um=0.08)

    manifest, _, per_image, _, _ = run_confocal_pipeline(
        pipeline_config(tmp_path),
        confocal_root=confocal,
        output_directory=tmp_path / "out",
    )

    assert bool(per_image["pixel_size_available"].iloc[0]) is True
    assert abs(float(per_image["pixel_size_x_um"].iloc[0]) - 0.08) < 0.005
    assert abs(float(manifest["pixel_size_x_um"].iloc[0]) - 0.08) < 0.005


def test_widefield_calibration_is_not_used(tmp_path: Path) -> None:
    confocal = tmp_path / "confocal"
    write_calibrated_tiff(confocal / "5138.tif", pixel_size_um=0.08)

    _, per_patch, _, summary, _ = run_confocal_pipeline(
        pipeline_config(tmp_path),
        confocal_root=confocal,
        output_directory=tmp_path / "out",
    )

    assert summary["widefield_calibration_used"] is False
    assert set(pd.to_numeric(per_patch["pixel_size_x_um"], errors="coerce").dropna().round(2)) == {0.08}


def test_primary_gate_recorded_as_moderate(tmp_path: Path) -> None:
    confocal = tmp_path / "confocal"
    write_calibrated_tiff(confocal / "5138.tif")

    _, _, _, summary, _ = run_confocal_pipeline(pipeline_config(tmp_path), confocal, tmp_path / "out")

    assert summary["primary_gate_used"] == PRIMARY_CONFOCAL_GATE


def test_relaxed_gate_not_used_as_primary(tmp_path: Path) -> None:
    confocal = tmp_path / "confocal"
    write_calibrated_tiff(confocal / "5138.tif")

    _, _, _, summary, _ = run_confocal_pipeline(pipeline_config(tmp_path), confocal, tmp_path / "out")

    assert "sensitivity_only" in summary["relaxed_gate_status"]
    assert summary["primary_gate_used"] != "moderate_relaxed_combined"


def test_missing_pixel_size_disables_micron_spacing_for_image(tmp_path: Path) -> None:
    confocal = tmp_path / "confocal"
    write_uncalibrated_png(confocal / "uncalibrated.png")

    _, per_patch, per_image, summary, _ = run_confocal_pipeline(
        pipeline_config(tmp_path),
        confocal_root=confocal,
        output_directory=tmp_path / "out",
    )

    assert bool(per_image["pixel_size_available"].iloc[0]) is False
    assert int(per_image["valid_selected_spacing_patches"].iloc[0]) == 0
    assert pd.to_numeric(per_patch["spacing_estimate_um"], errors="coerce").isna().all()
    assert summary["calibrated_images"] == 0


def test_per_image_summary_contains_selected_spacing_fields(tmp_path: Path) -> None:
    confocal = tmp_path / "confocal"
    write_calibrated_tiff(confocal / "5138.tif")

    _, _, per_image, _, _ = run_confocal_pipeline(pipeline_config(tmp_path), confocal, tmp_path / "out")

    for column in [
        "valid_selected_spacing_patches",
        "selected_spacing_valid_fraction",
        "selected_spacing_median_um",
        "selected_spacing_iqr_um",
        "selected_spacing_range_um",
    ]:
        assert column in per_image.columns


def test_summary_json_serializable(tmp_path: Path) -> None:
    confocal = tmp_path / "confocal"
    write_calibrated_tiff(confocal / "5138.tif")

    _, _, _, summary, paths = run_confocal_pipeline(pipeline_config(tmp_path), confocal, tmp_path / "out")

    json.dumps(summary)
    assert json.loads(paths["summary_json"].read_text(encoding="utf-8"))["mode"] == "confocal_pipeline"


def test_previews_disabled_by_default(tmp_path: Path) -> None:
    confocal = tmp_path / "confocal"
    write_calibrated_tiff(confocal / "5138.tif")

    _, _, _, summary, paths = run_confocal_pipeline(pipeline_config(tmp_path), confocal, tmp_path / "out")

    assert summary["previews_written"] is False
    assert summary["preview_paths"] == []
    assert not paths["previews"].exists()


def test_existing_widefield_outputs_are_not_modified(tmp_path: Path) -> None:
    confocal = tmp_path / "confocal"
    write_calibrated_tiff(confocal / "5138.tif")
    widefield = tmp_path / "results" / "tables" / "features_per_image.csv"
    widefield.parent.mkdir(parents=True, exist_ok=True)
    widefield.write_text("image_id,image_oop\n2.007-1,0.1\n", encoding="utf-8")
    before = widefield.read_bytes()

    run_confocal_pipeline(pipeline_config(tmp_path), confocal, tmp_path / "out")

    assert widefield.read_bytes() == before


def test_consolidated_per_image_spacing_median_maps_from_spacing_table() -> None:
    manifest = pd.DataFrame(
        [
            {
                "confocal_image_id": "5138",
                "filename": "5138.tif",
                "image_shape_y": 128,
                "image_shape_x": 128,
                "pixel_size_x_um": 0.08,
                "pixel_size_y_um": 0.08,
                "pixel_size_available": True,
            }
        ]
    )
    baseline = pd.DataFrame([{"confocal_image_id": "5138", "processing_status": "ok", "error_message": ""}])
    same = pd.DataFrame(
        [
            {
                "confocal_image_id": "5138",
                "filename": "5138.tif",
                "total_patches": 4,
                "candidate_patch_count": 2,
                "candidate_patch_fraction": 0.5,
                "selected_region_median_oop_128": 0.7,
                "all_region_median_oop_128": 0.6,
                "selected_vs_all_oop_difference_128": 0.1,
            }
        ]
    )
    spacing_image = pd.DataFrame(
        [
            {
                "confocal_image_id": "5138",
                "selected_median_spacing_um": 2.03,
                "selected_iqr_spacing_um": 0.2,
                "spacing_valid_patch_count_selected": 2,
                "spacing_valid_fraction_selected": 1.0,
            }
        ]
    )
    spacing_patch = pd.DataFrame(
        [
            {
                "confocal_image_id": "5138",
                "candidate_striation_region": True,
                "spacing_valid": True,
                "spacing_estimate_um": 1.9,
            },
            {
                "confocal_image_id": "5138",
                "candidate_striation_region": True,
                "spacing_valid": True,
                "spacing_estimate_um": 2.2,
            },
        ]
    )

    out = build_pipeline_image_table(manifest, baseline, same, spacing_image, spacing_patch)

    assert out["selected_spacing_median_um"].iloc[0] == 2.03
    assert out["selected_spacing_range_um"].iloc[0] == "1.9000-2.2000"
