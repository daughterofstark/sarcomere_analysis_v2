from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sarcomere_analysis.confocal_spacing_audit import run_confocal_spacing_audit


def spacing_config(tmp_path: Path) -> dict:
    return {
        "paths": {"raw_tiff_dir": str(tmp_path / "raw"), "output_dir": str(tmp_path / "results")},
        "outputs": {"manifest_csv": str(tmp_path / "results" / "tables" / "manifest.csv")},
        "calibration": {
            "pixel_size_um": 0.1299,
            "expected_sarcomere_spacing_um": {"min": 1.5, "max": 2.4},
        },
        "filename_pattern": {"regex": r"^(?P<donor_id>\d+\.\d+)-(?P<region_id>\d+)$"},
        "run": {"include_extensions": [".tif", ".tiff"], "recursive": False},
        "preprocessing": {
            "lower_percentile": 0.0,
            "upper_percentile": 100.0,
            "background_method": "none",
            "enable_denoise": False,
        },
        "spacing": {
            "method": "autocorrelation",
            "min_periodicity_confidence": 0.05,
            "autocorrelation": {
                "profile_bin_px": 1.0,
                "min_profile_length_px": 32,
                "peak_baseline_percentile": 25.0,
            },
        },
    }


def stripe_image(size: int = 128, period_px: int = 20) -> np.ndarray:
    y, x = np.indices((size, size))
    image = 0.5 + 0.4 * np.sin(2.0 * np.pi * x / period_px)
    image += 0.02 * y / max(size - 1, 1)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image * 255).astype(np.uint8)).save(path)


def write_inputs(
    tmp_path: Path,
    calibrated: bool = True,
    candidate: bool = True,
    flat: bool = False,
    expected: bool = True,
    complex_flag: bool = False,
) -> tuple[Path, Path]:
    image_path = tmp_path / "confocal.png"
    write_png(image_path, np.full((128, 128), 0.5, dtype=np.float32) if flat else stripe_image())
    calibration = pd.DataFrame(
        [
            {
                "confocal_image_id": "5138" if expected else "3112",
                "filename": image_path.name,
                "source_path": str(image_path),
                "image_shape_y": 128,
                "image_shape_x": 128,
                "pixel_size_x_um": 0.1 if calibrated else np.nan,
                "pixel_size_y_um": 0.1 if calibrated else np.nan,
                "pixel_size_available": calibrated,
                "isotropic_pixels": calibrated,
                "expected_positive_example": expected,
                "noted_complex_example": complex_flag,
            }
        ]
    )
    patch = pd.DataFrame(
        [
            {
                "confocal_image_id": "5138" if expected else "3112",
                "filename": image_path.name,
                "patch_id": "p0",
                "y0": 0,
                "x0": 0,
                "y1": 128,
                "x1": 128,
                "center_y": 64,
                "center_x": 64,
                "candidate_striation_region": candidate,
                "patch_oop_128": 0.9,
                "patch_mean_orientation_deg_128": 0.0,
                "patch_orientation_coherence_mean_128": 0.8,
                "gradient_energy": 0.01,
                "intensity_std": 0.2,
                "contrast": 0.5,
                "expected_positive_example": expected,
                "noted_complex_example": complex_flag,
            }
        ]
    )
    calibration_path = tmp_path / "calibration.csv"
    patch_path = tmp_path / "same_grid.csv"
    calibration.to_csv(calibration_path, index=False)
    patch.to_csv(patch_path, index=False)
    return calibration_path, patch_path


def run_synthetic(tmp_path: Path, **kwargs):
    calibration_path, patch_path = write_inputs(tmp_path, **kwargs)
    return run_confocal_spacing_audit(
        spacing_config(tmp_path),
        calibration_table=calibration_path,
        same_grid_oop_table=patch_path,
        output_directory=tmp_path / "out",
        spacing_min_um=1.5,
        spacing_max_um=2.4,
    )


def test_converts_spacing_band_to_pixels_per_image(tmp_path: Path) -> None:
    per_patch, _, _, _ = run_synthetic(tmp_path)

    assert round(float(per_patch.loc[0, "spacing_band_min_px"]), 3) == 15.0
    assert round(float(per_patch.loc[0, "spacing_band_max_px"]), 3) == 24.0


def test_does_not_use_widefield_calibration(tmp_path: Path) -> None:
    per_patch, _, summary, _ = run_synthetic(tmp_path)

    assert round(float(per_patch.loc[0, "pixel_size_x_um"]), 3) == 0.1
    assert summary["widefield_calibration_used"] is False


def test_skips_images_without_calibration(tmp_path: Path) -> None:
    per_patch, per_image, summary, _ = run_synthetic(tmp_path, calibrated=False)

    assert per_patch.loc[0, "spacing_valid"] is False or not bool(per_patch.loc[0, "spacing_valid"])
    assert per_patch.loc[0, "spacing_failure_reason"] == "missing_per_image_pixel_size"
    assert per_image.loc[0, "spacing_valid_patch_count_selected"] == 0
    assert summary["calibrated_image_count"] == 0


def test_computes_spacing_estimate_in_um_from_px(tmp_path: Path) -> None:
    per_patch, _, _, _ = run_synthetic(tmp_path)

    valid = per_patch.loc[per_patch["spacing_valid"]].iloc[0]
    assert abs(valid["spacing_estimate_px"] - 20.0) <= 1.0
    assert abs(valid["spacing_estimate_um"] - valid["spacing_estimate_px"] * 0.1) < 1e-6


def test_selected_region_spacing_yield_computed(tmp_path: Path) -> None:
    _, per_image, summary, _ = run_synthetic(tmp_path)

    assert per_image.loc[0, "candidate_patch_count"] == 1
    assert per_image.loc[0, "spacing_valid_patch_count_selected"] == 1
    assert summary["valid_spacing_patch_count_selected"] == 1


def test_handles_no_valid_spacing_gracefully(tmp_path: Path) -> None:
    per_patch, per_image, summary, _ = run_synthetic(tmp_path, flat=True)

    assert per_patch.loc[0, "spacing_valid"] is False or not bool(per_patch.loc[0, "spacing_valid"])
    assert per_image.loc[0, "spacing_valid_patch_count_selected"] == 0
    assert summary["selected_spacing_um_summary"]["median"] is None


def test_expected_positive_and_complex_flags_preserved(tmp_path: Path) -> None:
    _, per_image, summary, _ = run_synthetic(tmp_path, expected=False, complex_flag=True)

    assert bool(per_image.loc[0, "noted_complex_example"]) is True
    assert summary["special_image_summaries"][0]["noted_complex_example"] is True


def test_summary_json_serializable(tmp_path: Path) -> None:
    _, _, summary, paths = run_synthetic(tmp_path)

    json.dumps(summary)
    assert paths["summary_json"].exists()


def test_previews_disabled_by_default(tmp_path: Path) -> None:
    _, _, summary, paths = run_synthetic(tmp_path)

    assert summary["previews_written"] is False
    assert not paths["previews"].exists()


def test_existing_widefield_and_confocal_outputs_not_modified(tmp_path: Path) -> None:
    calibration_path, patch_path = write_inputs(tmp_path)
    widefield = tmp_path / "results" / "tables" / "features_per_image.csv"
    confocal = tmp_path / "results" / "confocal_same_grid_oop" / "confocal_same_grid_oop_per_patch.csv"
    widefield.parent.mkdir(parents=True, exist_ok=True)
    confocal.parent.mkdir(parents=True, exist_ok=True)
    widefield.write_text("image_id,image_oop\n2.007-1,0.1\n", encoding="utf-8")
    confocal.write_text("confocal_image_id,patch_id\n5138,p0\n", encoding="utf-8")
    before_widefield = widefield.read_bytes()
    before_confocal = confocal.read_bytes()

    run_confocal_spacing_audit(
        spacing_config(tmp_path),
        calibration_table=calibration_path,
        same_grid_oop_table=patch_path,
        output_directory=tmp_path / "out",
    )

    assert widefield.read_bytes() == before_widefield
    assert confocal.read_bytes() == before_confocal
