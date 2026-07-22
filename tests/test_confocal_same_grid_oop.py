from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sarcomere_analysis.confocal_same_grid_oop import (
    SAME_GRID_IMAGE_COLUMNS,
    SAME_GRID_PATCH_COLUMNS,
    run_confocal_same_grid_oop,
)


def same_grid_config(tmp_path: Path) -> dict:
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
        "masking": {"tissue_method": "percentile", "tissue_percentile": 5.0, "min_object_size_px": 1, "fill_holes": False},
        "orientation": {
            "tensor_sigma_px": 1.0,
            "weight_mode": "energy_x_coherence",
            "min_orientation_weight_sum": 0.0,
            "min_orientation_valid_pixels": 4,
            "heterogeneity_method": "std",
            "eps": 1.0e-12,
        },
    }


def stripe_image(size: int = 64, period: int = 8) -> np.ndarray:
    y, x = np.indices((size, size))
    image = 0.5 + 0.35 * np.sin(2.0 * np.pi * x / period)
    image += 0.1 * (y / max(size - 1, 1))
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def write_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray((image * 255).astype(np.uint8)).save(path)


def synthetic_inputs(tmp_path: Path, include_selective: bool = False, bad_patch: bool = False) -> tuple[Path, Path]:
    output_root = tmp_path / "results"
    output_root.mkdir(parents=True, exist_ok=True)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_5138 = image_dir / "5138.png"
    image_3112 = image_dir / "3112.png"
    write_image(image_5138, stripe_image())
    write_image(image_3112, np.full((64, 64), 0.5, dtype=np.float32))

    manifest = pd.DataFrame(
        [
            {
                "confocal_image_id": "5138",
                "filename": "5138.png",
                "source_path": str(image_5138),
                "expected_positive_example": True,
                "noted_complex_example": False,
            },
            {
                "confocal_image_id": "3112",
                "filename": "3112.png",
                "source_path": str(image_3112),
                "expected_positive_example": False,
                "noted_complex_example": True,
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    rows = []
    for image_id, filename, expected, complex_flag in [
        ("5138", "5138.png", True, False),
        ("3112", "3112.png", False, True),
    ]:
        for idx, (y0, x0) in enumerate([(0, 0), (0, 32), (32, 0), (32, 32)]):
            y1, x1 = y0 + 32, x0 + 32
            if bad_patch and image_id == "3112" and idx == 3:
                y1 = 96
            rows.append(
                {
                    "confocal_image_id": image_id,
                    "filename": filename,
                    "patch_id": f"{image_id}_p{idx:05d}",
                    "y0": y0,
                    "x0": x0,
                    "y1": y1,
                    "x1": x1,
                    "center_y": y0 + 16,
                    "center_x": x0 + 16,
                    "candidate_striation_region": idx < 2,
                    "expected_positive_example": expected,
                    "noted_complex_example": complex_flag,
                    "gradient_energy": 0.01 if idx < 2 else 0.001,
                    "intensity_std": 0.2 if idx < 2 else 0.02,
                    "contrast": 0.5 if idx < 2 else 0.05,
                }
            )
    patch_table = pd.DataFrame(rows)
    patch_path = tmp_path / "patches.csv"
    patch_table.to_csv(patch_path, index=False)

    if include_selective:
        selective_dir = output_root / "confocal_selective_analysis"
        selective_dir.mkdir(parents=True)
        selective = patch_table.copy()
        selective["selected_variant"] = "moderate"
        selective.loc[selective["patch_id"].str.endswith("00002"), "candidate_striation_region"] = True
        selective.to_csv(selective_dir / "confocal_selective_per_patch.csv", index=False)

    return patch_path, manifest_path


def run_synthetic_same_grid(tmp_path: Path, **kwargs):
    patch_path, manifest_path = synthetic_inputs(
        tmp_path,
        include_selective=kwargs.pop("include_selective", False),
        bad_patch=kwargs.pop("bad_patch", False),
    )
    return run_confocal_same_grid_oop(
        same_grid_config(tmp_path),
        patch_table=patch_path,
        manifest=manifest_path,
        output_directory=tmp_path / "same_grid_out",
        **kwargs,
    )


def test_uses_same_grid_patch_coordinates(tmp_path: Path) -> None:
    per_patch, _, _, _ = run_synthetic_same_grid(tmp_path)

    row = per_patch.loc[per_patch["patch_id"] == "5138_p00001"].iloc[0]
    assert int(row["x0"]) == 32
    assert int(row["y1"]) == 32


def test_computes_oop_for_same_grid_patches(tmp_path: Path) -> None:
    per_patch, _, _, _ = run_synthetic_same_grid(tmp_path)

    assert set(SAME_GRID_PATCH_COLUMNS).issubset(per_patch.columns)
    assert per_patch.loc[per_patch["confocal_image_id"] == "5138", "patch_oop_128"].notna().any()


def test_selected_vs_all_oop_summaries_compute(tmp_path: Path) -> None:
    _, per_image, summary, _ = run_synthetic_same_grid(tmp_path)

    assert set(SAME_GRID_IMAGE_COLUMNS).issubset(per_image.columns)
    image_row = per_image.loc[per_image["confocal_image_id"] == "5138"].iloc[0]
    assert pd.notna(image_row["selected_region_median_oop_128"])
    assert "selected_vs_all_oop_summary" in summary


def test_moderate_candidate_flags_override_when_selective_output_exists(tmp_path: Path) -> None:
    per_patch, _, summary, _ = run_synthetic_same_grid(tmp_path, include_selective=True)

    selected = per_patch.loc[per_patch["confocal_image_id"] == "5138", "candidate_striation_region"].tolist()
    assert selected == [True, True, True, False]
    assert summary["candidate_source"] == "confocal_selective_analysis_moderate"
    assert per_patch["candidate_source"].eq("confocal_selective_analysis_moderate").all()


def test_expected_positive_and_complex_flags_preserved(tmp_path: Path) -> None:
    _, per_image, _, _ = run_synthetic_same_grid(tmp_path)

    expected = per_image.loc[per_image["confocal_image_id"] == "5138", "expected_positive_example"].iloc[0]
    complex_flag = per_image.loc[per_image["confocal_image_id"] == "3112", "noted_complex_example"].iloc[0]
    assert bool(expected) is True
    assert bool(complex_flag) is True


def test_patch_errors_do_not_crash_whole_run(tmp_path: Path) -> None:
    per_patch, _, summary, _ = run_synthetic_same_grid(tmp_path, bad_patch=True)

    assert (per_patch["processing_status"] == "error").any()
    assert (per_patch["processing_status"] == "ok").any()
    assert summary["patches_error"] >= 1


def test_spacing_in_microns_not_reported_for_confocal_same_grid(tmp_path: Path) -> None:
    _, per_image, summary, _ = run_synthetic_same_grid(tmp_path)

    assert summary["spacing_status"] == "not_computed_in_microns_confocal_pixel_size_unknown"
    assert not any("spacing_um" in column for column in per_image.columns)


def test_summary_json_is_serializable(tmp_path: Path) -> None:
    _, _, summary, _ = run_synthetic_same_grid(tmp_path)

    json.dumps(summary)


def test_previews_disabled_by_default(tmp_path: Path) -> None:
    _, _, summary, paths = run_synthetic_same_grid(tmp_path)

    assert summary["previews_written"] is False
    assert not paths["previews"].exists()


def test_existing_widefield_and_confocal_inputs_are_not_modified(tmp_path: Path) -> None:
    output_root = tmp_path / "results"
    sentinel_dir = output_root / "tables"
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    sentinel = sentinel_dir / "analysis_per_image.csv"
    sentinel.write_text("do,not,modify\n", encoding="utf-8")
    patch_path, manifest_path = synthetic_inputs(tmp_path)
    before_patch = patch_path.read_bytes()
    before_manifest = manifest_path.read_bytes()
    before_sentinel = sentinel.read_bytes()

    run_confocal_same_grid_oop(
        same_grid_config(tmp_path),
        patch_table=patch_path,
        manifest=manifest_path,
        output_directory=tmp_path / "same_grid_out",
    )

    assert patch_path.read_bytes() == before_patch
    assert manifest_path.read_bytes() == before_manifest
    assert sentinel.read_bytes() == before_sentinel
