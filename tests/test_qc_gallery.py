from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from sarcomere_analysis.qc_gallery import (
    GALLERY_COLUMNS,
    build_qc_gallery_index,
    write_gallery_html,
    write_gallery_index,
)


def gallery_config(tmp_path: Path) -> dict:
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


def write_synthetic_gallery_inputs(tmp_path: Path, with_previews: bool = False) -> None:
    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        [
            {"image_id": "3.110-1", "donor_id": "3.110", "region_id": "1", "image_path": "/tmp/3.110-1.tif"},
            {"image_id": "4.070-2", "donor_id": "4.070", "region_id": "2", "image_path": "/tmp/4.070-2.tif"},
        ]
    ).to_csv(tables / "manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "image_id": "3.110-1",
                "donor_id": "3.110",
                "tissue_fraction": 0.8,
                "image_oop": 0.2,
                "image_oop_heterogeneity": 0.1,
                "n_spacing_valid_patches": 1,
                "spacing_valid_fraction": 0.02,
                "image_spacing_median_um": 1.6,
            },
            {
                "image_id": "4.070-2",
                "donor_id": "4.070",
                "tissue_fraction": 0.5,
                "image_oop": 0.3,
                "image_oop_heterogeneity": 0.2,
                "n_spacing_valid_patches": 0,
                "spacing_valid_fraction": 0.0,
                "image_spacing_median_um": np.nan,
            },
        ]
    ).to_csv(tables / "per_image_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"image_id": "3.110-1", "donor_id": "3.110", "status": "ok"},
            {"image_id": "4.070-2", "donor_id": "4.070", "status": "ok"},
        ]
    ).to_csv(tables / "batch_run_summary.csv", index=False)
    if with_previews:
        previews = tmp_path / "results" / "previews"
        previews.mkdir(parents=True)
        image = Image.fromarray(np.zeros((8, 8), dtype=np.uint8))
        for image_id in ["3.110-1", "4.070-2"]:
            for suffix in [
                "tissue_mask_overlay",
                "orientation",
                "coherence",
                "oop_heatmap",
                "spacing_heatmap",
            ]:
                image.save(previews / f"{image_id}_{suffix}.png")


def test_gallery_index_builds_from_synthetic_outputs(tmp_path: Path) -> None:
    write_synthetic_gallery_inputs(tmp_path, with_previews=True)
    index = build_qc_gallery_index(gallery_config(tmp_path))
    assert list(index.columns) == GALLERY_COLUMNS
    assert len(index) == 2
    assert set(index["status"]) == {"ok"}


def test_missing_previews_recorded_without_crashing(tmp_path: Path) -> None:
    write_synthetic_gallery_inputs(tmp_path, with_previews=False)
    index = build_qc_gallery_index(gallery_config(tmp_path))
    assert index["qc_flag_summary"].str.startswith("missing_previews").all()
    assert (index["tissue_mask_overlay_path"] == "").all()


def test_require_existing_previews_fails_when_missing(tmp_path: Path) -> None:
    write_synthetic_gallery_inputs(tmp_path, with_previews=False)
    with pytest.raises(FileNotFoundError):
        build_qc_gallery_index(gallery_config(tmp_path), require_existing_previews=True)


def test_gallery_html_and_index_are_created_under_results(tmp_path: Path) -> None:
    write_synthetic_gallery_inputs(tmp_path, with_previews=True)
    cfg = gallery_config(tmp_path)
    index = build_qc_gallery_index(cfg)
    index_path = write_gallery_index(index, cfg)
    html_path = write_gallery_html(index, cfg)
    assert index_path.exists()
    assert html_path.exists()
    assert str(index_path).startswith(str(tmp_path / "results"))
    assert str(html_path).startswith(str(tmp_path / "results"))


def test_image_and_donor_ids_are_preserved_as_strings(tmp_path: Path) -> None:
    write_synthetic_gallery_inputs(tmp_path, with_previews=True)
    index = build_qc_gallery_index(gallery_config(tmp_path), sort_by="image_id", ascending=True)
    assert index.loc[0, "image_id"] == "3.110-1"
    assert index.loc[0, "donor_id"] == "3.110"
