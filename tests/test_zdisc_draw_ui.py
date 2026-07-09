from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from sarcomere_analysis.zdisc_draw_ui import (
    headless_check,
    load_draw_index,
    load_mask,
    paint_mask,
    save_mask,
    write_progress,
)


def draw_config(tmp_path: Path) -> dict:
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


def make_draw_set(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    cfg = draw_config(tmp_path)
    root = tmp_path / "results" / "zdisc_annotation"
    images = root / "images"
    masks = root / "masks"
    overlays = root / "overlays"
    for directory in [images, masks, overlays]:
        directory.mkdir(parents=True)
    rows = []
    for idx in range(2):
        annotation_id = f"ANN_{idx + 1:04d}"
        image_path = images / f"{annotation_id}.png"
        mask_path = masks / f"{annotation_id}_mask.png"
        overlay_path = overlays / f"{annotation_id}_overlay.png"
        Image.fromarray(np.full((9, 11), 60 + idx, dtype=np.uint8)).save(image_path)
        Image.fromarray(np.zeros((9, 11), dtype=np.uint8)).save(mask_path)
        rows.append(
            {
                "annotation_id": annotation_id,
                "image_id": f"2.007-{idx + 1}",
                "donor_id": "2.007",
                "patch_id": f"2.007-{idx + 1}_p0000{idx}",
                "annotation_image_path": str(image_path),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
        )
    index_path = root / "zdisc_annotation_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return cfg, index_path, images, masks


def test_loading_index_works(tmp_path: Path) -> None:
    _, index_path, _, _ = make_draw_set(tmp_path)

    index = load_draw_index(index_path)

    assert len(index) == 2
    assert index.loc[0, "annotation_id"] == "ANN_0001"
    assert index.loc[0, "donor_id"] == "2.007"


def test_mask_image_shape_validation_works(tmp_path: Path) -> None:
    cfg, index_path, _, masks = make_draw_set(tmp_path)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(masks / "ANN_0001_mask.png")

    summary = headless_check(cfg, index_path=index_path)

    assert summary["shape_mismatch_count"] == 1


def test_brush_painting_label_1_modifies_expected_pixels() -> None:
    mask = np.zeros((9, 9), dtype=np.uint8)

    painted = paint_mask(mask, x=4, y=4, label=1, radius=1)

    assert painted[4, 4] == 1
    assert painted[4, 3] == 1
    assert painted[3, 4] == 1
    assert painted[0, 0] == 0


def test_eraser_sets_pixels_back_to_0() -> None:
    mask = np.ones((9, 9), dtype=np.uint8)

    erased = paint_mask(mask, x=4, y=4, label=0, radius=1)

    assert erased[4, 4] == 0
    assert erased[4, 3] == 0
    assert erased[0, 0] == 1


def test_label_2_painting_works() -> None:
    mask = np.zeros((9, 9), dtype=np.uint8)

    painted = paint_mask(mask, x=2, y=2, label=2, radius=2)

    assert painted[2, 2] == 2
    assert set(np.unique(painted)).issubset({0, 2})


def test_255_values_are_interpreted_as_label_1_for_loading(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    raw = np.zeros((5, 5), dtype=np.uint8)
    raw[1, 1] = 255
    Image.fromarray(raw).save(mask_path)

    mask = load_mask(mask_path, expected_shape=(5, 5))

    assert mask[1, 1] == 1


def test_save_mask_writes_only_labels_0_1_2(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = 255
    mask[2, 2] = 2

    save_mask(mask, mask_path)
    loaded = np.asarray(Image.open(mask_path))

    assert set(int(value) for value in np.unique(loaded)).issubset({0, 1, 2})
    assert loaded[1, 1] == 1


def test_invalid_mask_labels_fail(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = 7
    Image.fromarray(mask).save(mask_path)

    with pytest.raises(ValueError, match="unsupported labels"):
        load_mask(mask_path)


def test_progress_json_is_serializable(tmp_path: Path) -> None:
    _, index_path, _, _ = make_draw_set(tmp_path)
    index = load_draw_index(index_path)
    progress_path = tmp_path / "progress.json"

    write_progress(progress_path, index, position=1, current_label=2, brush_radius=3)
    loaded = json.loads(progress_path.read_text(encoding="utf-8"))

    assert loaded["annotation_id"] == "ANN_0002"
    assert loaded["current_label"] == 2
    assert loaded["brush_radius"] == 3


def test_headless_check_does_not_modify_masks(tmp_path: Path) -> None:
    cfg, index_path, _, masks = make_draw_set(tmp_path)
    mask_path = masks / "ANN_0001_mask.png"
    before = mask_path.read_bytes()

    summary = headless_check(cfg, index_path=index_path)

    assert summary["shape_mismatch_count"] == 0
    assert mask_path.read_bytes() == before


def test_no_production_feature_or_analysis_tables_are_modified(tmp_path: Path) -> None:
    cfg, index_path, _, _ = make_draw_set(tmp_path)
    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    feature_path = tables / "features_per_image.csv"
    analysis_path = tables / "analysis_per_image.csv"
    feature_path.write_text("image_id,donor_id,image_oop\n2.007-1,2.007,0.5\n", encoding="utf-8")
    analysis_path.write_text("image_id,donor_id,is_healthy\n2.007-1,2.007,False\n", encoding="utf-8")
    before_feature = feature_path.read_bytes()
    before_analysis = analysis_path.read_bytes()

    headless_check(cfg, index_path=index_path)

    assert feature_path.read_bytes() == before_feature
    assert analysis_path.read_bytes() == before_analysis
