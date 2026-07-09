from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sarcomere_analysis.full_image_annotation_features import (
    extract_full_image_zdisc_annotation_features,
    extract_one_full_image_mask_features,
)
from sarcomere_analysis.zdisc_annotation_features import estimate_mask_orientation


def feature_config(tmp_path: Path) -> dict:
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


def make_full_feature_set(tmp_path: Path, masks: list[np.ndarray]) -> tuple[dict, Path]:
    cfg = feature_config(tmp_path)
    root = tmp_path / "results" / "full_image_zdisc_annotation"
    images = root / "images"
    mask_dir = root / "masks"
    overlays = root / "overlays"
    for directory in [images, mask_dir, overlays]:
        directory.mkdir(parents=True)
    rows = []
    for idx, mask in enumerate(masks, start=1):
        annotation_id = f"FULL_{idx:04d}"
        image_id = f"2.007-{idx}"
        image_path = images / f"{annotation_id}__{image_id}.png"
        mask_path = mask_dir / f"{annotation_id}__{image_id}_mask.png"
        Image.fromarray(np.full(mask.shape, 100, dtype=np.uint8)).save(image_path)
        Image.fromarray(mask.astype(np.uint8)).save(mask_path)
        rows.append(
            {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "donor_id": "2.007",
                "patch_id": image_id,
                "annotation_image_path": str(image_path),
                "mask_path": str(mask_path),
                "overlay_path": str(overlays / f"{annotation_id}_overlay.png"),
            }
        )
    index_path = root / "full_image_annotation_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return cfg, index_path


def test_empty_mask_returns_annotation_status_empty(tmp_path: Path) -> None:
    cfg, index_path = make_full_feature_set(tmp_path, [np.zeros((10, 10), dtype=np.uint8)])

    features, summary, _ = extract_full_image_zdisc_annotation_features(cfg, index_path=index_path)

    assert features.loc[0, "annotation_status"] == "empty"
    assert summary["empty_masks"] == 1


def test_label_1_mask_returns_has_zdisc_labels_true(tmp_path: Path) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    cfg, index_path = make_full_feature_set(tmp_path, [mask])

    features, _, _ = extract_full_image_zdisc_annotation_features(cfg, index_path=index_path)

    assert bool(features.loc[0, "has_zdisc_labels"])
    assert features.loc[0, "annotation_status"] == "zdisc_labeled"


def test_label_2_only_mask_returns_ignore_only(tmp_path: Path) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 2
    cfg, index_path = make_full_feature_set(tmp_path, [mask])

    features, _, _ = extract_full_image_zdisc_annotation_features(cfg, index_path=index_path)

    assert features.loc[0, "annotation_status"] == "ignore_only"
    assert bool(features.loc[0, "has_ignore_labels"])
    assert not bool(features.loc[0, "has_zdisc_labels"])


def test_mixed_mask_returns_mixed(tmp_path: Path) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    mask[6:8, 6:8] = 2
    cfg, index_path = make_full_feature_set(tmp_path, [mask])

    features, _, _ = extract_full_image_zdisc_annotation_features(cfg, index_path=index_path)

    assert features.loc[0, "annotation_status"] == "mixed"


def test_connected_components_counted_correctly(tmp_path: Path) -> None:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[8:10, 8:10] = 1
    cfg, index_path = make_full_feature_set(tmp_path, [mask])

    features, _, _ = extract_full_image_zdisc_annotation_features(cfg, index_path=index_path, min_zdisc_pixels=1)

    assert int(features.loc[0, "zdisc_component_count"]) == 2
    assert float(features.loc[0, "median_component_size"]) == 4.0


def test_orientation_for_synthetic_diagonal_line_is_finite_and_plausible() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    for i in range(4, 16):
        mask[i, i] = True

    result = estimate_mask_orientation(mask, component_count=1, min_zdisc_pixels=5, min_components=1)

    assert result["orientation_estimable"]
    assert 35.0 <= result["manual_mask_orientation_deg"] <= 55.0


def test_orientation_returns_nan_for_too_few_pixels() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True

    result = estimate_mask_orientation(mask, component_count=1, min_zdisc_pixels=5, min_components=1)

    assert not result["orientation_estimable"]
    assert np.isnan(result["manual_mask_orientation_deg"])


def test_summary_json_is_serializable(tmp_path: Path) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    cfg, index_path = make_full_feature_set(tmp_path, [mask])

    _, _, paths = extract_full_image_zdisc_annotation_features(cfg, index_path=index_path)
    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    assert loaded["mask_count"] == 1


def test_input_masks_are_not_modified(tmp_path: Path) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    cfg, index_path = make_full_feature_set(tmp_path, [mask])
    row = pd.read_csv(index_path, dtype=str).iloc[0]
    mask_path = Path(row["mask_path"])
    before = mask_path.read_bytes()

    extract_full_image_zdisc_annotation_features(cfg, index_path=index_path)

    assert mask_path.read_bytes() == before


def test_extract_one_preserves_identifiers_as_strings(tmp_path: Path) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    cfg, index_path = make_full_feature_set(tmp_path, [mask])
    _ = cfg
    row = pd.read_csv(index_path, dtype=str).iloc[0]

    features = extract_one_full_image_mask_features(row)

    assert isinstance(features["image_id"], str)
    assert isinstance(features["donor_id"], str)
    assert features["donor_id"] == "2.007"
