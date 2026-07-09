from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tifffile

from sarcomere_analysis.full_image_annotation import (
    audit_full_image_annotations,
    headless_check_full_image_annotations,
    prepare_full_image_annotation_set,
    select_full_images,
)


def full_config(tmp_path: Path) -> dict:
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
            "output_dtype": "float32",
        },
    }


def make_full_inputs(tmp_path: Path, n: int = 9) -> tuple[dict, Path, Path, Path]:
    cfg = full_config(tmp_path)
    raw_dir = tmp_path / "raw"
    tables = tmp_path / "results" / "tables"
    raw_dir.mkdir(parents=True)
    tables.mkdir(parents=True)
    rows = []
    feature_rows = []
    manifest_rows = []
    for idx in range(n):
        donor = f"{2 + idx % 4}.{idx + 7:03d}"
        image_id = f"{donor}-{idx + 1}"
        image_path = raw_dir / f"{image_id}.tif"
        image = np.arange(16 * 12, dtype=np.uint16).reshape(16, 12) + idx
        tifffile.imwrite(image_path, image)
        rows.append(
            {
                "image_id": image_id,
                "donor_id": donor,
                "image_oop": idx / max(n - 1, 1),
                "orientation_valid_fraction": 0.5,
                "status": "ok",
                "image_path": str(image_path),
            }
        )
        feature_rows.append({"image_id": image_id, "donor_id": donor, "image_oop": idx / max(n - 1, 1)})
        manifest_rows.append({"image_id": image_id, "donor_id": donor, "image_path": str(image_path)})
    analysis_path = tables / "analysis_per_image.csv"
    feature_path = tables / "features_per_image.csv"
    manifest_path = tables / "enriched_manifest.csv"
    pd.DataFrame(rows).to_csv(analysis_path, index=False)
    pd.DataFrame(feature_rows).to_csv(feature_path, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return cfg, analysis_path, feature_path, manifest_path


def test_deterministic_full_image_selection(tmp_path: Path) -> None:
    cfg, analysis_path, _, _ = make_full_inputs(tmp_path, n=9)
    table = pd.read_csv(analysis_path, dtype={"image_id": str, "donor_id": str})
    _ = cfg

    first = select_full_images(table, n_images=6, seed=10)
    second = select_full_images(table, n_images=6, seed=10)

    assert first["image_id"].tolist() == second["image_id"].tolist()
    assert len(first) == 6


def test_masks_match_image_shape(tmp_path: Path) -> None:
    cfg, analysis_path, feature_path, manifest_path = make_full_inputs(tmp_path, n=6)

    index, summary, _ = prepare_full_image_annotation_set(
        cfg,
        n_images=3,
        seed=1,
        analysis_table=analysis_path,
        feature_table=feature_path,
        manifest_table=manifest_path,
        overwrite=True,
    )

    assert summary["selected_images"] == 3
    for _, row in index.iterrows():
        image = np.asarray(Image.open(row["annotation_image_path"]))
        mask = np.asarray(Image.open(row["mask_path"]))
        assert mask.shape == image.shape[:2]


def test_label_validation_0_1_2(tmp_path: Path) -> None:
    cfg, analysis_path, feature_path, manifest_path = make_full_inputs(tmp_path, n=6)
    index, _, _ = prepare_full_image_annotation_set(
        cfg,
        n_images=2,
        seed=1,
        analysis_table=analysis_path,
        feature_table=feature_path,
        manifest_table=manifest_path,
        overwrite=True,
    )
    mask = np.zeros((16, 12), dtype=np.uint8)
    mask[1, 1] = 1
    mask[2, 2] = 2
    Image.fromarray(mask).save(index.loc[0, "mask_path"])

    _, summary, _ = audit_full_image_annotations(cfg)

    assert summary["invalid_label_masks"] == 0
    assert summary["masks_with_zdisc_labels"] == 1


def test_audit_detects_missing_and_shape_mismatch(tmp_path: Path) -> None:
    cfg, analysis_path, feature_path, manifest_path = make_full_inputs(tmp_path, n=6)
    index, _, _ = prepare_full_image_annotation_set(
        cfg,
        n_images=3,
        seed=1,
        analysis_table=analysis_path,
        feature_table=feature_path,
        manifest_table=manifest_path,
        overwrite=True,
    )
    Path(index.loc[0, "mask_path"]).unlink()
    Image.fromarray(np.zeros((5, 5), dtype=np.uint8)).save(index.loc[1, "mask_path"])

    _, summary, _ = audit_full_image_annotations(cfg)

    assert summary["missing_masks"] == 1
    assert summary["shape_mismatch_masks"] >= 1


def test_summary_json_serializable(tmp_path: Path) -> None:
    cfg, analysis_path, feature_path, manifest_path = make_full_inputs(tmp_path, n=6)

    _, _, paths = prepare_full_image_annotation_set(
        cfg,
        n_images=3,
        seed=1,
        analysis_table=analysis_path,
        feature_table=feature_path,
        manifest_table=manifest_path,
        overwrite=True,
    )
    loaded = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    assert loaded["selected_images"] == 3


def test_no_production_tables_modified(tmp_path: Path) -> None:
    cfg, analysis_path, feature_path, manifest_path = make_full_inputs(tmp_path, n=6)
    before_analysis = analysis_path.read_bytes()
    before_features = feature_path.read_bytes()

    prepare_full_image_annotation_set(
        cfg,
        n_images=3,
        seed=1,
        analysis_table=analysis_path,
        feature_table=feature_path,
        manifest_table=manifest_path,
        overwrite=True,
    )
    audit_full_image_annotations(cfg)

    assert analysis_path.read_bytes() == before_analysis
    assert feature_path.read_bytes() == before_features


def test_drawing_backend_can_headless_check_paths(tmp_path: Path) -> None:
    cfg, analysis_path, feature_path, manifest_path = make_full_inputs(tmp_path, n=6)
    prepare_full_image_annotation_set(
        cfg,
        n_images=3,
        seed=1,
        analysis_table=analysis_path,
        feature_table=feature_path,
        manifest_table=manifest_path,
        overwrite=True,
    )

    summary = headless_check_full_image_annotations(cfg)

    assert summary["rows"] == 3
    assert summary["missing_image_count"] == 0
    assert summary["missing_mask_count"] == 0
    assert summary["shape_mismatch_count"] == 0
