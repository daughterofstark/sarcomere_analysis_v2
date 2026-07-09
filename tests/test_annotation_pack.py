from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from sarcomere_analysis.annotation_pack import (
    ANNOTATION_TEMPLATE_COLUMNS,
    annotation_template_from_index,
    export_annotation_pack,
    select_annotation_patches,
    write_annotation_pack_outputs,
)


def annotation_config(tmp_path: Path) -> dict:
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


def synthetic_candidates(include_invalid: bool = True) -> pd.DataFrame:
    rows = []
    donors = ["2.007", "3.032", "4.083", "5.003"]
    oop_values = [0.1, 0.2, 0.45, 0.55, 0.75, 0.9]
    for idx in range(48):
        donor = donors[idx % len(donors)]
        image = f"{donor}-{(idx % 3) + 1}"
        rows.append(
            {
                "image_id": image,
                "donor_id": donor,
                "patch_id": f"{image}_p{idx:05d}",
                "x0": 0,
                "y0": 0,
                "x1": 16,
                "y1": 16,
                "patch_oop": oop_values[idx % len(oop_values)],
                "patch_mean_orientation_deg": float(idx % 180),
                "valid_for_orientation": True,
                "image_path": f"/tmp/{image}.tif",
            }
        )
    if include_invalid:
        for idx in range(4):
            donor = donors[idx]
            image = f"{donor}-invalid"
            rows.append(
                {
                    "image_id": image,
                    "donor_id": donor,
                    "patch_id": f"{image}_p_invalid",
                    "x0": 0,
                    "y0": 0,
                    "x1": 16,
                    "y1": 16,
                    "patch_oop": np.nan,
                    "patch_mean_orientation_deg": np.nan,
                    "valid_for_orientation": False,
                    "image_path": f"/tmp/{image}.tif",
                }
            )
    return pd.DataFrame(rows)


def test_sampling_is_deterministic() -> None:
    candidates = synthetic_candidates()
    first, _ = select_annotation_patches(candidates, n_patches=20, seed=7)
    second, _ = select_annotation_patches(candidates, n_patches=20, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_max_total_patches_respected() -> None:
    selected, summary = select_annotation_patches(synthetic_candidates(), n_patches=12, seed=1)
    assert len(selected) == 12
    assert summary["selected_patches"] == 12


def test_selected_patches_preserve_image_and_donor_ids_as_strings() -> None:
    selected, _ = select_annotation_patches(synthetic_candidates(), n_patches=12, seed=1)
    assert selected["image_id"].map(type).eq(str).all()
    assert selected["donor_id"].map(type).eq(str).all()


def test_required_template_columns_exist() -> None:
    selected, _ = select_annotation_patches(synthetic_candidates(), n_patches=5, seed=1)
    template = annotation_template_from_index(selected)
    assert list(template.columns) == ANNOTATION_TEMPLATE_COLUMNS


def test_invalid_low_quality_patches_optional_and_absent_does_not_crash() -> None:
    selected, summary = select_annotation_patches(synthetic_candidates(include_invalid=False), n_patches=10, seed=3)
    assert len(selected) == 10
    assert summary["invalid_control_selected"] == 0


def test_summary_json_serializable(tmp_path: Path) -> None:
    selected, summary = select_annotation_patches(synthetic_candidates(), n_patches=10, seed=2)
    template = annotation_template_from_index(selected)
    paths = write_annotation_pack_outputs(selected, template, summary, tmp_path)
    loaded = json.loads(paths["annotation_summary"].read_text(encoding="utf-8"))
    assert loaded["selected_patches"] == 10


def test_no_input_tables_are_modified() -> None:
    candidates = synthetic_candidates()
    before = candidates.copy(deep=True)
    select_annotation_patches(candidates, n_patches=10, seed=4)
    pd.testing.assert_frame_equal(candidates, before)


def test_crop_export_works_on_tiny_synthetic_image(tmp_path: Path) -> None:
    cfg = annotation_config(tmp_path)
    tables = tmp_path / "results" / "tables"
    raw = tmp_path / "raw"
    tables.mkdir(parents=True)
    raw.mkdir()
    image_path = raw / "2.007-1.tif"
    tifffile.imwrite(image_path, np.arange(32 * 32, dtype=np.uint16).reshape(32, 32))

    patch = pd.DataFrame(
        [
            {
                "image_id": "2.007-1",
                "donor_id": "2.007",
                "patch_id": "2.007-1_p00000",
                "x0": 0,
                "y0": 0,
                "x1": 16,
                "y1": 16,
                "patch_oop": 0.8,
                "patch_mean_orientation_deg": 45.0,
                "valid_for_orientation": True,
            }
        ]
    )
    patch.to_csv(tables / "features_per_patch.csv", index=False)
    pd.DataFrame([{"image_id": "2.007-1", "donor_id": "2.007"}]).to_csv(tables / "features_per_image.csv", index=False)
    pd.DataFrame([{"image_id": "2.007-1", "donor_id": "2.007"}]).to_csv(tables / "analysis_per_image.csv", index=False)
    pd.DataFrame([{"image_id": "2.007-1", "donor_id": "2.007", "image_path": str(image_path)}]).to_csv(
        tables / "enriched_manifest.csv",
        index=False,
    )

    index, template, summary, paths = export_annotation_pack(cfg, n_patches=1, seed=1, overwrite=True)

    assert len(index) == 1
    assert len(template) == 1
    assert summary["crop_count"] == 1
    assert paths["annotation_index"].exists()
    crop_path = Path(index.loc[0, "crop_path"])
    assert crop_path.exists()
    assert crop_path.suffix == ".png"
