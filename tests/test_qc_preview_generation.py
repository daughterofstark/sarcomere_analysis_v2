from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
from PIL import Image

from scripts.generate_qc_previews import (
    SUMMARY_COLUMNS,
    parse_image_id_subset,
    run_preview_generation,
)
from test_step_8_cli_batch import script_path, write_step8_config, write_synthetic_tiffs


def read_config(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def preview_files(results_dir: Path, image_id: str) -> list[Path]:
    previews = results_dir / "previews"
    return [
        previews / f"{image_id}_tissue_mask_overlay.png",
        previews / f"{image_id}_orientation.png",
        previews / f"{image_id}_coherence.png",
        previews / f"{image_id}_oop_heatmap.png",
        previews / f"{image_id}_spacing_heatmap.png",
    ]


def test_missing_only_skips_existing_preview_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=1)
    config_path = write_step8_config(tmp_path, raw_dir)
    cfg = read_config(config_path)

    first, _ = run_preview_generation(cfg, str(config_path), limit=1)
    second, _ = run_preview_generation(cfg, str(config_path), limit=1)

    assert int(first.loc[0, "generated_count"]) == 5
    assert int(second.loc[0, "generated_count"]) == 0
    assert int(second.loc[0, "skipped_existing_count"]) == 5


def test_overwrite_regenerates_existing_preview_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=1)
    config_path = write_step8_config(tmp_path, raw_dir)
    cfg = read_config(config_path)
    run_preview_generation(cfg, str(config_path), limit=1)

    broken_preview = preview_files(tmp_path / "results", "2.001-1")[0]
    broken_preview.write_text("not a png", encoding="utf-8")
    summary, _ = run_preview_generation(cfg, str(config_path), limit=1, overwrite=True)

    assert int(summary.loc[0, "generated_count"]) == 5
    assert int(summary.loc[0, "skipped_existing_count"]) == 0
    with Image.open(broken_preview) as image:
        assert image.size == (64, 64)


def test_summary_csv_has_stable_schema_and_string_ids(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=1)
    config_path = write_step8_config(tmp_path, raw_dir)
    cfg = read_config(config_path)
    summary, path = run_preview_generation(cfg, str(config_path), limit=1)

    assert list(summary.columns) == SUMMARY_COLUMNS
    on_disk = pd.read_csv(path, dtype={"image_id": str, "donor_id": str})
    assert list(on_disk.columns) == SUMMARY_COLUMNS
    assert on_disk.loc[0, "image_id"] == "2.001-1"
    assert on_disk.loc[0, "donor_id"] == "2.001"


def test_limit_restricts_number_processed(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=3)
    config_path = write_step8_config(tmp_path, raw_dir)
    cfg = read_config(config_path)
    summary, _ = run_preview_generation(cfg, str(config_path), limit=2)

    assert len(summary) == 2


def test_continue_on_error_records_failures(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=1)
    config_path = write_step8_config(tmp_path, raw_dir)
    cfg = read_config(config_path)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"image_id": "2.001-1", "donor_id": "2.001", "image_path": str(raw_dir / "2.001-1.tif")},
            {"image_id": "3.110-1", "donor_id": "3.110", "image_path": str(raw_dir / "missing.tif")},
        ]
    ).to_csv(manifest_path, index=False)

    summary, _ = run_preview_generation(
        cfg,
        str(config_path),
        manifest_override=str(manifest_path),
        continue_on_error=True,
    )

    assert set(summary["status"]) == {"ok", "error"}
    assert summary.loc[summary["status"] == "error", "donor_id"].iloc[0] == "3.110"


def test_generated_paths_stay_under_results_previews(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=1)
    config_path = write_step8_config(tmp_path, raw_dir)
    cfg = read_config(config_path)
    summary, _ = run_preview_generation(cfg, str(config_path), limit=1)

    path_columns = [column for column in SUMMARY_COLUMNS if column.endswith("_path")]
    for column in path_columns:
        path = Path(str(summary.loc[0, column]))
        assert path.is_relative_to(tmp_path / "results" / "previews")


def test_script_works_on_synthetic_tiff_manifest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_tiffs(raw_dir, n=2)
    config_path = write_step8_config(tmp_path, raw_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path("generate_qc_previews.py")),
            "--config",
            str(config_path),
            "--image-id",
            "2.001-1,2.002-2",
            "--limit",
            "1",
            "--continue-on-error",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "processed_images: 1" in completed.stdout
    assert (tmp_path / "results" / "tables" / "qc_preview_generation_summary.csv").exists()
    assert len(parse_image_id_subset(["2.001-1,2.002-2"])) == 2
