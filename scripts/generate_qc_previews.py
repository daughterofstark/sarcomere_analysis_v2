#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config, manifest_csv_path
from sarcomere_analysis.io import build_manifest
from sarcomere_analysis.outputs import ensure_output_dirs, preview_paths
from sarcomere_analysis.pipeline import WriteOptions, run_single_image


PREVIEW_KEYS = [
    "tissue_mask_overlay",
    "orientation",
    "coherence",
    "oop_heatmap",
    "spacing_heatmap",
]

SUMMARY_COLUMNS = [
    "image_id",
    "donor_id",
    "status",
    "error_message",
    "runtime_seconds",
    "tissue_mask_overlay_path",
    "orientation_preview_path",
    "coherence_preview_path",
    "oop_heatmap_path",
    "spacing_heatmap_path",
    "generated_count",
    "skipped_existing_count",
]

PATH_COLUMN_BY_KEY = {
    "tissue_mask_overlay": "tissue_mask_overlay_path",
    "orientation": "orientation_preview_path",
    "coherence": "coherence_preview_path",
    "oop_heatmap": "oop_heatmap_path",
    "spacing_heatmap": "spacing_heatmap_path",
}


def parse_image_id_subset(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    image_ids: set[str] = set()
    for value in values:
        image_ids.update(item.strip() for item in value.split(",") if item.strip())
    return image_ids


def load_manifest(cfg: dict, manifest_override: str | None) -> pd.DataFrame:
    path = Path(manifest_override) if manifest_override else manifest_csv_path(cfg)
    if path.exists():
        return pd.read_csv(path, dtype={"image_id": str, "donor_id": str, "region_id": str})
    return build_manifest(cfg)


def select_manifest_rows(manifest: pd.DataFrame, image_ids: set[str] | None, limit: int | None) -> pd.DataFrame:
    selected = manifest.copy()
    if image_ids is not None:
        selected = selected.loc[selected["image_id"].isin(image_ids)].copy()
    if limit is not None:
        selected = selected.head(limit).copy()
    return selected.reset_index(drop=True)


def expected_preview_paths(image_id: str, cfg: dict) -> dict[str, Path]:
    return {key: preview_paths(image_id, cfg)[key] for key in PREVIEW_KEYS}


def summary_path(cfg: dict) -> Path:
    return ensure_output_dirs(cfg)["tables"] / "qc_preview_generation_summary.csv"


def write_summary(rows: list[dict[str, object]], cfg: dict) -> Path:
    path = summary_path(cfg)
    df = pd.DataFrame(rows)
    for column in SUMMARY_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df[SUMMARY_COLUMNS]
    df.to_csv(path, index=False)
    return path


def _base_summary(row: pd.Series, cfg: dict) -> dict[str, object]:
    image_id = str(row["image_id"])
    donor_id = str(row["donor_id"]) if "donor_id" in row and pd.notna(row["donor_id"]) else ""
    paths = expected_preview_paths(image_id, cfg)
    summary: dict[str, object] = {
        "image_id": image_id,
        "donor_id": donor_id,
        "status": "",
        "error_message": "",
        "runtime_seconds": 0.0,
        "generated_count": 0,
        "skipped_existing_count": 0,
    }
    for key, column in PATH_COLUMN_BY_KEY.items():
        summary[column] = str(paths[key])
    return summary


def process_preview_row(row: pd.Series, cfg: dict, overwrite: bool, config_path: str) -> dict[str, object]:
    started = perf_counter()
    summary = _base_summary(row, cfg)
    image_id = str(summary["image_id"])
    donor_id = str(summary["donor_id"]) if summary["donor_id"] != "" else None
    paths = expected_preview_paths(image_id, cfg)
    existing_before = {key: path.exists() for key, path in paths.items()}

    if not overwrite and all(existing_before.values()):
        summary["status"] = "ok"
        summary["runtime_seconds"] = perf_counter() - started
        summary["skipped_existing_count"] = len(PREVIEW_KEYS)
        return summary

    try:
        result = run_single_image(
            image_path=Path(str(row["image_path"])),
            image_id=image_id,
            donor_id=donor_id,
            cfg=cfg,
            write_options=WriteOptions(preview=True, overwrite_previews=overwrite),
            config_path=config_path,
        )
        summary["status"] = "ok"
        summary["runtime_seconds"] = result.runtime_seconds
        summary["generated_count"] = sum(1 for key in PREVIEW_KEYS if key in result.output_paths)
        summary["skipped_existing_count"] = 0 if overwrite else sum(1 for exists in existing_before.values() if exists)
    except Exception as exc:
        summary["status"] = "error"
        summary["error_message"] = str(exc)
        summary["runtime_seconds"] = perf_counter() - started
        summary["skipped_existing_count"] = 0 if overwrite else sum(1 for exists in existing_before.values() if exists)
    return summary


def run_preview_generation(
    cfg: dict,
    config_path: str,
    manifest_override: str | None = None,
    image_ids: set[str] | None = None,
    limit: int | None = None,
    workers: int = 1,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> tuple[pd.DataFrame, Path]:
    ensure_output_dirs(cfg)
    manifest = load_manifest(cfg, manifest_override)
    selected = select_manifest_rows(manifest, image_ids, limit)
    if selected.empty:
        raise SystemExit("No images selected for preview generation.")

    rows: list[dict[str, object]] = []
    workers = max(1, int(workers))
    if workers == 1:
        iterator = [process_preview_row(row, cfg, overwrite, config_path) for _, row in selected.iterrows()]
    else:
        iterator = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_preview_row, row, cfg, overwrite, config_path) for _, row in selected.iterrows()]
            for future in as_completed(futures):
                iterator.append(future.result())

    for row in iterator:
        rows.append(row)
        if row["status"] == "error" and not continue_on_error:
            path = write_summary(rows, cfg)
            raise SystemExit(f"{row['error_message']}\nWrote partial summary: {path}")

    path = write_summary(rows, cfg)
    return pd.DataFrame(rows)[SUMMARY_COLUMNS], path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate missing QC preview PNGs without rewriting metric tables.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", help="Manifest CSV override.")
    parser.add_argument("--image-id", action="append", help="Optional image id subset; may be repeated or comma-separated.")
    parser.add_argument("--limit", type=int, help="Limit number of selected images.")
    parser.add_argument("--workers", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--missing-only", action="store_true", help="Skip existing previews. This is the default.")
    mode.add_argument("--overwrite", action="store_true", help="Regenerate previews even if they already exist.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary, path = run_preview_generation(
        cfg=cfg,
        config_path=args.config,
        manifest_override=args.manifest,
        image_ids=parse_image_id_subset(args.image_id),
        limit=args.limit,
        workers=args.workers,
        overwrite=bool(args.overwrite),
        continue_on_error=bool(args.continue_on_error),
    )
    ok_count = int((summary["status"] == "ok").sum())
    error_count = int((summary["status"] == "error").sum())
    print(f"processed_images: {len(summary)}")
    print(f"ok: {ok_count}")
    print(f"errors: {error_count}")
    print(f"generated_previews: {int(summary['generated_count'].sum())}")
    print(f"skipped_existing_previews: {int(summary['skipped_existing_count'].sum())}")
    print(f"Wrote qc_preview_generation_summary: {path}")


if __name__ == "__main__":
    main()
