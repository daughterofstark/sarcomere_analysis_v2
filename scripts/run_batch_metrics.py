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
from sarcomere_analysis.outputs import ensure_output_dirs
from sarcomere_analysis.pipeline import WriteOptions, run_single_image
from sarcomere_analysis.schemas import (
    BATCH_RUN_SUMMARY_COLUMNS,
    IMAGE_METRICS_COLUMNS,
    PATCH_METRICS_COLUMNS,
    stabilize_columns,
)


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


def process_row(row: pd.Series, cfg: dict, options: WriteOptions, config_path: str) -> tuple[dict[str, object], pd.DataFrame | None, dict[str, object] | None]:
    started = perf_counter()
    image_id = str(row["image_id"])
    donor_id = str(row["donor_id"]) if "donor_id" in row and pd.notna(row["donor_id"]) else None
    try:
        result = run_single_image(
            image_path=Path(str(row["image_path"])),
            image_id=image_id,
            donor_id=donor_id,
            cfg=cfg,
            write_options=options,
            config_path=config_path,
        )
        summary = {
            "image_id": image_id,
            "donor_id": donor_id,
            "status": "ok",
            "error_message": "",
            "runtime_seconds": result.runtime_seconds,
            "per_patch_metrics_path": result.output_paths.get("per_patch_metrics", ""),
            "per_image_metrics_path": result.output_paths.get("per_image_metrics", ""),
            "provenance_path": result.output_paths.get("run_provenance", ""),
        }
        return summary, result.patch_metrics, result.image_metrics
    except Exception as exc:
        summary = {
            "image_id": image_id,
            "donor_id": donor_id,
            "status": "error",
            "error_message": str(exc),
            "runtime_seconds": perf_counter() - started,
            "per_patch_metrics_path": "",
            "per_image_metrics_path": "",
            "provenance_path": "",
        }
        return summary, None, None


def write_combined_outputs(
    cfg: dict,
    patch_tables: list[pd.DataFrame],
    image_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> dict[str, Path]:
    tables_dir = ensure_output_dirs(cfg)["tables"]
    paths = {
        "combined_per_image": tables_dir / "per_image_metrics.csv",
        "combined_per_patch": tables_dir / "per_patch_metrics.csv",
        "batch_summary": tables_dir / "batch_run_summary.csv",
    }
    image_df = stabilize_columns(pd.DataFrame(image_rows), IMAGE_METRICS_COLUMNS)
    patch_df = stabilize_columns(pd.concat(patch_tables, ignore_index=True) if patch_tables else pd.DataFrame(), PATCH_METRICS_COLUMNS)
    summary_df = stabilize_columns(pd.DataFrame(summary_rows), BATCH_RUN_SUMMARY_COLUMNS)
    image_df.to_csv(paths["combined_per_image"], index=False)
    patch_df.to_csv(paths["combined_per_patch"], index=False)
    summary_df.to_csv(paths["batch_summary"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch image metrics over a manifest.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", help="Manifest CSV override.")
    parser.add_argument("--limit", type=int, help="Limit number of images for smoke testing.")
    parser.add_argument("--image-id", action="append", help="Optional image id subset; may be repeated or comma-separated.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--write-preview", action="store_true")
    parser.add_argument("--write-tables", action="store_true")
    parser.add_argument("--write-provenance", action="store_true")
    parser.add_argument("--write-all", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    manifest = load_manifest(cfg, args.manifest)
    selected = select_manifest_rows(manifest, parse_image_id_subset(args.image_id), args.limit)
    if selected.empty:
        raise SystemExit("No images selected for batch run.")

    options = WriteOptions(
        tables=bool(args.write_tables or args.write_all),
        preview=bool(args.write_preview or args.write_all),
        provenance=bool(args.write_provenance or args.write_all),
    )

    summary_rows: list[dict[str, object]] = []
    patch_tables: list[pd.DataFrame] = []
    image_rows: list[dict[str, object]] = []

    workers = max(1, int(args.workers))
    if workers == 1:
        iterator = [process_row(row, cfg, options, args.config) for _, row in selected.iterrows()]
    else:
        iterator = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_row, row, cfg, options, args.config) for _, row in selected.iterrows()]
            for future in as_completed(futures):
                iterator.append(future.result())

    for summary, patch_metrics, image_metrics in iterator:
        summary_rows.append(summary)
        if summary["status"] == "error" and not args.continue_on_error:
            write_combined_outputs(cfg, patch_tables, image_rows, summary_rows)
            raise SystemExit(str(summary["error_message"]))
        if patch_metrics is not None:
            patch_tables.append(patch_metrics)
        if image_metrics is not None:
            image_rows.append(image_metrics)

    paths = write_combined_outputs(cfg, patch_tables, image_rows, summary_rows)
    ok_count = sum(1 for row in summary_rows if row["status"] == "ok")
    error_count = sum(1 for row in summary_rows if row["status"] == "error")
    print(f"processed_images: {len(summary_rows)}")
    print(f"ok: {ok_count}")
    print(f"errors: {error_count}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
