#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config, output_dir
from sarcomere_analysis.diagnostics.spacing_candidates import (
    diagnose_spacing_candidates_for_manifest,
    load_manifest_for_candidates,
    select_candidate_manifest_rows,
    summarize_spacing_candidates,
    write_spacing_candidate_outputs,
)


def parse_image_ids(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    image_ids: set[str] = set()
    for value in values:
        image_ids.update(item.strip() for item in value.split(",") if item.strip())
    return image_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Write candidate-level spacing diagnostics without changing metrics.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image-id", action="append", help="Image id subset; may be repeated or comma-separated.")
    parser.add_argument("--all", action="store_true", help="Run over all manifest images, optionally limited by --max-images.")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output-dir", help="Output directory. Defaults to config output_dir/diagnostics.")
    parser.add_argument("--compare-main-table", action="store_true")
    args = parser.parse_args()

    image_ids = parse_image_ids(args.image_id)
    if not args.all and not image_ids:
        parser.error("Pass --image-id or --all. Full dataset is never run by default.")

    cfg = load_config(args.config)
    manifest = load_manifest_for_candidates(cfg)
    selected = select_candidate_manifest_rows(manifest, image_ids, args.all, args.max_images)
    if selected.empty:
        raise SystemExit("No images selected for candidate diagnostics.")

    started = perf_counter()
    candidates = diagnose_spacing_candidates_for_manifest(cfg, selected)
    main_patch = None
    if args.compare_main_table:
        main_path = output_dir(cfg) / "tables" / "per_patch_metrics.csv"
        if main_path.exists():
            main_patch = pd.read_csv(main_path, dtype={"image_id": str, "donor_id": str, "patch_id": str})
    summary = summarize_spacing_candidates(candidates, main_patch)
    out_dir = Path(args.output_dir) if args.output_dir else output_dir(cfg) / "diagnostics"
    paths = write_spacing_candidate_outputs(candidates, summary, out_dir)

    print(f"selected_images: {len(selected)}")
    print(f"candidate_rows: {len(candidates)}")
    print(f"final_accepted_patch_count: {summary['final_accepted_patch_count']}")
    print(f"patches_with_any_local_peak: {summary['patches_with_any_local_peak']}")
    print(f"patches_with_local_peak_inside_expected_band: {summary['patches_with_local_peak_inside_expected_band']}")
    print(f"patches_where_best_global_peak_outside_expected_band: {summary['patches_where_best_global_peak_outside_expected_band']}")
    print(f"main_table_comparison: {summary['main_table_comparison']}")
    print(f"runtime_seconds: {perf_counter() - started:.3f}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
