#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_same_grid_oop import run_confocal_same_grid_oop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute confocal OOP/orientation directly on the candidate-mask patch grid."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--patch-table")
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--write-previews", action="store_true")
    args = parser.parse_args()

    _, per_image, summary, paths = run_confocal_same_grid_oop(
        load_config(args.config),
        patch_table=args.patch_table,
        manifest=args.manifest,
        output_directory=args.output_dir,
        write_previews=args.write_previews,
    )

    print(f"candidate_source: {summary['candidate_source']}")
    print(f"same_grid_patch_rows: {summary['same_grid_patch_rows']}")
    print(f"patches_processed_ok: {summary['patches_processed_ok']}")
    print(f"patches_error: {summary['patches_error']}")
    print(f"candidate_patch_count: {summary['candidate_patch_count']}")
    print(f"spacing_status: {summary['spacing_status']}")
    print(f"selected_vs_all_oop_summary: {summary['selected_vs_all_oop_summary']}")
    print("per_image:")
    if not per_image.empty:
        print(
            per_image[
                [
                    "confocal_image_id",
                    "filename",
                    "candidate_patch_count",
                    "candidate_patch_fraction",
                    "selected_region_median_oop_128",
                    "all_region_median_oop_128",
                    "selected_vs_all_oop_difference_128",
                    "selected_region_median_coherence_128",
                    "all_region_median_coherence_128",
                    "interpretation_flag",
                ]
            ].to_string(index=False)
        )
    print(f"per_patch: {paths['per_patch']}")
    print(f"per_image_csv: {paths['per_image']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    if args.write_previews:
        print(f"preview_count: {len(summary['preview_paths'])}")
        print(f"preview_dir: {paths['previews']}")


if __name__ == "__main__":
    main()
