#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_selective_analysis import run_confocal_selective_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize confocal features inside selected candidate striation regions.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--selected-variant", default="moderate")
    parser.add_argument("--patch-table")
    parser.add_argument("--baseline-patch-table")
    parser.add_argument("--sensitivity-variants")
    parser.add_argument("--sensitivity-per-image")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-candidate-patches", type=int, default=10)
    parser.add_argument("--write-previews", action="store_true")
    args = parser.parse_args()

    _, per_image, summary, paths = run_confocal_selective_analysis(
        load_config(args.config),
        selected_variant=args.selected_variant,
        patch_table=args.patch_table,
        baseline_patch_table=args.baseline_patch_table,
        sensitivity_variants=args.sensitivity_variants,
        sensitivity_per_image=args.sensitivity_per_image,
        output_directory=args.output_dir,
        min_candidate_patches=args.min_candidate_patches,
        write_previews=args.write_previews,
    )
    print(f"selected_variant: {summary['selected_variant']}")
    print(f"total_patches: {summary['total_patches']}")
    print(f"candidate_patch_count: {summary['candidate_patch_count']}")
    print(f"baseline_patch_join_audit: {summary['baseline_patch_join_audit']}")
    print("per_image:")
    if not per_image.empty:
        print(
            per_image[
                [
                    "confocal_image_id",
                    "filename",
                    "candidate_patch_count",
                    "candidate_patch_fraction",
                    "selected_region_median_oop",
                    "selected_region_median_coherence",
                    "selected_region_median_gradient_energy",
                    "selected_region_median_intensity_std",
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
