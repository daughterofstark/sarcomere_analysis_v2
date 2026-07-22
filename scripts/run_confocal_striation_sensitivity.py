#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_striation_sensitivity import run_confocal_striation_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description="Run confocal confident-striation candidate-mask sensitivity audit.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--patch-table")
    parser.add_argument("--image-table")
    parser.add_argument("--output-dir")
    parser.add_argument("--write-previews", action="store_true")
    parser.add_argument("--max-preview-variants", type=int, default=3)
    args = parser.parse_args()

    variants, per_image, summary, paths = run_confocal_striation_sensitivity(
        load_config(args.config),
        patch_table=args.patch_table,
        image_table=args.image_table,
        output_directory=args.output_dir,
        write_previews=args.write_previews,
        max_preview_variants=args.max_preview_variants,
    )
    print(f"variant_count: {len(variants)}")
    print(f"classification_counts: {summary['classification_counts']}")
    print(f"plausible_variants: {summary['plausible_variants']}")
    print("variants:")
    print(
        variants[
            [
                "variant_id",
                "classification",
                "overall_candidate_fraction",
                "candidate_fraction_5138",
                "candidate_fraction_6052",
                "candidate_fraction_3112",
                "median_candidate_fraction_by_image",
                "images_gt_90_candidate_fraction",
                "images_lt_05_candidate_fraction",
            ]
        ].to_string(index=False)
    )
    print(f"per_image_rows: {len(per_image)}")
    print(f"variants_csv: {paths['variants']}")
    print(f"per_image_csv: {paths['per_image']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    if args.write_previews:
        print(f"preview_count: {len(summary['preview_paths'])}")
        print(f"preview_dir: {paths['previews']}")


if __name__ == "__main__":
    main()
