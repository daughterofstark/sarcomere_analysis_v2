#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_gate_refinement import DEFAULT_FOCUS_IMAGES, run_confocal_gate_refinement


def main() -> None:
    parser = argparse.ArgumentParser(description="Run review-guided confocal gate refinement audit.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--write-previews", action="store_true")
    parser.add_argument("--focus-images", nargs="+", default=DEFAULT_FOCUS_IMAGES)
    args = parser.parse_args()

    variants, per_image, summary, paths = run_confocal_gate_refinement(
        load_config(args.config),
        output_directory=args.output_dir,
        write_previews=args.write_previews,
        focus_images=args.focus_images,
    )
    print(f"variants_tested: {summary['variants_tested']}")
    print(f"classification_counts: {summary['classification_counts']}")
    print(f"recommendation: {summary['recommendation']}")
    print(f"best_plausible_relaxed_variant: {summary['best_plausible_relaxed_variant']}")
    print("variant_table:")
    print(variants.to_string(index=False))
    print("focus_images:")
    print(per_image.loc[per_image["confocal_image_id"].isin(args.focus_images)].to_string(index=False))
    print(f"variants_csv: {paths['variants']}")
    print(f"per_image_csv: {paths['per_image']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    if args.write_previews:
        print(f"preview_dir: {paths['previews']}")
        print(f"preview_count: {len(summary['preview_paths'])}")


if __name__ == "__main__":
    main()
