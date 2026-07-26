#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_spacing_audit import run_confocal_spacing_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibrated confocal spacing audit on selected candidate regions.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--calibration-table")
    parser.add_argument("--same-grid-oop-table")
    parser.add_argument("--output-dir")
    parser.add_argument("--spacing-min-um", type=float, default=1.5)
    parser.add_argument("--spacing-max-um", type=float, default=2.4)
    parser.add_argument("--write-previews", action="store_true")
    args = parser.parse_args()

    _, per_image, summary, paths = run_confocal_spacing_audit(
        load_config(args.config),
        calibration_table=args.calibration_table,
        same_grid_oop_table=args.same_grid_oop_table,
        output_directory=args.output_dir,
        spacing_min_um=args.spacing_min_um,
        spacing_max_um=args.spacing_max_um,
        write_previews=args.write_previews,
    )
    print(f"image_count: {summary['image_count']}")
    print(f"calibrated_image_count: {summary['calibrated_image_count']}")
    print(f"widefield_calibration_used: {summary['widefield_calibration_used']}")
    print(f"candidate_patch_count: {summary['candidate_patch_count']}")
    print(f"valid_spacing_patch_count_all: {summary['valid_spacing_patch_count_all']}")
    print(f"valid_spacing_patch_count_selected: {summary['valid_spacing_patch_count_selected']}")
    print(f"valid_spacing_fraction_selected: {summary['valid_spacing_fraction_selected']}")
    print(f"selected_spacing_um_summary: {summary['selected_spacing_um_summary']}")
    print("per_image:")
    if not per_image.empty:
        print(
            per_image[
                [
                    "confocal_image_id",
                    "filename",
                    "pixel_size_um",
                    "candidate_patch_count",
                    "spacing_valid_patch_count_selected",
                    "spacing_valid_fraction_selected",
                    "selected_median_spacing_um",
                    "interpretation_flag",
                ]
            ].to_string(index=False)
        )
    print(f"per_patch: {paths['per_patch']}")
    print(f"per_image_csv: {paths['per_image']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    if args.write_previews:
        print(f"preview_dir: {paths['previews']}")
        print(f"preview_count: {len(summary['preview_paths'])}")


if __name__ == "__main__":
    main()
