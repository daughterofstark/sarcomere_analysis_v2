#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_pipeline import run_confocal_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the confocal-first exploratory pipeline wrapper.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--confocal-root", required=True)
    parser.add_argument("--output-dir", default="results/confocal_pipeline")
    parser.add_argument("--spacing-min-um", type=float, default=1.5)
    parser.add_argument("--spacing-max-um", type=float, default=2.4)
    parser.add_argument("--write-previews", action="store_true")
    args = parser.parse_args()

    manifest, per_patch, per_image, summary, paths = run_confocal_pipeline(
        load_config(args.config),
        confocal_root=args.confocal_root,
        output_directory=args.output_dir,
        write_previews=args.write_previews,
        spacing_min_um=args.spacing_min_um,
        spacing_max_um=args.spacing_max_um,
    )
    print(f"primary_gate_used: {summary['primary_gate_used']}")
    print(f"relaxed_gate_status: {summary['relaxed_gate_status']}")
    print(f"widefield_calibration_used: {summary['widefield_calibration_used']}")
    print(f"images_processed: {summary['images_processed']}")
    print(f"errors: {summary['errors']}")
    print(f"calibrated_images: {summary['calibrated_images']}")
    print(f"total_patches: {summary['total_patches']}")
    print(f"selected_candidate_patches: {summary['selected_candidate_patches']}")
    print(f"valid_selected_spacing_patches: {summary['valid_selected_spacing_patches']}")
    print(f"median_selected_oop: {summary['median_selected_oop']}")
    print(f"median_selected_spacing_um: {summary['median_selected_spacing_um']}")
    print(f"manifest_rows: {len(manifest)}")
    print(f"per_patch_rows: {len(per_patch)}")
    print(f"per_image_rows: {len(per_image)}")
    print(f"manifest: {paths['manifest']}")
    print(f"per_patch: {paths['per_patch']}")
    print(f"per_image: {paths['per_image']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    if args.write_previews:
        print(f"preview_count: {len(summary['preview_paths'])}")
        print(f"preview_dir: {paths['previews']}")


if __name__ == "__main__":
    main()
