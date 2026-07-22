#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_striation_mask import run_confocal_striation_mask_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory confocal confident-striation candidate mask audit.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--confocal-manifest", default="results/confocal_baseline/confocal_manifest.csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--min-gradient-energy", type=float)
    parser.add_argument("--min-orientation-coherence", type=float)
    parser.add_argument("--min-intensity-std", type=float)
    parser.add_argument("--max-saturation-fraction", type=float)
    parser.add_argument("--write-previews", action="store_true")
    args = parser.parse_args()

    per_patch, per_image, summary, paths = run_confocal_striation_mask_audit(
        load_config(args.config),
        confocal_manifest=args.confocal_manifest,
        output_directory=args.output_dir,
        write_previews=args.write_previews,
        patch_size=args.patch_size,
        stride=args.stride,
        min_gradient_energy=args.min_gradient_energy,
        min_orientation_coherence=args.min_orientation_coherence,
        min_intensity_std=args.min_intensity_std,
        max_saturation_fraction=args.max_saturation_fraction,
    )
    print(f"confocal_images: {summary['confocal_image_count']}")
    print(f"processed_ok: {summary['processed_ok']}")
    print(f"processed_error: {summary['processed_error']}")
    print(f"total_patches: {len(per_patch)}")
    print(f"candidate_patch_count: {summary['candidate_patch_count']}")
    print(f"candidate_patch_fraction: {summary['candidate_patch_fraction']}")
    print("per_image_candidates:")
    if not per_image.empty:
        print(
            per_image[
                [
                    "confocal_image_id",
                    "filename",
                    "candidate_patch_count",
                    "candidate_patch_fraction",
                    "expected_positive_example",
                    "noted_complex_example",
                ]
            ].to_string(index=False)
        )
    print(f"per_patch: {paths['per_patch']}")
    print(f"per_image: {paths['per_image']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    if args.write_previews:
        print(f"preview_count: {len(summary['preview_paths'])}")
        print(f"preview_dir: {paths['previews']}")


if __name__ == "__main__":
    main()
