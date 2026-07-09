#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.annotation_pack import export_annotation_pack
from sarcomere_analysis.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a stratified manual annotation pack for OOP/orientation validation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--patch-table")
    parser.add_argument("--image-table")
    parser.add_argument("--analysis-table")
    parser.add_argument("--manifest-table")
    parser.add_argument("--output-dir")
    parser.add_argument("--n-patches", type=int, default=80)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    index, _, summary, paths = export_annotation_pack(
        cfg,
        patch_table=args.patch_table,
        image_table=args.image_table,
        analysis_table=args.analysis_table,
        manifest_table=args.manifest_table,
        output_directory=args.output_dir,
        n_patches=args.n_patches,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"selected_patches: {summary['selected_patches']}")
    print(f"valid_orientation_selected: {summary['valid_orientation_selected']}")
    print(f"invalid_control_selected: {summary['invalid_control_selected']}")
    print(f"oop_bin_counts: {summary['oop_bin_counts']}")
    print(f"unique_donors: {summary['unique_donors']}")
    print(f"unique_images: {summary['unique_images']}")
    print(f"max_patches_per_donor: {summary['max_patches_per_donor']}")
    print(f"max_patches_per_image: {summary['max_patches_per_image']}")
    print(f"crop_count: {summary['crop_count']}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")
    if "crop_path" in index.columns:
        print(f"crop_directory: {Path(paths['annotation_index']).parent / 'crops'}")


if __name__ == "__main__":
    main()
