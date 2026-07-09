#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.full_image_annotation import prepare_full_image_annotation_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare full-image local Z-disc/striation annotation PNGs and masks.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-images", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir")
    parser.add_argument("--analysis-table")
    parser.add_argument("--feature-table")
    parser.add_argument("--manifest-table")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, summary, paths = prepare_full_image_annotation_set(
        cfg,
        n_images=args.n_images,
        seed=args.seed,
        output_directory=args.output_dir,
        analysis_table=args.analysis_table,
        feature_table=args.feature_table,
        manifest_table=args.manifest_table,
        overwrite=args.overwrite,
    )
    print(f"selected_images: {summary['selected_images']}")
    print(f"oop_bin_counts: {summary['oop_bin_counts']}")
    print(f"unique_donors: {summary['unique_donors']}")
    print(f"images_dir: {paths['images_dir']}")
    print(f"masks_dir: {paths['masks_dir']}")
    print(f"overlays_dir: {paths['overlays_dir']}")
    print(f"index_path: {paths['index']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    print("label_convention: 0=background/unlabeled, 1=visible Z-disc/striation, 2=ignore/uncertain/autofluorescence/ambiguous")


if __name__ == "__main__":
    main()
