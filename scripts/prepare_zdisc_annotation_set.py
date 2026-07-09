#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.zdisc_annotation import prepare_zdisc_annotation_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local editable Z-disc/striation annotation crops and masks.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--annotation-index")
    parser.add_argument("--output-dir")
    parser.add_argument("--n-crops", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    index, summary, paths = prepare_zdisc_annotation_set(
        cfg,
        n_crops=args.n_crops,
        seed=args.seed,
        annotation_index_path=args.annotation_index,
        output_directory=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"selected_crops: {summary['selected_crops']}")
    print(f"valid_orientation_crops: {summary['valid_orientation_crops']}")
    print(f"invalid_or_low_quality_controls: {summary['invalid_or_low_quality_controls']}")
    print(f"oop_bin_counts: {summary['oop_bin_counts']}")
    print(f"unique_donors: {summary['unique_donors']}")
    print(f"unique_images: {summary['unique_images']}")
    print(f"images_dir: {paths['images_dir']}")
    print(f"masks_dir: {paths['masks_dir']}")
    print(f"overlays_dir: {paths['overlays_dir']}")
    print(f"index_path: {paths['index']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    print("label_convention: 0=background/unlabeled, 1=visible Z-disc/striation, 2=ignore/uncertain/autofluorescence/ambiguous")
    if len(index) != args.n_crops:
        print(f"warning: requested {args.n_crops} crops but selected {len(index)}")


if __name__ == "__main__":
    main()
