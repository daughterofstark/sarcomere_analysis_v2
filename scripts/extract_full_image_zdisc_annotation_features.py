#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.full_image_annotation_features import extract_full_image_zdisc_annotation_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-image features from manually drawn full-image Z-disc masks.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--index")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-zdisc-pixels", type=int, default=10)
    parser.add_argument("--min-components", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, summary, paths = extract_full_image_zdisc_annotation_features(
        cfg,
        index_path=args.index,
        output_directory=args.output_dir,
        min_zdisc_pixels=args.min_zdisc_pixels,
        min_components=args.min_components,
    )
    print(f"mask_count: {summary['mask_count']}")
    print(f"masks_with_zdisc_labels: {summary['masks_with_zdisc_labels']}")
    print(f"empty_masks: {summary['empty_masks']}")
    print(f"ignore_only_masks: {summary['ignore_only_masks']}")
    print(f"mixed_masks: {summary['mixed_masks']}")
    print(f"orientation_estimable_masks: {summary['orientation_estimable_masks']}")
    print(f"annotation_status_counts: {summary['annotation_status_counts']}")
    print(f"features_csv: {paths['features_csv']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")


if __name__ == "__main__":
    main()
