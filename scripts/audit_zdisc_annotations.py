#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.zdisc_annotation import audit_zdisc_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local Z-disc/striation annotation masks.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--annotation-index")
    parser.add_argument("--output-dir")
    parser.add_argument("--write-overlays", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, summary, paths = audit_zdisc_annotations(
        cfg,
        annotation_index_path=args.annotation_index,
        output_directory=args.output_dir,
        write_overlays=args.write_overlays,
    )
    print(f"selected_crops: {summary['selected_crops']}")
    print(f"missing_masks: {summary['missing_masks']}")
    print(f"shape_mismatch_masks: {summary['shape_mismatch_masks']}")
    print(f"invalid_label_masks: {summary['invalid_label_masks']}")
    print(f"empty_masks: {summary['empty_masks']}")
    print(f"masks_with_zdisc_labels: {summary['masks_with_zdisc_labels']}")
    print(f"label_pixel_totals: {summary['label_pixel_totals']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    if args.write_overlays:
        print(f"overlays_dir: {paths['overlays_dir']}")


if __name__ == "__main__":
    main()
