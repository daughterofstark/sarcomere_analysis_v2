#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_metadata import audit_confocal_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit per-image confocal pixel-size metadata.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--confocal-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--write-manual-template", action="store_true")
    args = parser.parse_args()

    _, summary, paths = audit_confocal_metadata(
        load_config(args.config),
        confocal_manifest=args.confocal_manifest,
        output_directory=args.output_dir,
        write_manual_template=args.write_manual_template,
    )
    print(f"image_count: {summary['image_count']}")
    print(f"pixel_size_available_count: {summary['pixel_size_available_count']}")
    print(f"pixel_size_missing_count: {summary['pixel_size_missing_count']}")
    print(f"unique_pixel_sizes_um: {summary['unique_pixel_sizes_um']}")
    print(f"pixel_sizes_differ_across_images: {summary['pixel_sizes_differ_across_images']}")
    print(f"widefield_calibration_used: {summary['widefield_calibration_used']}")
    print(f"calibration_csv: {paths['calibration']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    if args.write_manual_template:
        print(f"manual_template: {paths['manual_template']}")


if __name__ == "__main__":
    main()
