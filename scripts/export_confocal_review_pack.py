#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_review_pack import DEFAULT_REVIEW_IMAGES, export_confocal_review_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a shareable confocal review pack for Natalia.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--images", nargs="+", default=DEFAULT_REVIEW_IMAGES)
    parser.add_argument("--output-dir")
    parser.add_argument("--write-zip", action="store_true")
    args = parser.parse_args()

    review_summary, summary, paths = export_confocal_review_pack(
        load_config(args.config),
        images=args.images,
        output_directory=args.output_dir,
        write_zip=args.write_zip,
    )
    print(f"images_included: {summary['images_included']}")
    print(f"review_image_files_copied: {summary['review_image_files_copied']}")
    print(f"missing_preview_count: {summary['missing_preview_count']}")
    print(f"summary_rows: {len(review_summary)}")
    print(f"review_images_dir: {paths['review_images']}")
    print(f"summary_csv: {paths['summary_csv']}")
    print(f"notes_md: {paths['notes_md']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    if args.write_zip:
        print(f"zip: {paths['zip']}")


if __name__ == "__main__":
    main()
