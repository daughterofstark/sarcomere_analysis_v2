#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_analysis_dataset import build_confocal_analysis_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build downstream-safe confocal analysis table and manual review template.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--endpoint-dir", default="results/confocal_endpoint_report")
    parser.add_argument("--pipeline-dir", default="results/confocal_larger_pipeline")
    parser.add_argument("--audit-dir", default="results/confocal_larger_audit")
    parser.add_argument("--freeze-dir", default="results/confocal_freeze_report")
    parser.add_argument("--review-pack-dir", default="results/confocal_endpoint_review_pack")
    parser.add_argument("--output-dir", default="results/confocal_analysis_dataset")
    parser.add_argument("--docs-dir")
    args = parser.parse_args()

    per_image, review_template, summary, paths = build_confocal_analysis_dataset(
        load_config(args.config),
        endpoint_dir=args.endpoint_dir,
        pipeline_dir=args.pipeline_dir,
        audit_dir=args.audit_dir,
        freeze_dir=args.freeze_dir,
        review_pack_dir=args.review_pack_dir,
        output_directory=args.output_dir,
        docs_directory=args.docs_dir,
    )

    print(f"images_total: {len(per_image)}")
    print(f"oop_allowed_count: {summary['oop_allowed_count']}")
    print(f"spacing_allowed_count: {summary['spacing_allowed_count']}")
    print(f"review_template_rows: {len(review_template)}")
    print(f"spacing_reportable_image_list: {summary['spacing_reportable_image_list']}")
    print(f"per_image: {paths['per_image']}")
    print(f"review_template: {paths['review_template']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    print(f"markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
