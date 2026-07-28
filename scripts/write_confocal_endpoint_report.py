#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_endpoint_report import write_confocal_endpoint_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Write endpoint-aware larger confocal cohort classification report.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--audit-dir", default="results/confocal_larger_audit")
    parser.add_argument("--pipeline-dir", default="results/confocal_larger_pipeline")
    parser.add_argument("--output-dir", default="results/confocal_endpoint_report")
    parser.add_argument("--docs-dir")
    args = parser.parse_args()

    per_image, summary, paths = write_confocal_endpoint_report(
        load_config(args.config),
        audit_dir=args.audit_dir,
        pipeline_dir=args.pipeline_dir,
        output_directory=args.output_dir,
        docs_directory=args.docs_dir,
    )

    print(f"images: {len(per_image)}")
    print(f"endpoint_class_counts: {summary['endpoint_class_counts']}")
    print(f"spacing_reportable_image_count: {summary['spacing_reportable_image_count']}")
    print(f"oop_reportable_image_count: {summary['oop_reportable_image_count']}")
    print(f"oop_only_image_count: {summary['oop_only_image_count']}")
    print(f"review_needed_image_count: {summary['review_needed_image_count']}")
    print(f"per_image: {paths['per_image']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    print(f"markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
