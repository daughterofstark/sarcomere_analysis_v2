#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_freeze_report import write_confocal_freeze_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Write final confocal decision/freeze report from existing outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pipeline-dir", default="results/confocal_larger_pipeline")
    parser.add_argument("--audit-dir", default="results/confocal_larger_audit")
    parser.add_argument("--endpoint-dir", default="results/confocal_endpoint_report")
    parser.add_argument("--review-pack-dir", default="results/confocal_endpoint_review_pack")
    parser.add_argument("--output-dir", default="results/confocal_freeze_report")
    parser.add_argument("--docs-dir")
    args = parser.parse_args()

    report, paths = write_confocal_freeze_report(
        load_config(args.config),
        pipeline_dir=args.pipeline_dir,
        audit_dir=args.audit_dir,
        endpoint_dir=args.endpoint_dir,
        review_pack_dir=args.review_pack_dir,
        output_directory=args.output_dir,
        docs_directory=args.docs_dir,
    )

    endpoint = report["endpoint_result"]
    manual = report["manual_visual_spot_check"]
    print(f"primary_gate: {report['primary_gate']}")
    print(f"relaxed_gate: {report['relaxed_gate']}")
    print(f"images: {report['larger_dataset']['images']}")
    print(f"oop_reportable_images: {endpoint['oop_reportable_images']}")
    print(f"spacing_reportable_images: {endpoint['spacing_reportable_images']}")
    print(f"manual_reviewed_in_chat: {manual['reviewed_in_chat_count']}")
    print(f"manual_reviewed_pass: {manual['reviewed_pass_count']}/{manual['reviewed_pass_denominator']}")
    print(f"json: {paths['json']}")
    print(f"txt: {paths['txt']}")
    print(f"markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
