#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_endpoint_review_pack import export_confocal_endpoint_review_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Export visual endpoint review pack for the larger confocal cohort.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--endpoint-dir", default="results/confocal_endpoint_report")
    parser.add_argument("--pipeline-dir", default="results/confocal_larger_pipeline")
    parser.add_argument("--audit-dir", default="results/confocal_larger_audit")
    parser.add_argument("--output-dir", default="results/confocal_endpoint_review_pack")
    parser.add_argument("--write-zip", action="store_true")
    parser.add_argument("--n-oop-only-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    index, summary, paths = export_confocal_endpoint_review_pack(
        load_config(args.config),
        endpoint_dir=args.endpoint_dir,
        pipeline_dir=args.pipeline_dir,
        audit_dir=args.audit_dir,
        output_directory=args.output_dir,
        write_zip=args.write_zip,
        n_oop_only_examples=args.n_oop_only_examples,
        seed=args.seed,
    )

    print(f"images_included: {len(index)}")
    print(f"review_group_counts: {summary['review_group_counts']}")
    print(f"endpoint_class_counts: {summary['endpoint_class_counts']}")
    print(f"spacing_reportable_images: {summary['spacing_reportable_images']}")
    print(f"low_candidate_review_images: {summary['low_candidate_review_images']}")
    print(f"oop_only_example_images: {summary['oop_only_example_images']}")
    print(f"review_image_files_copied: {summary['review_image_files_copied']}")
    print(f"missing_preview_count: {summary['missing_preview_count']}")
    print(f"index: {paths['index']}")
    print(f"notes: {paths['notes']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    if args.write_zip:
        print(f"zip: {paths['zip']}")


if __name__ == "__main__":
    main()
