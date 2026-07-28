#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_cohort_audit import audit_confocal_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and triage a consolidated larger confocal pipeline run.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pipeline-dir", default="results/confocal_larger_pipeline")
    parser.add_argument("--pilot-dir", default="results/confocal_pipeline")
    parser.add_argument("--output-dir", default="results/confocal_larger_audit")
    parser.add_argument("--collect-previews", action="store_true")
    parser.add_argument("--spacing-min-um", type=float, default=1.5)
    parser.add_argument("--spacing-max-um", type=float, default=2.4)
    args = parser.parse_args()

    triage, cohort_summary, spacing_distribution, summary, paths = audit_confocal_cohort(
        load_config(args.config),
        pipeline_dir=args.pipeline_dir,
        pilot_dir=args.pilot_dir,
        output_directory=args.output_dir,
        collect_previews=args.collect_previews,
        spacing_min_um=args.spacing_min_um,
        spacing_max_um=args.spacing_max_um,
    )

    cohort = cohort_summary.iloc[0].to_dict()
    print(f"images_processed: {summary['images_processed']}")
    print(f"errors: {summary['errors']}")
    print(f"calibrated_images: {summary['calibrated_images']}")
    print(f"total_patches: {summary['total_patches']}")
    print(f"selected_candidate_patches: {summary['selected_candidate_patches']}")
    print(f"valid_selected_spacing_patches: {summary['valid_selected_spacing_patches']}")
    print(f"selected_spacing_valid_fraction: {summary['selected_spacing_valid_fraction']}")
    print(f"median_selected_oop: {summary['median_selected_oop']}")
    print(f"median_selected_spacing_um: {summary['median_selected_spacing_um']}")
    print(f"image_count_by_interpretation_class: {summary['image_count_by_interpretation_class']}")
    print(f"pilot_selected_spacing_valid_fraction: {summary['pilot_comparison']['pilot_selected_spacing_valid_fraction']}")
    print(f"larger_selected_spacing_valid_fraction: {summary['pilot_comparison']['larger_selected_spacing_valid_fraction']}")
    print(f"spacing_distribution_rows: {len(spacing_distribution)}")
    print(f"triage: {paths['triage']}")
    print(f"cohort_summary: {paths['cohort_summary']}")
    print(f"spacing_distribution: {paths['spacing_distribution']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    if args.collect_previews:
        print(f"review_preview_count: {len(summary['review_preview_paths'])}")
        print(f"review_preview_dir: {paths['review_previews']}")


if __name__ == "__main__":
    main()
