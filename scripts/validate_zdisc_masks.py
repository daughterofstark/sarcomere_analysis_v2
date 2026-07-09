#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.validation_zdisc_masks import validate_zdisc_masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pilot validation of automated OOP/orientation against manual Z-disc masks.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--annotation-features")
    parser.add_argument("--patch-features")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-n-for-correlation", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, summary, paths = validate_zdisc_masks(
        cfg,
        annotation_features=args.annotation_features,
        patch_features=args.patch_features,
        output_directory=args.output_dir,
        min_n_for_correlation=args.min_n_for_correlation,
    )
    print(f"total_annotation_masks: {summary['total_annotation_masks']}")
    print(f"matched_rows_to_automated_patch_features: {summary['matched_rows_to_automated_patch_features']}")
    print(f"unmatched_rows: {summary['unmatched_rows']}")
    print(f"donor_id_mismatches: {summary['donor_id_mismatches']}")
    print(f"masks_with_zdisc_labels: {summary['masks_with_zdisc_labels']}")
    print(f"masks_orientation_estimable: {summary['masks_orientation_estimable']}")
    print(f"n_orientation_pairs: {summary['n_orientation_pairs']}")
    print(f"median_axial_error_deg: {summary['median_axial_error_deg']}")
    print(f"mean_axial_error_deg: {summary['mean_axial_error_deg']}")
    print(f"iqr_axial_error_deg: {summary['iqr_axial_error_deg']}")
    print(f"oop_medians_by_annotation_status: {summary['oop_medians_by_annotation_status']}")
    print(f"spearman_zdisc_fraction_vs_patch_oop: {summary['spearman_zdisc_fraction_vs_patch_oop']}")
    print(f"matched_csv: {paths['matched_csv']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")


if __name__ == "__main__":
    main()
