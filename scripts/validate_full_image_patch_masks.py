#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.validation_full_image_patch_masks import validate_full_image_patch_masks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pilot patch-level validation of OOP/orientation from sparse full-image Z-disc masks."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--annotation-index")
    parser.add_argument("--mask-dir")
    parser.add_argument("--patch-features")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-zdisc-pixels", type=int, default=10)
    parser.add_argument("--min-n-for-correlation", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, summary, paths = validate_full_image_patch_masks(
        cfg,
        annotation_index=args.annotation_index,
        mask_dir=args.mask_dir,
        patch_features=args.patch_features,
        output_directory=args.output_dir,
        min_zdisc_pixels=args.min_zdisc_pixels,
        min_n_for_correlation=args.min_n_for_correlation,
    )

    print(f"full_images_with_masks: {summary['full_images_with_masks']}")
    print(f"total_automated_patches_in_annotated_images: {summary['total_automated_patches_in_annotated_images']}")
    print(f"matched_patch_rows: {summary['matched_patch_rows']}")
    print(f"unmatched_patch_rows: {summary['unmatched_patch_rows']}")
    print(f"donor_id_mismatches: {summary['donor_id_mismatches']}")
    print(f"patches_with_manual_zdisc_labels: {summary['patches_with_manual_zdisc_labels']}")
    print(f"patches_empty: {summary['patches_empty']}")
    print(f"patches_ignore_only: {summary['patches_ignore_only']}")
    print(f"patches_manual_orientation_estimable: {summary['patches_manual_orientation_estimable']}")
    print(f"n_orientation_pairs: {summary['n_orientation_pairs']}")
    print(f"median_axial_error_deg: {summary['median_axial_error_deg']}")
    print(f"mean_axial_error_deg: {summary['mean_axial_error_deg']}")
    print(f"iqr_axial_error_deg: {summary['iqr_axial_error_deg']}")
    print(f"oop_medians_by_manual_patch_status: {summary['oop_medians_by_manual_patch_status']}")
    print(f"spearman_zdisc_fraction_vs_patch_oop: {summary['spearman_zdisc_fraction_vs_patch_oop']}")
    print(f"matched_csv: {paths['matched_csv']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")


if __name__ == "__main__":
    main()
