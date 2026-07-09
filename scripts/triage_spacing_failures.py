#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config, output_dir
from sarcomere_analysis.diagnostics.spacing_failure_triage import (
    read_spacing_triage_inputs,
    triage_spacing_failures,
    write_spacing_failure_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage why corrected spacing estimates are sparse.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--patch-table", help="Override per_patch_metrics.csv path.")
    parser.add_argument("--image-table", help="Override per_image_metrics.csv path.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to config output_dir/diagnostics.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    patch, image, diagnostic_summary = read_spacing_triage_inputs(cfg, args.patch_table, args.image_table)
    summary, by_image = triage_spacing_failures(patch, image, diagnostic_summary)
    out_dir = Path(args.output_dir) if args.output_dir else output_dir(cfg) / "diagnostics"
    paths = write_spacing_failure_outputs(summary, by_image, out_dir)

    print(f"total_patches: {summary['total_patches']}")
    print(f"qc_valid_spacing_patches: {summary['qc_valid_spacing_patches']}")
    print(f"patches_reaching_spacing_estimator: {summary['patches_reaching_spacing_estimator']}")
    print(f"final_valid_spacing_patches: {summary['final_valid_spacing_patches']}")
    print(f"images_with_no_valid_spacing: {summary['images_with_no_valid_spacing']}")
    print(f"failure_mode_assessment: {summary['failure_mode_assessment']['plain_language']}")
    print(f"candidate_level_detail_available: {summary['candidate_level_detail_available']}")
    print(f"top_spacing_invalid_reason_counts: {summary['top_spacing_invalid_reason_counts']}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
