#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.synthetic_oop import validate_synthetic_oop


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen OOP/orientation estimator on controlled synthetic striated images.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-replicates", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--write-example-images", action="store_true")
    args = parser.parse_args()

    _, summary, paths = validate_synthetic_oop(
        load_config(args.config),
        output_directory=args.output_dir,
        seed=args.seed,
        n_replicates=args.n_replicates,
        size=args.size,
        write_example_images=args.write_example_images,
    )

    print(f"synthetic_examples: {summary['synthetic_examples']}")
    print(f"clean_case_n: {summary['clean_case_n']}")
    print(f"clean_case_median_angular_error_deg: {summary['clean_case_median_angular_error_deg']}")
    print(f"clean_case_max_angular_error_deg: {summary['clean_case_max_angular_error_deg']}")
    print(f"recovered_oop_median_by_disorder_level: {summary['recovered_oop_median_by_disorder_level']}")
    print(f"oop_monotonicity_low_gt_medium_gt_high: {summary['oop_monotonicity_low_gt_medium_gt_high']}")
    print(f"degradation_failure_modes: {summary['degradation_failure_modes']}")
    print(f"results_csv: {paths['results_csv']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")


if __name__ == "__main__":
    main()
