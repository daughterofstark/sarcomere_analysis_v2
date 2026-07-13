#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.expert_annotation_validation import validate_expert_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Import/audit blinded expert annotations and compare organisation scores to automated OOP.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--annotations")
    parser.add_argument("--internal-key")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-n-correlation", type=int, default=10)
    parser.add_argument("--min-confidence", type=int, default=3)
    parser.add_argument("--allow-orientation-exploratory", action="store_true")
    args = parser.parse_args()

    _, _, summary, paths = validate_expert_annotations(
        load_config(args.config),
        annotations=args.annotations,
        internal_key=args.internal_key,
        output_directory=args.output_dir,
        min_n_correlation=args.min_n_correlation,
        min_confidence=args.min_confidence,
        allow_orientation_exploratory=args.allow_orientation_exploratory,
    )

    print(f"total_rows: {summary['audit']['total_rows']}")
    print(f"matched_rows: {summary['audit']['annotations_matched_to_internal_key']}")
    print(f"unmatched_annotation_ids: {summary['audit']['unmatched_annotation_ids']}")
    print(f"duplicate_annotation_ids: {summary['audit']['duplicate_annotation_ids']}")
    print(f"visibility_counts: {summary['visibility_vs_automated_oop']['counts']}")
    print(f"organisation_score_counts: {summary['organisation_score_vs_automated_oop']['counts']}")
    print(f"confidence_completed: {summary['audit']['completed_confidence_score_count']}")
    print(f"visibility_oop_medians: {summary['visibility_vs_automated_oop']['oop_medians']}")
    print(f"organisation_oop_medians: {summary['organisation_score_vs_automated_oop']['oop_medians']}")
    print(f"organisation_oop_spearman: {summary['organisation_score_vs_automated_oop']['spearman']}")
    print(f"confidence_filtered: {summary['confidence_filtered']}")
    print(f"dominant_orientation_primary: {summary['orientation']['dominant_orientation_used_as_primary']}")
    print(f"spacing_validation_status: {summary['spacing']['spacing_validation_status']}")
    print(f"normalized_csv: {paths['normalized_csv']}")
    print(f"matched_csv: {paths['matched_csv']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")


if __name__ == "__main__":
    main()
