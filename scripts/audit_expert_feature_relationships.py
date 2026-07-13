#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.expert_feature_audit import audit_expert_feature_relationships


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit relationships between expert annotations and existing automated patch features.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--matched")
    parser.add_argument("--patch-features")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--min-confidence", type=int, default=3)
    args = parser.parse_args()

    _, visibility, organisation, _, summary, paths = audit_expert_feature_relationships(
        load_config(args.config),
        matched=args.matched,
        patch_features=args.patch_features,
        output_directory=args.output_dir,
        min_n=args.min_n,
        min_confidence=args.min_confidence,
    )
    print(f"rows: {summary['audit']['rows_in_matched_expert_annotations']}")
    print(f"numeric_features_considered: {summary['audit']['numeric_automated_features_considered']}")
    print("top_visibility_features:")
    print(visibility.head(10).to_string(index=False))
    print("top_organisation_features:")
    print(organisation.head(10).to_string(index=False))
    print(f"oop_specific_statement: {summary['oop_specific_statement']}")
    print(f"feature_table: {paths['feature_table']}")
    print(f"visibility_summary: {paths['visibility_summary']}")
    print(f"organisation_summary: {paths['organisation_summary']}")
    print(f"confidence_summary: {paths['confidence_summary']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")


if __name__ == "__main__":
    main()
