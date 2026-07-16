#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.expert_crop_feature_audit import audit_expert_crop_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit automated features computed directly on expert-visible annotation crop PNGs."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--crop-dir")
    parser.add_argument("--internal-key")
    parser.add_argument("--matched-annotations")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--min-confidence", type=int, default=3)
    args = parser.parse_args()

    _, visibility, organisation, _, summary, paths = audit_expert_crop_features(
        load_config(args.config),
        crop_dir=args.crop_dir,
        internal_key=args.internal_key,
        matched_annotations=args.matched_annotations,
        output_directory=args.output_dir,
        min_n=args.min_n,
        min_confidence=args.min_confidence,
    )
    print(f"rows: {summary['audit']['rows']}")
    print(f"crop_pngs_found: {summary['audit']['crop_pngs_found']}")
    print(f"crop_pngs_missing: {summary['audit']['crop_pngs_missing']}")
    print(f"previous_patch_oop_vs_organisation: {summary['previous_production_patch_oop_vs_organisation']}")
    print(f"crop_oop_vs_organisation: {summary['crop_oop_vs_organisation']}")
    print(f"crop_oop_vs_organisation_confidence_filtered: {summary['crop_oop_vs_organisation_confidence_filtered']}")
    print("top_visibility_features:")
    print(visibility.head(10).to_string(index=False))
    print("top_organisation_features:")
    print(organisation.head(10).to_string(index=False))
    print(f"feature_table: {paths['feature_table']}")
    print(f"visibility_summary: {paths['visibility_summary']}")
    print(f"organisation_summary: {paths['organisation_summary']}")
    print(f"confidence_summary: {paths['confidence_summary']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")


if __name__ == "__main__":
    main()
