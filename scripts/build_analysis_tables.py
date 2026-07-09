#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.analysis_tables import (
    build_analysis_tables,
    load_analysis_inputs,
    write_analysis_outputs,
)
from sarcomere_analysis.config import load_config, output_dir
from sarcomere_analysis.metadata import healthy_donor_ids_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build joined analysis-ready feature/metadata tables without statistics.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--features-image")
    parser.add_argument("--features-donor")
    parser.add_argument("--enriched-manifest")
    parser.add_argument("--donor-metadata")
    parser.add_argument("--output-dir")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_features, donor_features, enriched_manifest, donor_metadata = load_analysis_inputs(
        cfg,
        features_image=args.features_image,
        features_donor=args.features_donor,
        enriched_manifest=args.enriched_manifest,
        donor_metadata=args.donor_metadata,
    )
    per_image, per_donor, summary = build_analysis_tables(
        image_features,
        donor_features,
        enriched_manifest,
        donor_metadata,
        strict=args.strict,
        expected_healthy_donor_count=len(healthy_donor_ids_from_config(cfg)),
    )
    out_dir = Path(args.output_dir) if args.output_dir else output_dir(cfg) / "tables"
    paths = write_analysis_outputs(per_image, per_donor, summary, out_dir)

    print(f"image_rows: {summary['image_rows']}")
    print(f"donor_rows: {summary['donor_rows']}")
    print(f"unique_donors: {summary['unique_donors']}")
    print(f"healthy_donor_count: {summary['healthy_donor_count']}")
    print(f"healthy_image_rows: {summary['healthy_image_rows']}")
    print(f"missing_image_metadata_count: {summary['missing_image_metadata_count']}")
    print(f"missing_donor_metadata_count: {summary['missing_donor_metadata_count']}")
    print(f"spacing_global_status: {summary['spacing_global_status']}")
    print(f"spacing_is_exploratory_low_yield: {summary['spacing_is_exploratory_low_yield']}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
