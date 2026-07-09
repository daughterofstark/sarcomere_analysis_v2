#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.validation_io import (
    audit_validation_measurements,
    default_validation_output_dir,
    load_analysis_per_image,
    load_validation_csv,
    write_validation_audit_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit manual/FIJI validation measurements against analysis_per_image.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--validation-csv")
    parser.add_argument("--analysis-per-image")
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-unknown-types", action="store_true")
    parser.add_argument("--allow-example-rows", action="store_true")
    args = parser.parse_args()

    if not args.validation_csv:
        raise SystemExit(
            "No --validation-csv provided. Generate a template with: "
            "../sarcgraph-env/bin/python scripts/prepare_validation_template.py --config configs/default.yaml"
        )

    cfg = load_config(args.config)
    measurements = load_validation_csv(
        args.validation_csv,
        allow_unknown_types=args.allow_unknown_types,
        allow_example_rows=args.allow_example_rows,
    )
    analysis = load_analysis_per_image(cfg, args.analysis_per_image)
    matched, unmatched, summary = audit_validation_measurements(
        measurements,
        analysis,
        is_example_audit=args.allow_example_rows,
    )
    out_dir = Path(args.output_dir) if args.output_dir else default_validation_output_dir(cfg)
    paths = write_validation_audit_outputs(matched, unmatched, summary, out_dir)

    print(f"is_example_audit: {summary['is_example_audit']}")
    print(f"total_manual_rows: {summary['total_manual_rows']}")
    print(f"unique_images_referenced: {summary['unique_images_referenced']}")
    print(f"unique_donors_referenced: {summary['unique_donors_referenced']}")
    print(f"measurement_type_counts: {summary['measurement_type_counts']}")
    print(f"rows_matched_to_analysis_per_image: {summary['rows_matched_to_analysis_per_image']}")
    print(f"unmatched_image_id_rows: {summary['unmatched_image_id_rows']}")
    print(f"donor_id_mismatch_rows: {summary['donor_id_mismatch_rows']}")
    print(f"missing_required_field_rows: {summary['missing_required_field_rows']}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
