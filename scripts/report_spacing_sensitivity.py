#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config, output_dir
from sarcomere_analysis.diagnostics.spacing_sensitivity import (
    DEFAULT_BAND_PADDING_GRID,
    DEFAULT_CONFIDENCE_GRID,
    DEFAULT_PEAK_RULES,
    build_spacing_sensitivity_report,
    read_candidate_table,
    write_sensitivity_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report diagnostic spacing sensitivity from candidate-level outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--candidate-table", help="Candidate table path. Defaults to results/diagnostics/spacing_candidates.csv.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to config output_dir/diagnostics.")
    parser.add_argument("--min-confidence-grid", nargs="+", type=float, default=DEFAULT_CONFIDENCE_GRID)
    parser.add_argument("--band-padding-grid", nargs="+", default=DEFAULT_BAND_PADDING_GRID)
    parser.add_argument("--peak-rules", nargs="+", default=DEFAULT_PEAK_RULES)
    args = parser.parse_args()

    cfg = load_config(args.config)
    diagnostics_dir = output_dir(cfg) / "diagnostics"
    candidate_path = Path(args.candidate_table) if args.candidate_table else diagnostics_dir / "spacing_candidates.csv"
    try:
        candidates = read_candidate_table(candidate_path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    variants, summary = build_spacing_sensitivity_report(
        candidates,
        confidence_grid=args.min_confidence_grid,
        band_padding_grid=args.band_padding_grid,
        peak_rules=args.peak_rules,
    )
    out_dir = Path(args.output_dir) if args.output_dir else diagnostics_dir
    paths = write_sensitivity_outputs(variants, summary, out_dir)

    print(f"candidate_rows: {summary['candidate_rows']}")
    print(f"variant_count: {summary['variant_count']}")
    print(f"interpretation_class_counts: {summary['interpretation_class_counts']}")
    print(f"artefact_risk_flag_counts: {summary['artefact_risk_flag_counts']}")
    print(f"max_accepted_image_count: {summary['max_accepted_image_count']}")
    print(f"max_accepted_patch_count: {summary['max_accepted_patch_count']}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
