#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.manual_spacing_assist import (
    default_manual_spacing_output_csv,
    default_manual_spacing_panel_path,
    interactive_line_from_clicks,
    run_manual_spacing_assist,
    write_diagnostic_panel,
    write_manual_spacing_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assist manual sarcomere spacing measurement on one annotation crop.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--crop", required=True)
    parser.add_argument("--x0", type=float)
    parser.add_argument("--y0", type=float)
    parser.add_argument("--x1", type=float)
    parser.add_argument("--y1", type=float)
    parser.add_argument("--interactive-clicks", action="store_true")
    parser.add_argument("--intervals", type=int, help="User-counted number of intervals along the line.")
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--min-peak-prominence", type=float)
    parser.add_argument("--confidence-score")
    parser.add_argument("--accepted-by-user", choices=["yes", "no", "unsure"], default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--write-panel", action="store_true")
    parser.add_argument("--panel-path")
    parser.add_argument("--output-csv")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.interactive_clicks:
        x0, y0, x1, y1 = interactive_line_from_clicks(args.crop)
    else:
        coords = [args.x0, args.y0, args.x1, args.y1]
        if any(value is None for value in coords):
            parser.error("Provide --x0 --y0 --x1 --y1, or use --interactive-clicks.")
        x0, y0, x1, y1 = (float(value) for value in coords)

    row, result = run_manual_spacing_assist(
        cfg,
        crop_path=args.crop,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        intervals=args.intervals,
        smooth_sigma=args.smooth_sigma,
        min_peak_prominence=args.min_peak_prominence,
        confidence_score=args.confidence_score,
        accepted_by_user=args.accepted_by_user,
        notes=args.notes,
    )
    output_csv = Path(args.output_csv) if args.output_csv else default_manual_spacing_output_csv(cfg)
    csv_path = write_manual_spacing_result(row, output_csv, append=args.append)
    panel_path = None
    if args.write_panel:
        panel_path = Path(args.panel_path) if args.panel_path else default_manual_spacing_panel_path(cfg, args.crop)
        write_diagnostic_panel(result, panel_path, row)

    print(f"annotation_id: {row['annotation_id']}")
    print(f"crop_path: {row['crop_path']}")
    print(f"line_length_px: {row['line_length_px']:.3f}")
    print(f"line_length_um: {row['line_length_um']:.3f}")
    print(f"detected_peak_count: {row['detected_peak_count']}")
    print(f"estimated_spacing_px: {row['estimated_spacing_px']}")
    print(f"estimated_spacing_um: {row['estimated_spacing_um']}")
    print(f"Wrote result_csv: {csv_path}")
    if panel_path is not None:
        print(f"Wrote panel: {panel_path}")


if __name__ == "__main__":
    main()
