#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.zdisc_draw_ui import headless_check, run_draw_ui


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw local Z-disc/striation annotations directly onto crop masks.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--index", help="Path to zdisc_annotation_index.csv.")
    parser.add_argument("--start", help="Annotation ID to start from, e.g. ANN_0001.")
    parser.add_argument("--brush-size", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--overwrite-progress", action="store_true")
    parser.add_argument("--headless-check", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.headless_check:
        summary = headless_check(cfg, index_path=args.index)
        print(f"index_path: {summary['index_path']}")
        print(f"rows: {summary['rows']}")
        print(f"missing_image_count: {summary['missing_image_count']}")
        print(f"missing_mask_count: {summary['missing_mask_count']}")
        print(f"shape_mismatch_count: {summary['shape_mismatch_count']}")
        print(f"invalid_mask_count: {summary['invalid_mask_count']}")
        if any(summary[key] for key in ["missing_image_count", "missing_mask_count", "shape_mismatch_count", "invalid_mask_count"]):
            raise SystemExit(1)
        return

    progress_path = run_draw_ui(
        cfg,
        index_path=args.index,
        start_annotation_id=args.start,
        brush_size=args.brush_size,
        alpha=args.alpha,
        overwrite_progress=args.overwrite_progress,
    )
    print(f"Progress saved to: {progress_path}")


if __name__ == "__main__":
    main()
