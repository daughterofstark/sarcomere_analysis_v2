#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.annotation_ui import headless_check, run_annotation_ui
from sarcomere_analysis.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate the local OOP/orientation crop pack.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--start", help="Annotation ID to start from, e.g. ANN_0001.")
    parser.add_argument("--overwrite", action="store_true", help="Reinitialize annotation_filled.csv from the template.")
    parser.add_argument("--output-csv", help="Optional annotation output CSV path.")
    parser.add_argument("--headless-check", action="store_true", help="Validate pack paths without opening the UI.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.headless_check:
        summary = headless_check(cfg, output_csv=args.output_csv)
        print(f"index_path: {summary['index_path']}")
        print(f"template_path: {summary['template_path']}")
        print(f"output_csv: {summary['output_csv']}")
        print(f"autosave_csv: {summary['autosave_csv']}")
        print(f"annotation_rows: {summary['annotation_rows']}")
        print(f"index_rows: {summary['index_rows']}")
        print(f"template_rows: {summary['template_rows']}")
        print(f"crop_count: {summary['crop_count']}")
        print(f"missing_crop_count: {summary['missing_crop_count']}")
        if summary["missing_crop_count"]:
            print(f"missing_crops_first_10: {summary['missing_crops']}")
            raise SystemExit(1)
        return

    output_path = run_annotation_ui(
        cfg,
        start_annotation_id=args.start,
        output_csv=args.output_csv,
        overwrite=args.overwrite,
    )
    print(f"Annotation output: {output_path}")


if __name__ == "__main__":
    main()
