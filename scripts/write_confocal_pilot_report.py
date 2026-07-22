#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_pilot_report import write_confocal_pilot_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the confocal pilot interpretation report.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--docs-dir")
    args = parser.parse_args()

    report, paths = write_confocal_pilot_report(
        load_config(args.config),
        output_directory=args.output_dir,
        docs_directory=args.docs_dir,
    )
    print(f"final_confocal_pilot_classification: {report['final_confocal_pilot_classification']}")
    print(f"image_count: {report['confocal_dataset_intake'].get('image_count')}")
    print(f"moderate_gate_classification: {report['selective_confident_striation_mask'].get('moderate_gate_classification')}")
    print(f"selected_vs_all_oop_summary: {report['same_grid_selected_region_oop'].get('selected_vs_all_oop_summary')}")
    print(f"json: {paths['json']}")
    print(f"txt: {paths['txt']}")
    print(f"markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
