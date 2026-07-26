#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_gate_decision import write_confocal_gate_decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the confocal gate visual-review decision record.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--docs-dir")
    args = parser.parse_args()

    decision, paths = write_confocal_gate_decision(
        load_config(args.config),
        output_directory=args.output_dir,
        docs_directory=args.docs_dir,
    )
    print(f"primary_gate: {decision['primary_gate']}")
    print(f"secondary_sensitivity_gate: {decision['secondary_sensitivity_gate']}")
    print("final_decision: keep moderate primary; relaxed_combined sensitivity only")
    print(f"json: {paths['json']}")
    print(f"txt: {paths['txt']}")
    print(f"markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
