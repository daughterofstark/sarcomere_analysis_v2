#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.validation_io import write_validation_template


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a manual/FIJI validation CSV template.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="templates/manual_validation_template.csv")
    args = parser.parse_args()

    _ = load_config(args.config)
    path = write_validation_template(args.output)
    print(f"Wrote validation template: {path}")
    print("Template rows are examples only. Replace them before real validation, or pass --allow-example-rows for a template audit.")


if __name__ == "__main__":
    main()
