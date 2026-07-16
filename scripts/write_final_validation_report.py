#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.final_validation_report import write_final_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the final validation interpretation report from existing validation summaries."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--docs-dir")
    args = parser.parse_args()

    report, paths = write_final_validation_report(
        load_config(args.config),
        output_directory=args.output_dir,
        docs_directory=args.docs_dir,
    )
    final = report["final_interpretation"]
    print(f"oop_orientation_implementation: {final['oop_orientation_implementation']}")
    print(f"real_tissue_oop_as_expert_organisation_endpoint: {final['real_tissue_oop_as_expert_organisation_endpoint']}")
    print(f"striation_visibility: {final['striation_visibility']}")
    print(f"sarcomere_spacing: {final['sarcomere_spacing']}")
    print(f"json: {paths['json']}")
    print(f"txt: {paths['txt']}")
    print(f"markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
