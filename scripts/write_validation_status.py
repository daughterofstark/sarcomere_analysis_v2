#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.validation_status import write_validation_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a concise validation status summary from existing validation outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--docs-dir")
    args = parser.parse_args()

    status, paths = write_validation_status(
        load_config(args.config),
        output_directory=args.output_dir,
        docs_directory=args.docs_dir,
    )
    print(f"synthetic_oop_status: {status['synthetic_oop_validation']['status']}")
    print(f"manual_crop_status: {status['manual_crop_zdisc_mask_validation']['status']}")
    print(f"manual_full_image_status: {status['manual_full_image_zdisc_mask_validation']['status']}")
    print(f"manual_full_image_patch_status: {status['manual_full_image_patch_mask_validation']['status']}")
    print(f"spacing_status: {status['spacing']['status']}")
    print(f"overall_real_tissue_oop: {status['overall_validation_decision']['real_tissue_oop']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    print(f"markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
