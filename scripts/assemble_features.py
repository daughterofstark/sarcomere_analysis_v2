#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config, output_dir
from sarcomere_analysis.features import (
    assemble_feature_tables,
    load_feature_inputs,
    write_feature_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble OOP-first feature tables from existing pipeline outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--patch-table")
    parser.add_argument("--image-table")
    parser.add_argument("--batch-summary")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-spacing-patches-per-image", type=int, default=5)
    parser.add_argument("--min-spacing-patches-per-donor", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    patches, images, batch = load_feature_inputs(
        cfg,
        patch_table=args.patch_table,
        image_table=args.image_table,
        batch_summary=args.batch_summary,
    )
    per_patch, per_image, per_donor, summary = assemble_feature_tables(
        patches,
        images,
        batch,
        min_spacing_patches_per_image=args.min_spacing_patches_per_image,
        min_spacing_patches_per_donor=args.min_spacing_patches_per_donor,
    )
    out_dir = Path(args.output_dir) if args.output_dir else output_dir(cfg) / "tables"
    paths = write_feature_outputs(per_patch, per_image, per_donor, summary, out_dir)

    print(f"per_patch_rows: {summary['per_patch_rows']}")
    print(f"per_image_rows: {summary['per_image_rows']}")
    print(f"per_donor_rows: {summary['per_donor_rows']}")
    print(f"donor_count: {summary['donor_count']}")
    print(f"spacing_global_status: {summary['spacing_global_status']}")
    print(f"images_with_insufficient_spacing_yield: {summary['images_with_insufficient_spacing_yield']}")
    print(f"donors_with_insufficient_spacing_yield: {summary['donors_with_insufficient_spacing_yield']}")
    if summary["warnings"]:
        print(f"warnings: {summary['warnings']}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
