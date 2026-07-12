#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.expert_annotation_pack import export_expert_annotation_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a blinded expert annotation pack for striation organisation review.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--patch-table")
    parser.add_argument("--image-table")
    parser.add_argument("--analysis-table")
    parser.add_argument("--manifest-table")
    parser.add_argument("--output-dir")
    parser.add_argument("--n-total", type=int, default=75)
    parser.add_argument("--n-per-bin", type=int)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-per-donor", type=int, default=4)
    parser.add_argument("--max-per-image", type=int, default=3)
    parser.add_argument("--write-zip", action="store_true")
    args = parser.parse_args()

    _, _, _, summary, paths = export_expert_annotation_pack(
        load_config(args.config),
        patch_table=args.patch_table,
        image_table=args.image_table,
        analysis_table=args.analysis_table,
        manifest_table=args.manifest_table,
        output_directory=args.output_dir,
        n_total=args.n_total,
        n_per_bin=args.n_per_bin,
        seed=args.seed,
        max_per_donor=args.max_per_donor,
        max_per_image=args.max_per_image,
        write_zip=args.write_zip,
    )

    print(f"selected_patches: {summary['selected_patches']}")
    print(f"oop_bin_counts: {summary['oop_bin_counts']}")
    print(f"bin_shortfall: {summary['bin_shortfall']}")
    print(f"unique_donors: {summary['unique_donors']}")
    print(f"unique_images: {summary['unique_images']}")
    print(f"max_patches_per_donor: {summary['max_patches_per_donor']}")
    print(f"max_patches_per_image: {summary['max_patches_per_image']}")
    print(f"template_csv: {paths['template_csv']}")
    print(f"internal_key_csv: {paths['internal_key_csv']}")
    print(f"instructions_md: {paths['instructions_md']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    print(f"patch_dir: {paths['patch_dir']}")
    if "zip" in paths:
        print(f"zip: {paths['zip']}")


if __name__ == "__main__":
    main()
