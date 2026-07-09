#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.qc_gallery import (
    build_qc_gallery_index,
    write_gallery_html,
    write_gallery_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a QC gallery index from existing outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sort-by", default="spacing_valid_fraction")
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--ascending", action="store_true")
    direction.add_argument("--descending", action="store_true")
    parser.add_argument("--require-existing-previews", action="store_true")
    parser.add_argument("--write-html", action="store_true")
    parser.add_argument("--write-index", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ascending = bool(args.ascending and not args.descending)
    index = build_qc_gallery_index(
        cfg,
        limit=args.limit,
        sort_by=args.sort_by,
        ascending=ascending,
        require_existing_previews=args.require_existing_previews,
    )
    print(f"indexed_images: {len(index)}")
    missing = index["qc_flag_summary"].astype(str).str.startswith("missing_previews").sum()
    print(f"images_with_missing_previews: {int(missing)}")

    if args.write_index:
        path = write_gallery_index(index, cfg)
        print(f"Wrote gallery_index: {path}")
    if args.write_html:
        path = write_gallery_html(index, cfg)
        print(f"Wrote gallery_html: {path}")


if __name__ == "__main__":
    main()
