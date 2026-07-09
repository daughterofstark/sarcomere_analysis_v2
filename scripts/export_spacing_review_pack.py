#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.diagnostics.spacing_review_pack import (
    REVIEW_CLASSES,
    export_review_pack,
)


def parse_image_ids(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    image_ids: set[str] = set()
    for value in values:
        image_ids.update(item.strip() for item in value.split(",") if item.strip())
    return image_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Export diagnostic spacing candidate review panels.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--candidate-table")
    parser.add_argument("--patch-table")
    parser.add_argument("--output-dir")
    parser.add_argument("--classes", nargs="+", choices=REVIEW_CLASSES, default=REVIEW_CLASSES)
    parser.add_argument("--image-id", action="append", help="Image id subset; may be repeated or comma-separated.")
    parser.add_argument("--max-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    started = perf_counter()
    index, summary, paths = export_review_pack(
        cfg,
        candidate_table=args.candidate_table,
        patch_table=args.patch_table,
        output_directory=args.output_dir,
        classes=args.classes,
        max_per_class=args.max_per_class,
        seed=args.seed,
        image_ids=parse_image_ids(args.image_id),
        overwrite=args.overwrite,
    )

    print(f"candidate_rows: {summary['candidate_rows']}")
    print(f"candidate_image_count: {summary['candidate_image_count']}")
    print(f"manifest_image_count: {summary['manifest_image_count']}")
    print(f"review_is_limited_to_candidate_images: {summary['review_is_limited_to_candidate_images']}")
    print(f"selected_rows: {summary['selected_rows']}")
    print(f"rendered_panels: {summary['rendered_panels']}")
    print(f"render_errors: {summary['render_errors']}")
    print(f"selected_counts_by_class: {summary['selected_counts_by_class']}")
    print(f"missing_classes: {summary['missing_classes']}")
    print(f"runtime_seconds: {perf_counter() - started:.3f}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")
    if not index.empty:
        print(f"panel_directory: {Path(summary['output_dir'])}")


if __name__ == "__main__":
    main()
