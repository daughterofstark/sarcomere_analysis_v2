#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.share_ready_audit import run_share_ready_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit repository hygiene before GitHub sharing.")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir")
    parser.add_argument("--docs-dir")
    parser.add_argument("--large-threshold-mb", type=float, default=25.0)
    args = parser.parse_args()

    audit, paths = run_share_ready_audit(
        args.repo_root,
        output_directory=args.output_dir,
        docs_directory=args.docs_dir,
        large_threshold_bytes=int(args.large_threshold_mb * 1024 * 1024),
    )
    print(f"safe_to_push_git: {audit['safe_to_push_git']}")
    print(f"safe_to_share_folder_archive: {audit['safe_to_share_folder_archive']}")
    print(f"large_file_count: {audit['large_file_count']}")
    print(f"tracked_large_file_count: {audit['tracked_large_file_count']}")
    print(f"raw_microscopy_file_count: {audit['raw_microscopy_file_count']}")
    print(f"raw_microscopy_tracked_count: {audit['raw_microscopy_tracked_count']}")
    print(f"zip_file_count: {audit['zip_file_count']}")
    print(f"zip_tracked_count: {audit['zip_tracked_count']}")
    print(f"results_tracked_count: {audit['results_tracked_count']}")
    print(f"absolute_path_hit_count: {audit['absolute_path_hit_count']}")
    print(f"tracked_local_absolute_path_hit_count: {audit['tracked_local_absolute_path_hit_count']}")
    print(f"private_marker_hit_count: {audit['private_marker_hit_count']}")
    print(f"tracked_private_path_leakage_hit_count: {audit['tracked_private_path_leakage_hit_count']}")
    print(f"gitignore_missing_patterns: {audit['gitignore_check']['missing_patterns']}")
    print(f"json: {paths['json']}")
    print(f"txt: {paths['txt']}")
    print(f"markdown: {paths['markdown']}")


if __name__ == "__main__":
    main()
