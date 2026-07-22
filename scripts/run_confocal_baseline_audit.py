#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.confocal_intake import run_confocal_baseline_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a non-destructive baseline audit on confocal pilot images.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--confocal-root", default="/Users/medhasharma/sarcomere_tools/Confocal")
    parser.add_argument("--output-dir")
    parser.add_argument("--write-previews", action="store_true")
    args = parser.parse_args()

    manifest, per_image, per_patch, summary, paths = run_confocal_baseline_audit(
        load_config(args.config),
        confocal_root=args.confocal_root,
        output_directory=args.output_dir,
        write_previews=args.write_previews,
    )
    print(f"confocal_images_found: {len(manifest)}")
    print(f"processed_ok: {summary['processed_ok']}")
    print(f"processed_error: {summary['processed_error']}")
    print(f"patch_rows: {len(per_patch)}")
    print(f"spacing_calibration_status: {summary['spacing_calibration_status']}")
    if not manifest.empty:
        print("filenames:")
        for filename in manifest["filename"].astype(str).tolist():
            print(f"- {filename}")
    print(f"manifest: {paths['manifest']}")
    print(f"per_image: {paths['per_image']}")
    print(f"per_patch: {paths['per_patch']}")
    print(f"summary_json: {paths['summary_json']}")
    print(f"summary_txt: {paths['summary_txt']}")
    if args.write_previews:
        print(f"preview_count: {len(summary['preview_paths'])}")
        print(f"preview_dir: {paths['previews']}")


if __name__ == "__main__":
    main()
