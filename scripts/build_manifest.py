#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import get_calibration, load_config, manifest_csv_path
from sarcomere_analysis.io import build_manifest, write_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TIFF manifest for sarcomere analysis.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image-dir", help="Override paths.raw_tiff_dir from the config.")
    parser.add_argument("--output", help="Override output manifest path.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing.")
    parser.add_argument("--allow-empty", action="store_true", help="Allow an empty manifest without exiting nonzero.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.image_dir:
        config["paths"]["raw_tiff_dir"] = args.image_dir
    output_path = Path(args.output) if args.output else manifest_csv_path(config)
    manifest = build_manifest(config)
    calibration = get_calibration(config)

    print(f"Discovered images: {len(manifest)}")
    print(f"Donors: {manifest['donor_id'].nunique() if not manifest.empty else 0}")
    print(f"Pixel size: {calibration.pixel_size_um:.4f} um/px")
    print(
        "Expected spacing band: "
        f"{calibration.expected_spacing_um_min:.2f}-"
        f"{calibration.expected_spacing_um_max:.2f} um "
        f"({calibration.expected_spacing_px_min:.2f}-"
        f"{calibration.expected_spacing_px_max:.2f} px)"
    )
    if not manifest.empty:
        print(manifest.head(10).to_string(index=False))
    elif not args.allow_empty:
        print("No TIFFs found. Use --allow-empty to permit an empty manifest.", file=sys.stderr)
        raise SystemExit(1)

    if args.dry_run:
        print(f"\nDry run only. Would write: {output_path}")
        return

    write_table(manifest, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
