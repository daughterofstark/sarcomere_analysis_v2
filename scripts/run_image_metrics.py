#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config, manifest_csv_path
from sarcomere_analysis.io import build_manifest, parse_image_filename
from sarcomere_analysis.pipeline import WriteOptions, run_single_image


def resolve_image_path(
    config: dict,
    image_path: str | None,
    image_id: str | None,
    manifest_override: str | None = None,
) -> tuple[Path, str, str | None]:
    if image_path is not None:
        path = Path(image_path)
        donor_id = None
        try:
            donor_id = parse_image_filename(path, str(config["filename_pattern"]["regex"])).get("donor_id")
        except ValueError:
            pass
        return path, path.stem, donor_id

    manifest_path = Path(manifest_override) if manifest_override is not None else manifest_csv_path(config)
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, dtype={"image_id": str, "donor_id": str, "region_id": str})
    else:
        manifest = build_manifest(config)

    matches = manifest.loc[manifest["image_id"] == image_id]
    if matches.empty:
        raise ValueError(f"Image id not found in manifest: {image_id}")
    path = Path(str(matches.iloc[0]["image_path"]))
    donor_id = matches.iloc[0].get("donor_id")
    return path, str(image_id), str(donor_id) if pd.notna(donor_id) else None


def print_image_summary(result) -> None:
    metrics = result.image_metrics
    print(f"image_id: {result.image_id}")
    print(f"donor_id: {result.donor_id}")
    print("Patch QC:")
    print(f"total_patches: {metrics['total_patches']}")
    print(f"valid_orientation_patches: {metrics['valid_orientation_patches']}")
    print("Orientation/OOP:")
    print(f"image_oop: {metrics['image_oop']}")
    print(f"image_oop_heterogeneity: {metrics['image_oop_heterogeneity']}")
    print("Spacing scaffold:")
    print(f"valid_spacing_patches: {metrics['n_spacing_valid_patches']}")
    print(f"n_spacing_valid_patches: {metrics['n_spacing_valid_patches']}")
    print(f"spacing_valid_fraction: {metrics['spacing_valid_fraction']}")
    print(f"image_spacing_median_um: {metrics['image_spacing_median_um']}")
    print(f"image_spacing_mean_um: {metrics['image_spacing_mean_um']}")
    print(f"runtime_seconds: {result.runtime_seconds:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image metrics for one image.")
    parser.add_argument("--config", default="configs/default.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to one TIFF image.")
    source.add_argument("--image-id", help="Image id to load from the manifest.")
    parser.add_argument("--manifest", help="Manifest CSV override.")
    parser.add_argument("--write-preview", action="store_true", help="Write standardized PNG previews.")
    parser.add_argument("--write-tables", action="store_true", help="Write standardized per-patch and per-image tables.")
    parser.add_argument("--write-provenance", action="store_true", help="Write run provenance JSON.")
    parser.add_argument("--write-all", action="store_true", help="Write tables, previews, and provenance.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    write_preview = bool(args.write_preview or args.write_all)
    write_tables = bool(args.write_tables or args.write_all)
    write_provenance = bool(args.write_provenance or args.write_all)

    image_path, image_id, donor_id = resolve_image_path(cfg, args.image, args.image_id, args.manifest)
    result = run_single_image(
        image_path=image_path,
        image_id=image_id,
        donor_id=donor_id,
        cfg=cfg,
        write_options=WriteOptions(tables=write_tables, preview=write_preview, provenance=write_provenance),
        config_path=args.config,
    )

    print_image_summary(result)
    for label, path in result.output_paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
