#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config
from sarcomere_analysis.metadata import (
    default_metadata_output_dir,
    enrich_manifest,
    load_external_metadata,
    load_manifest_table,
    write_metadata_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create enriched manifest and donor metadata tables.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--metadata")
    parser.add_argument("--output-dir")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = load_manifest_table(cfg, args.manifest)
    external_metadata = load_external_metadata(args.metadata) if args.metadata else None
    enriched, donor_metadata, summary = enrich_manifest(
        manifest,
        cfg,
        metadata=external_metadata,
        strict=args.strict,
    )
    out_dir = Path(args.output_dir) if args.output_dir else default_metadata_output_dir(cfg)
    paths = write_metadata_outputs(enriched, donor_metadata, summary, out_dir)

    print(f"enriched_manifest_rows: {summary['enriched_manifest_rows']}")
    print(f"donor_metadata_rows: {summary['donor_metadata_rows']}")
    print(f"donor_count: {summary['donor_count']}")
    print(f"healthy_donor_count: {summary['healthy_donor_count']}")
    print(f"unmatched_manifest_donors: {summary['unmatched_manifest_donors']}")
    print(f"unmatched_metadata_donors: {summary['unmatched_metadata_donors']}")
    if summary["warnings"]:
        print(f"warnings: {summary['warnings']}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
