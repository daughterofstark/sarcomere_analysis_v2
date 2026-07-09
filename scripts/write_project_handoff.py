#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config, output_dir
from sarcomere_analysis.project_audit import build_project_audit, write_project_audit_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Write project status audit and handoff snapshot.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--test-status", help='Recorded test status, for example "178 passed".')
    parser.add_argument("--output-dir", help="Override results output directory for audit files.")
    parser.add_argument("--docs-dir", help="Override docs output directory for Markdown handoff files.")
    parser.add_argument("--strict", action="store_true", help="Fail on safety-check problems. Enabled by default for core checks.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    if args.output_dir:
        cfg = dict(cfg)
        cfg["paths"] = dict(cfg["paths"])
        cfg["paths"]["output_dir"] = str(Path(args.output_dir))
    audit = build_project_audit(
        cfg,
        config_path=args.config,
        project_root=project_root,
        docs_dir=args.docs_dir,
        test_status=args.test_status,
        strict=True,
    )
    results_root = Path(args.output_dir) if args.output_dir else output_dir(cfg)
    docs_root = Path(args.docs_dir) if args.docs_dir else project_root / "docs"
    paths = write_project_audit_outputs(audit, results_root, docs_root)

    rows = {key: item.get("row_count") for key, item in audit["core_output_inventory"].items()}
    print(f"project_path: {audit['repository_state']['project_path']}")
    print(f"test_status: {audit['test_status']['provided_status']}")
    print(f"safety_checks_passed: {audit['safety_checks']['passed']}")
    print(f"row_counts: {rows}")
    print(f"spacing_status: {audit['safety_checks']['spacing_status']}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
