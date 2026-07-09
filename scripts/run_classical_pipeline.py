#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config, output_dir
from sarcomere_analysis.pipeline import run_classical_pipeline, write_pipeline_run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible classical sarcomere analysis pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--with-previews", action="store_true")
    parser.add_argument("--with-spacing-diagnostics", action="store_true")
    parser.add_argument("--with-validation-template", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    config_for_steps = Path(args.config)
    if args.output_dir:
        cfg, config_for_steps = effective_config_for_output_dir(cfg, args.config, args.output_dir, dry_run=args.dry_run)

    summary, _ = run_classical_pipeline(
        cfg,
        config_path=config_for_steps,
        project_root=project_root,
        with_previews=args.with_previews,
        with_spacing_diagnostics=args.with_spacing_diagnostics,
        with_validation_template=args.with_validation_template,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
        force=args.force,
        skip_existing=args.skip_existing,
        python_executable=sys.executable,
    )

    if args.dry_run:
        print("Dry run: planned classical pipeline steps")
        for step in summary["steps"]:
            print(f"- {step['name']}: {step['status']}")
            print(f"  command: {' '.join(step['command'])}")
            for path in step["expected_outputs"]:
                print(f"  output: {path}")
        return

    paths = write_pipeline_run_summary(summary, cfg)
    print("Classical pipeline complete")
    for step in summary["steps"]:
        print(f"{step['name']}: {step['status']} ({step['runtime_seconds']:.3f}s)")
    print(f"row_counts: {summary['row_counts']}")
    print(f"spacing_global_status: {summary['spacing_global_status']}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")


def effective_config_for_output_dir(
    cfg: dict,
    config_path: str | Path,
    override_output_dir: str | Path,
    dry_run: bool,
) -> tuple[dict, Path]:
    effective = dict(cfg)
    effective["paths"] = dict(cfg["paths"])
    effective["outputs"] = dict(cfg["outputs"])
    effective["paths"]["output_dir"] = str(Path(override_output_dir))
    effective["outputs"]["manifest_csv"] = str(Path(override_output_dir) / "tables" / "manifest.csv")
    if dry_run:
        return effective, Path(config_path)
    out_dir = output_dir(effective)
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_path = out_dir / "pipeline_effective_config.yaml"
    with effective_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(effective, handle, sort_keys=False)
    return effective, effective_path


if __name__ == "__main__":
    main()
