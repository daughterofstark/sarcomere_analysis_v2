#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config, manifest_csv_path, output_dir
from sarcomere_analysis.io import build_manifest, load_tiff
from sarcomere_analysis.masking import compute_tissue_mask
from sarcomere_analysis.orientation import compute_orientation_analysis
from sarcomere_analysis.preprocessing import preprocess_image
from sarcomere_analysis.qc import compute_patch_qc
from sarcomere_analysis.spacing.diagnostics import (
    SPACING_DIAGNOSTIC_COLUMNS,
    diagnose_spacing_analysis,
    summarize_spacing_diagnostics,
    write_autocorrelation_debug_plot,
    write_diagnostic_tables,
    write_spacing_confidence_heatmap,
)


def parse_image_id_subset(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    image_ids: set[str] = set()
    for value in values:
        image_ids.update(item.strip() for item in value.split(",") if item.strip())
    return image_ids


def load_manifest(cfg: dict, manifest_override: str | None) -> pd.DataFrame:
    path = Path(manifest_override) if manifest_override else manifest_csv_path(cfg)
    if path.exists():
        return pd.read_csv(path, dtype={"image_id": str, "donor_id": str, "region_id": str})
    return build_manifest(cfg)


def select_manifest_rows(manifest: pd.DataFrame, image_ids: set[str] | None, limit: int | None) -> pd.DataFrame:
    selected = manifest.copy()
    if image_ids is not None:
        selected = selected.loc[selected["image_id"].isin(image_ids)].copy()
    if limit is not None:
        selected = selected.head(limit).copy()
    return selected.reset_index(drop=True)


def diagnostics_output_dir(cfg: dict, output_dir_override: str | None) -> Path:
    if output_dir_override is not None:
        return Path(output_dir_override)
    return output_dir(cfg) / "diagnostics"


def existing_diagnostics_for_selection(cfg: dict, selected: pd.DataFrame) -> pd.DataFrame | None:
    path = output_dir(cfg) / "tables" / "per_patch_metrics.csv"
    if not path.exists():
        return None
    header = pd.read_csv(path, nrows=0)
    if not set(SPACING_DIAGNOSTIC_COLUMNS).issubset(set(header.columns)):
        return None
    table = pd.read_csv(path, dtype={"image_id": str, "donor_id": str})
    image_ids = set(selected["image_id"].astype(str))
    return table.loc[table["image_id"].isin(image_ids), SPACING_DIAGNOSTIC_COLUMNS].reset_index(drop=True)


def recompute_image_diagnostics(row: pd.Series, cfg: dict, write_confidence_heatmap: bool, out_dir: Path) -> pd.DataFrame:
    image_id = str(row["image_id"])
    donor_id = str(row["donor_id"]) if "donor_id" in row and pd.notna(row["donor_id"]) else ""
    raw = load_tiff(Path(str(row["image_path"])))
    preprocessing = preprocess_image(raw, cfg)
    mask = compute_tissue_mask(preprocessing.image, cfg)
    patch_qc = compute_patch_qc(preprocessing.image, mask.mask, image_id, cfg)
    orientation = compute_orientation_analysis(preprocessing.image, mask.mask, patch_qc, cfg)
    patch_metrics = orientation.patch_metrics.copy()
    if "donor_id" not in patch_metrics.columns:
        patch_metrics.insert(1, "donor_id", donor_id)
    diagnostics = diagnose_spacing_analysis(preprocessing.image, patch_metrics, cfg)
    if write_confidence_heatmap:
        write_spacing_confidence_heatmap(diagnostics, preprocessing.image.shape, image_id, cfg, out_dir)
    return diagnostics


def recompute_image_context(row: pd.Series, cfg: dict) -> tuple:
    image_id = str(row["image_id"])
    donor_id = str(row["donor_id"]) if "donor_id" in row and pd.notna(row["donor_id"]) else ""
    raw = load_tiff(Path(str(row["image_path"])))
    preprocessing = preprocess_image(raw, cfg)
    mask = compute_tissue_mask(preprocessing.image, cfg)
    patch_qc = compute_patch_qc(preprocessing.image, mask.mask, image_id, cfg)
    orientation = compute_orientation_analysis(preprocessing.image, mask.mask, patch_qc, cfg)
    patch_metrics = orientation.patch_metrics.copy()
    if "donor_id" not in patch_metrics.columns:
        patch_metrics.insert(1, "donor_id", donor_id)
    return preprocessing.image, patch_metrics


def run_spacing_diagnostics(
    cfg: dict,
    manifest_override: str | None = None,
    image_ids: set[str] | None = None,
    limit: int | None = None,
    output_dir_override: str | None = None,
    write_patch_diagnostics: bool = False,
    write_summary: bool = False,
    write_confidence_heatmaps: bool = False,
    continue_on_error: bool = False,
) -> tuple[pd.DataFrame, dict[str, Path], list[dict[str, object]]]:
    manifest = load_manifest(cfg, manifest_override)
    selected = select_manifest_rows(manifest, image_ids, limit)
    if selected.empty:
        raise SystemExit("No images selected for spacing diagnostics.")

    out_dir = diagnostics_output_dir(cfg, output_dir_override)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = existing_diagnostics_for_selection(cfg, selected)
    if existing is not None:
        diagnostics = existing
        errors: list[dict[str, object]] = []
    else:
        frames = []
        errors = []
        for _, row in selected.iterrows():
            image_id = str(row["image_id"])
            donor_id = str(row["donor_id"]) if "donor_id" in row and pd.notna(row["donor_id"]) else ""
            started = perf_counter()
            try:
                frame = recompute_image_diagnostics(row, cfg, write_confidence_heatmaps, out_dir)
                frames.append(frame)
                print(f"diagnosed {image_id} in {perf_counter() - started:.3f}s")
            except Exception as exc:
                error = {
                    "image_id": image_id,
                    "donor_id": donor_id,
                    "error_message": str(exc),
                    "runtime_seconds": perf_counter() - started,
                }
                errors.append(error)
                if not continue_on_error:
                    raise
        diagnostics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SPACING_DIAGNOSTIC_COLUMNS)

    paths: dict[str, Path] = {}
    if write_summary or write_patch_diagnostics:
        paths = write_diagnostic_tables(diagnostics, out_dir, write_patch_diagnostics=write_patch_diagnostics)
    return diagnostics, paths, errors


def print_summary(diagnostics: pd.DataFrame, paths: dict[str, Path], errors: list[dict[str, object]]) -> None:
    summary, _ = summarize_spacing_diagnostics(diagnostics)
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    print(f"diagnostic_patch_rows: {len(diagnostics)}")
    print(f"errors: {len(errors)}")
    for key in [
        "accepted_spacing_count",
        "accepted_spacing_fraction",
        "accepted_selected_lag_px_median",
        "accepted_selected_lag_um_median",
        "accepted_near_lower_bound_count",
        "accepted_near_lower_bound_fraction",
        "accepted_near_upper_bound_count",
        "accepted_near_upper_bound_fraction",
        "top_rejection_stages",
        "top_invalid_reasons",
    ]:
        print(f"{key}: {row.get(key)}")
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")
    if errors:
        print("error_images:")
        for error in errors[:10]:
            print(f"{error['image_id']}: {error['error_message']}")


def write_debug_plot_for_patch(
    cfg: dict,
    manifest_override: str | None,
    image_id: str,
    patch_id: str,
    out_dir: Path,
) -> Path:
    manifest = load_manifest(cfg, manifest_override)
    selected = manifest.loc[manifest["image_id"].astype(str) == str(image_id)]
    if selected.empty:
        raise ValueError(f"debug image_id not found in manifest: {image_id}")
    image, patch_metrics = recompute_image_context(selected.iloc[0], cfg)
    matches = patch_metrics.loc[patch_metrics["patch_id"].astype(str) == str(patch_id)]
    if matches.empty:
        raise ValueError(f"debug patch_id not found for {image_id}: {patch_id}")
    row = matches.iloc[0]
    patch = image[int(row["y0"]) : int(row["y1"]), int(row["x0"]) : int(row["x1"])]
    return write_autocorrelation_debug_plot(
        patch,
        float(row["patch_mean_orientation_rad"]),
        cfg,
        str(image_id),
        str(patch_id),
        out_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose current spacing accept/reject behavior without changing metrics.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image-id", action="append", help="Optional image id subset; may be repeated or comma-separated.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--manifest", help="Manifest CSV override.")
    parser.add_argument("--output-dir", help="Diagnostics output directory. Defaults to config output_dir/diagnostics.")
    parser.add_argument("--write-patch-diagnostics", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--write-confidence-heatmaps", action="store_true")
    parser.add_argument("--debug-image-id", help="Image id for one autocorrelation debug plot.")
    parser.add_argument("--debug-patch-id", help="Patch id for one autocorrelation debug plot.")
    parser.add_argument("--write-autocorr-debug", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    diagnostics, paths, errors = run_spacing_diagnostics(
        cfg,
        manifest_override=args.manifest,
        image_ids=parse_image_id_subset(args.image_id),
        limit=args.limit,
        output_dir_override=args.output_dir,
        write_patch_diagnostics=args.write_patch_diagnostics,
        write_summary=args.write_summary,
        write_confidence_heatmaps=args.write_confidence_heatmaps,
        continue_on_error=args.continue_on_error,
    )
    print_summary(diagnostics, paths, errors)
    if args.write_autocorr_debug:
        if not args.debug_image_id or not args.debug_patch_id:
            raise SystemExit("--write-autocorr-debug requires --debug-image-id and --debug-patch-id")
        path = write_debug_plot_for_patch(
            cfg,
            args.manifest,
            args.debug_image_id,
            args.debug_patch_id,
            diagnostics_output_dir(cfg, args.output_dir),
        )
        print(f"Wrote autocorr_debug: {path}")


if __name__ == "__main__":
    main()
