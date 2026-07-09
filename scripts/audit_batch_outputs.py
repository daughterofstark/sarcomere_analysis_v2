#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sarcomere_analysis.config import load_config, manifest_csv_path, output_dir
from sarcomere_analysis.io import build_manifest
from sarcomere_analysis.schemas import (
    BATCH_RUN_SUMMARY_COLUMNS,
    IMAGE_METRICS_COLUMNS,
    PATCH_METRICS_COLUMNS,
)


KEY_IMAGE_METRICS = [
    "image_oop",
    "image_oop_heterogeneity",
    "image_spacing_median_um",
    "image_spacing_mean_um",
]

SUMMARY_COLUMNS = [
    "total_patches",
    "valid_orientation_patches",
    "n_spacing_valid_patches",
    "spacing_valid_fraction",
    "tissue_fraction",
    "image_oop",
    "image_oop_heterogeneity",
    "image_spacing_median_um",
    "image_spacing_mean_um",
]


def audit_batch_outputs(
    cfg: dict[str, Any],
    manifest_path: str | Path | None = None,
    tables_dir: str | Path | None = None,
) -> dict[str, Any]:
    tables = Path(tables_dir) if tables_dir is not None else output_dir(cfg) / "tables"
    manifest = load_manifest_for_audit(cfg, manifest_path)
    batch_summary = read_csv_if_exists(tables / "batch_run_summary.csv")
    per_image = read_csv_if_exists(tables / "per_image_metrics.csv")
    per_patch = read_csv_if_exists(tables / "per_patch_metrics.csv")

    missing_columns = {
        "batch_run_summary": missing_columns_for(batch_summary, BATCH_RUN_SUMMARY_COLUMNS),
        "per_image_metrics": missing_columns_for(per_image, IMAGE_METRICS_COLUMNS),
        "per_patch_metrics": missing_columns_for(per_patch, PATCH_METRICS_COLUMNS),
    }
    error_rows = batch_summary.loc[batch_summary.get("status", pd.Series(dtype=str)) == "error"].copy()

    audit = {
        "total_images_expected_from_manifest": int(len(manifest)),
        "total_images_processed": int(len(batch_summary)),
        "number_ok": int((batch_summary.get("status", pd.Series(dtype=str)) == "ok").sum()),
        "number_error": int((batch_summary.get("status", pd.Series(dtype=str)) == "error").sum()),
        "errors": error_rows[["image_id", "error_message"]].fillna("").to_dict(orient="records")
        if {"image_id", "error_message"}.issubset(error_rows.columns)
        else [],
        "total_patch_rows": int(len(per_patch)),
        "per_image_row_count": int(len(per_image)),
        "missing_required_columns": missing_columns,
        "nan_rates": nan_rates(per_image, KEY_IMAGE_METRICS),
        "summary_distribution": summary_distribution(per_image, SUMMARY_COLUMNS),
        "top_spacing_invalid_reason_values": value_counts(per_patch, "spacing_invalid_reason"),
        "top_patch_invalid_reason_values": value_counts(per_patch, "invalid_reason"),
        "donor_count_represented": int(per_image["donor_id"].nunique()) if "donor_id" in per_image.columns else 0,
        "images_per_donor_distribution": value_counts(per_image, "donor_id", limit=None),
    }
    return json_safe(audit)


def load_manifest_for_audit(cfg: dict[str, Any], manifest_path: str | Path | None) -> pd.DataFrame:
    path = Path(manifest_path) if manifest_path is not None else manifest_csv_path(cfg)
    if path.exists():
        return pd.read_csv(path, dtype={"image_id": str, "donor_id": str, "region_id": str})
    return build_manifest(cfg)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"image_id": str, "donor_id": str, "patch_id": str})


def missing_columns_for(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [column for column in required if column not in df.columns]


def nan_rates(df: pd.DataFrame, columns: list[str]) -> dict[str, float | None]:
    rates: dict[str, float | None] = {}
    for column in columns:
        if column not in df.columns or len(df) == 0:
            rates[column] = None
        else:
            rates[column] = float(df[column].isna().mean())
    return rates


def summary_distribution(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | None]]:
    distribution: dict[str, dict[str, float | None]] = {}
    for column in columns:
        if column not in df.columns or len(df) == 0:
            distribution[column] = empty_distribution()
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            distribution[column] = empty_distribution()
            continue
        distribution[column] = {
            "min": float(finite.min()),
            "p25": float(finite.quantile(0.25)),
            "median": float(finite.median()),
            "mean": float(finite.mean()),
            "p75": float(finite.quantile(0.75)),
            "max": float(finite.max()),
        }
    return distribution


def empty_distribution() -> dict[str, None]:
    return {"min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}


def value_counts(df: pd.DataFrame, column: str, limit: int | None = 20) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].fillna("<NA>").astype(str).value_counts()
    if limit is not None:
        counts = counts.head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def write_audit_outputs(audit: dict[str, Any], cfg: dict[str, Any]) -> tuple[Path, Path]:
    tables = output_dir(cfg) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    json_path = tables / "batch_audit_summary.json"
    txt_path = tables / "batch_audit_summary.txt"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    txt_path.write_text(format_text_summary(audit), encoding="utf-8")
    return json_path, txt_path


def format_text_summary(audit: dict[str, Any]) -> str:
    lines = [
        "Batch Audit Summary",
        f"expected_images: {audit['total_images_expected_from_manifest']}",
        f"processed_images: {audit['total_images_processed']}",
        f"ok: {audit['number_ok']}",
        f"errors: {audit['number_error']}",
        f"per_image_rows: {audit['per_image_row_count']}",
        f"patch_rows: {audit['total_patch_rows']}",
        f"donor_count: {audit['donor_count_represented']}",
        f"nan_rates: {audit['nan_rates']}",
        f"top_spacing_invalid_reason_values: {audit['top_spacing_invalid_reason_values']}",
        f"top_patch_invalid_reason_values: {audit['top_patch_invalid_reason_values']}",
    ]
    if audit["errors"]:
        lines.append(f"errors_detail: {audit['errors']}")
    return "\n".join(lines) + "\n"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit combined batch outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", help="Manifest CSV override.")
    parser.add_argument("--tables-dir", help="Directory containing combined output CSVs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    audit = audit_batch_outputs(cfg, args.manifest, args.tables_dir)
    json_path, txt_path = write_audit_outputs(audit, cfg)
    print(format_text_summary(audit), end="")
    print(f"Wrote audit_json: {json_path}")
    print(f"Wrote audit_txt: {txt_path}")


if __name__ == "__main__":
    main()
