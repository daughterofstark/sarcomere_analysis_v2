from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import manifest_csv_path, output_dir


DEFAULT_HEALTHY_DONOR_IDS = ["4.083", "5.003", "6.052", "7.028"]

ENRICHED_MANIFEST_REQUIRED_COLUMNS = [
    "image_id",
    "donor_id",
    "is_healthy",
    "image_path",
]

DONOR_METADATA_REQUIRED_COLUMNS = [
    "donor_id",
    "is_healthy",
    "n_images",
]


def healthy_donor_ids_from_config(cfg: dict[str, Any]) -> list[str]:
    configured = cfg.get("metadata", {}).get("healthy_donor_ids", DEFAULT_HEALTHY_DONOR_IDS)
    return [standardize_id(value) for value in configured]


def load_manifest_table(cfg: dict[str, Any], manifest_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(manifest_path) if manifest_path else manifest_csv_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Manifest table not found: {path}")
    manifest = pd.read_csv(path, dtype={"image_id": str, "donor_id": str, "region_id": str})
    return standardize_manifest(manifest)


def load_external_metadata(path: str | Path) -> pd.DataFrame:
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata table not found: {metadata_path}")
    suffix = metadata_path.suffix.lower()
    if suffix == ".csv":
        table = pd.read_csv(metadata_path, dtype={"donor_id": str})
    elif suffix in {".xlsx", ".xls"}:
        try:
            table = pd.read_excel(metadata_path, dtype={"donor_id": str})
        except ImportError as exc:
            raise ImportError("Reading Excel metadata requires an installed Excel engine. Use CSV metadata instead.") from exc
    else:
        raise ValueError(f"Unsupported metadata file type: {metadata_path.suffix}. Use CSV or XLSX.")
    if "donor_id" not in table.columns:
        raise ValueError("External metadata table must contain a donor_id column.")
    result = table.copy()
    result["donor_id"] = result["donor_id"].map(standardize_id)
    return result


def enrich_manifest(
    manifest: pd.DataFrame,
    cfg: dict[str, Any],
    metadata: pd.DataFrame | None = None,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_clean = standardize_manifest(manifest)
    healthy_ids = set(healthy_donor_ids_from_config(cfg))
    manifest_clean["is_healthy"] = manifest_clean["donor_id"].isin(healthy_ids)

    donor_base = (
        manifest_clean.groupby("donor_id", as_index=False)
        .agg(n_images=("image_id", "nunique"), is_healthy=("is_healthy", "max"))
        .loc[:, ["donor_id", "is_healthy", "n_images"]]
    )

    metadata_clean = None
    metadata_columns: list[str] = []
    unmatched_manifest_donors: list[str] = []
    unmatched_metadata_donors: list[str] = []
    warnings: list[str] = []
    if metadata is not None:
        metadata_clean = standardize_metadata(metadata)
        if metadata_clean["donor_id"].duplicated().any():
            duplicates = sorted(metadata_clean.loc[metadata_clean["donor_id"].duplicated(), "donor_id"].astype(str).unique())
            raise ValueError(f"External metadata contains duplicate donor_id values: {duplicates}")
        manifest_donors = set(manifest_clean["donor_id"].astype(str))
        metadata_donors = set(metadata_clean["donor_id"].astype(str))
        unmatched_manifest_donors = sorted(manifest_donors - metadata_donors)
        unmatched_metadata_donors = sorted(metadata_donors - manifest_donors)
        if strict and (unmatched_manifest_donors or unmatched_metadata_donors):
            raise ValueError(
                "Metadata join strict mode found unmatched donors: "
                f"manifest_only={unmatched_manifest_donors}, metadata_only={unmatched_metadata_donors}"
            )
        metadata_for_join, rename_map = resolve_metadata_column_conflicts(metadata_clean, manifest_clean.columns)
        if rename_map:
            warnings.append(f"Renamed metadata columns that conflicted with manifest columns: {rename_map}")
        metadata_columns = [column for column in metadata_for_join.columns if column != "donor_id"]
        manifest_clean = manifest_clean.merge(metadata_for_join, on="donor_id", how="left")
        donor_base = donor_base.merge(metadata_for_join, on="donor_id", how="left")

    enriched = stabilize_enriched_manifest(manifest_clean, metadata_columns)
    donor_metadata = stabilize_donor_metadata(donor_base, metadata_columns)
    summary = metadata_join_summary(
        enriched,
        donor_metadata,
        healthy_ids,
        metadata is not None,
        metadata_columns,
        unmatched_manifest_donors,
        unmatched_metadata_donors,
        warnings,
    )
    return enriched, donor_metadata, summary


def standardize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    result = manifest.copy(deep=True)
    required = ["image_id", "donor_id"]
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    result["image_id"] = result["image_id"].map(standardize_id)
    result["donor_id"] = result["donor_id"].map(standardize_id)
    if "region_id" in result.columns:
        result["region_id"] = result["region_id"].map(standardize_id)
    if "image_path" not in result.columns:
        path_columns = [column for column in result.columns if column.lower() in {"path", "filepath", "file_path"}]
        if path_columns:
            result["image_path"] = result[path_columns[0]]
        else:
            result["image_path"] = np.nan
    return result


def standardize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    result = metadata.copy(deep=True)
    if "donor_id" not in result.columns:
        raise ValueError("External metadata table must contain a donor_id column.")
    result["donor_id"] = result["donor_id"].map(standardize_id)
    return result


def standardize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    return text


def resolve_metadata_column_conflicts(metadata: pd.DataFrame, manifest_columns: pd.Index) -> tuple[pd.DataFrame, dict[str, str]]:
    result = metadata.copy(deep=True)
    rename_map: dict[str, str] = {}
    manifest_names = set(manifest_columns)
    for column in list(result.columns):
        if column == "donor_id":
            continue
        if column in manifest_names or column == "is_healthy":
            new_name = f"{column}_metadata"
            rename_map[column] = new_name
    if rename_map:
        result = result.rename(columns=rename_map)
    return result, rename_map


def stabilize_enriched_manifest(enriched: pd.DataFrame, metadata_columns: list[str]) -> pd.DataFrame:
    result = enriched.copy(deep=True)
    for column in ENRICHED_MANIFEST_REQUIRED_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    original_columns = [column for column in result.columns if column not in ENRICHED_MANIFEST_REQUIRED_COLUMNS and column not in metadata_columns]
    ordered_metadata = [column for column in metadata_columns if column in result.columns]
    return result[ENRICHED_MANIFEST_REQUIRED_COLUMNS + original_columns + ordered_metadata]


def stabilize_donor_metadata(donor_metadata: pd.DataFrame, metadata_columns: list[str]) -> pd.DataFrame:
    result = donor_metadata.copy(deep=True)
    for column in DONOR_METADATA_REQUIRED_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    ordered_metadata = [column for column in metadata_columns if column in result.columns]
    extras = [column for column in result.columns if column not in DONOR_METADATA_REQUIRED_COLUMNS and column not in ordered_metadata]
    return result[DONOR_METADATA_REQUIRED_COLUMNS + extras + ordered_metadata]


def write_metadata_outputs(
    enriched_manifest: pd.DataFrame,
    donor_metadata: pd.DataFrame,
    summary: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, Path]:
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "enriched_manifest": out_dir / "enriched_manifest.csv",
        "donor_metadata": out_dir / "donor_metadata.csv",
        "summary_json": out_dir / "metadata_join_summary.json",
        "summary_txt": out_dir / "metadata_join_summary.txt",
    }
    enriched_manifest.to_csv(paths["enriched_manifest"], index=False)
    donor_metadata.to_csv(paths["donor_metadata"], index=False)
    with paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
    paths["summary_txt"].write_text(metadata_summary_text(summary), encoding="utf-8")
    return paths


def metadata_join_summary(
    enriched_manifest: pd.DataFrame,
    donor_metadata: pd.DataFrame,
    healthy_ids: set[str],
    metadata_provided: bool,
    metadata_columns: list[str],
    unmatched_manifest_donors: list[str],
    unmatched_metadata_donors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    healthy_donors_present = sorted(set(donor_metadata.loc[donor_metadata["is_healthy"].astype(bool), "donor_id"].astype(str)))
    return json_safe(
        {
            "enriched_manifest_rows": int(len(enriched_manifest)),
            "donor_metadata_rows": int(len(donor_metadata)),
            "donor_count": int(donor_metadata["donor_id"].nunique()) if "donor_id" in donor_metadata.columns else 0,
            "metadata_provided": bool(metadata_provided),
            "metadata_columns_joined": metadata_columns,
            "healthy_donor_ids_configured": sorted(healthy_ids),
            "healthy_donors_present": healthy_donors_present,
            "healthy_donor_count": int(len(healthy_donors_present)),
            "unmatched_manifest_donors": unmatched_manifest_donors,
            "unmatched_metadata_donors": unmatched_metadata_donors,
            "warnings": warnings,
            "caution": (
                "This module only prepares metadata joins. It performs no clinical/statistical interpretation, "
                "does not join metadata into feature tables, and treats donor_id as a string identifier."
            ),
        }
    )


def metadata_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Metadata Join Summary",
        f"enriched_manifest_rows: {summary.get('enriched_manifest_rows')}",
        f"donor_metadata_rows: {summary.get('donor_metadata_rows')}",
        f"donor_count: {summary.get('donor_count')}",
        f"metadata_provided: {summary.get('metadata_provided')}",
        f"healthy_donor_count: {summary.get('healthy_donor_count')}",
        f"healthy_donors_present: {summary.get('healthy_donors_present')}",
        f"unmatched_manifest_donors: {summary.get('unmatched_manifest_donors')}",
        f"unmatched_metadata_donors: {summary.get('unmatched_metadata_donors')}",
        f"warnings: {summary.get('warnings', [])}",
        str(summary.get("caution", "")),
    ]
    return "\n".join(lines) + "\n"


def default_metadata_output_dir(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "tables"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
