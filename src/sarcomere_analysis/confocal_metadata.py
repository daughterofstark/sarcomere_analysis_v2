from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
import tifffile

from .config import output_dir
from .zdisc_annotation import json_safe


CONFOCAL_METADATA_COLUMNS = [
    "confocal_image_id",
    "filename",
    "source_path",
    "image_shape_y",
    "image_shape_x",
    "pixel_size_x_um",
    "pixel_size_y_um",
    "pixel_size_unit",
    "pixel_size_source",
    "pixel_size_available",
    "isotropic_pixels",
    "spacing_um_enabled",
    "spacing_um_policy",
    "calibration_warning",
    "expected_positive_example",
    "noted_complex_example",
]

MANUAL_TEMPLATE_COLUMNS = [
    "confocal_image_id",
    "filename",
    "pixel_size_x_um",
    "pixel_size_y_um",
    "pixel_size_unit",
    "notes",
]


def default_confocal_metadata_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_metadata"
    return {
        "root": root,
        "calibration": root / "confocal_metadata_calibration.csv",
        "summary_json": root / "confocal_metadata_summary.json",
        "summary_txt": root / "confocal_metadata_summary.txt",
        "manual_template": root / "confocal_manual_pixel_size_template.csv",
    }


def audit_confocal_metadata(
    cfg: dict[str, Any],
    confocal_manifest: str | Path | None = None,
    output_directory: str | Path | None = None,
    write_manual_template: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    root = output_dir(cfg)
    manifest_path = Path(confocal_manifest) if confocal_manifest else root / "confocal_baseline" / "confocal_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"confocal_image_id": str, "filename": str, "source_path": str})
    rows = [audit_manifest_row(row) for _, row in manifest.iterrows()]
    calibration = pd.DataFrame(rows, columns=CONFOCAL_METADATA_COLUMNS)
    summary = build_confocal_metadata_summary(calibration, write_manual_template)
    paths = default_confocal_metadata_paths(cfg, output_directory)
    write_confocal_metadata_outputs(calibration, summary, paths, write_manual_template)
    return calibration, summary, paths


def audit_manifest_row(row: pd.Series) -> dict[str, Any]:
    path = Path(str(row["source_path"]))
    metadata = extract_pixel_size_metadata(path)
    warning = calibration_warning(metadata)
    return {
        "confocal_image_id": str(row["confocal_image_id"]),
        "filename": str(row["filename"]),
        "source_path": str(path),
        "image_shape_y": int(row.get("image_shape_y", 0)) if pd.notna(row.get("image_shape_y", np.nan)) else np.nan,
        "image_shape_x": int(row.get("image_shape_x", 0)) if pd.notna(row.get("image_shape_x", np.nan)) else np.nan,
        "pixel_size_x_um": metadata["pixel_size_x_um"],
        "pixel_size_y_um": metadata["pixel_size_y_um"],
        "pixel_size_unit": metadata["pixel_size_unit"],
        "pixel_size_source": metadata["pixel_size_source"],
        "pixel_size_available": metadata["pixel_size_available"],
        "isotropic_pixels": metadata["isotropic_pixels"],
        "spacing_um_enabled": bool(metadata["pixel_size_available"] and metadata["isotropic_pixels"]),
        "spacing_um_policy": spacing_um_policy(metadata),
        "calibration_warning": warning,
        "expected_positive_example": bool(row.get("expected_positive_example", False)),
        "noted_complex_example": bool(row.get("noted_complex_example", False)),
    }


def extract_pixel_size_metadata(path: str | Path) -> dict[str, Any]:
    image_path = Path(path)
    base = missing_pixel_size("metadata_missing_or_unparseable")
    if not image_path.exists():
        base["pixel_size_source"] = "source_image_missing"
        return base
    try:
        if image_path.suffix.lower() in {".tif", ".tiff"}:
            return extract_tiff_pixel_size(image_path)
        return extract_pil_pixel_size(image_path)
    except Exception as exc:  # pragma: no cover - protects against unusual vendor metadata.
        failed = missing_pixel_size("metadata_read_error")
        failed["calibration_exception"] = str(exc)
        return failed


def extract_tiff_pixel_size(path: Path) -> dict[str, Any]:
    with tifffile.TiffFile(path) as tif:
        ome = extract_ome_pixel_size(tif.ome_metadata)
        if ome["pixel_size_available"]:
            return ome
        imagej = extract_imagej_pixel_size(tif)
        if imagej["pixel_size_available"]:
            return imagej
        resolution = extract_resolution_pixel_size(tif)
        if resolution["pixel_size_available"]:
            return resolution
    return missing_pixel_size("metadata_missing_or_unparseable")


def extract_ome_pixel_size(ome_metadata: str | None) -> dict[str, Any]:
    if not ome_metadata:
        return missing_pixel_size("ome_metadata_missing")
    try:
        root = ET.fromstring(ome_metadata)
    except ET.ParseError:
        return missing_pixel_size("ome_metadata_unparseable")
    for element in root.iter():
        if element.tag.split("}")[-1] != "Pixels":
            continue
        x_raw = element.attrib.get("PhysicalSizeX")
        y_raw = element.attrib.get("PhysicalSizeY")
        if x_raw is None or y_raw is None:
            continue
        unit = element.attrib.get("PhysicalSizeXUnit") or element.attrib.get("PhysicalSizeYUnit") or "µm"
        scale = unit_to_microns(unit)
        if scale is None:
            return missing_pixel_size(f"unsupported_ome_unit_{unit}")
        return pixel_size_record(float(x_raw) * scale, float(y_raw) * scale, "µm", "ome_physical_size")
    return missing_pixel_size("ome_physical_size_missing")


def extract_imagej_pixel_size(tif: tifffile.TiffFile) -> dict[str, Any]:
    imagej = tif.imagej_metadata or {}
    unit = str(imagej.get("unit", "")).strip()
    if not unit:
        return missing_pixel_size("imagej_unit_missing")
    scale = unit_to_microns(unit)
    if scale is None:
        return missing_pixel_size(f"unsupported_imagej_unit_{unit}")
    page = tif.pages[0]
    x_resolution = rational_tag_value(page.tags.get("XResolution"))
    y_resolution = rational_tag_value(page.tags.get("YResolution"))
    if x_resolution is None or y_resolution is None or x_resolution <= 0 or y_resolution <= 0:
        return missing_pixel_size("imagej_resolution_missing")
    return pixel_size_record(
        scale / x_resolution,
        scale / y_resolution,
        "µm",
        "imagej_unit_resolution_tags",
    )


def extract_resolution_pixel_size(tif: tifffile.TiffFile) -> dict[str, Any]:
    page = tif.pages[0]
    x_resolution = rational_tag_value(page.tags.get("XResolution"))
    y_resolution = rational_tag_value(page.tags.get("YResolution"))
    if x_resolution is None or y_resolution is None or x_resolution <= 0 or y_resolution <= 0:
        return missing_pixel_size("resolution_tags_missing")
    unit = resolution_unit_value(page.tags.get("ResolutionUnit"))
    if unit in {"inch", "in"}:
        return pixel_size_record(25400.0 / x_resolution, 25400.0 / y_resolution, "µm", "tiff_resolution_inch")
    if unit in {"centimeter", "cm"}:
        return pixel_size_record(10000.0 / x_resolution, 10000.0 / y_resolution, "µm", "tiff_resolution_centimeter")
    return missing_pixel_size("resolution_unit_missing_or_unitless")


def extract_pil_pixel_size(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
    if not dpi or len(dpi) < 2:
        return missing_pixel_size("pil_dpi_missing")
    x_dpi, y_dpi = float(dpi[0]), float(dpi[1])
    if x_dpi <= 0 or y_dpi <= 0:
        return missing_pixel_size("pil_dpi_invalid")
    return pixel_size_record(25400.0 / x_dpi, 25400.0 / y_dpi, "µm", "pil_dpi")


def pixel_size_record(x_um: float, y_um: float, unit: str, source: str) -> dict[str, Any]:
    x = float(x_um)
    y = float(y_um)
    available = bool(np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0)
    isotropic = bool(available and np.isclose(x, y, rtol=1.0e-4, atol=1.0e-9))
    return {
        "pixel_size_x_um": x if available else np.nan,
        "pixel_size_y_um": y if available else np.nan,
        "pixel_size_unit": unit if available else "",
        "pixel_size_source": source if available else "metadata_missing_or_unparseable",
        "pixel_size_available": available,
        "isotropic_pixels": isotropic if available else False,
    }


def missing_pixel_size(source: str) -> dict[str, Any]:
    return {
        "pixel_size_x_um": np.nan,
        "pixel_size_y_um": np.nan,
        "pixel_size_unit": "",
        "pixel_size_source": source,
        "pixel_size_available": False,
        "isotropic_pixels": False,
    }


def calibration_warning(metadata: dict[str, Any]) -> str:
    if not bool(metadata.get("pixel_size_available")):
        return "missing_pixel_size;do_not_use_widefield_fallback"
    if not bool(metadata.get("isotropic_pixels")):
        return "anisotropic_pixel_size_review_required"
    return "ok"


def spacing_um_policy(metadata: dict[str, Any]) -> str:
    if not bool(metadata.get("pixel_size_available")):
        return "disabled_missing_per_image_pixel_size"
    if not bool(metadata.get("isotropic_pixels")):
        return "disabled_anisotropic_pixel_size_requires_review"
    return "enabled_per_image_confocal_calibration"


def build_confocal_metadata_summary(calibration: pd.DataFrame, manual_template_requested: bool = False) -> dict[str, Any]:
    available = calibration["pixel_size_available"].fillna(False).astype(bool) if not calibration.empty else pd.Series(dtype=bool)
    unique_sizes = unique_pixel_sizes(calibration)
    return json_safe(
        {
            "mode": "confocal_metadata_calibration_audit",
            "image_count": int(len(calibration)),
            "pixel_size_available_count": int(available.sum()),
            "pixel_size_missing_count": int((~available).sum()),
            "unique_pixel_sizes_um": unique_sizes,
            "pixel_sizes_differ_across_images": len(unique_sizes) > 1,
            "anisotropic_pixel_count": int((available & ~calibration["isotropic_pixels"].fillna(False).astype(bool)).sum())
            if not calibration.empty
            else 0,
            "spacing_um_enabled_count": int(calibration["spacing_um_enabled"].fillna(False).astype(bool).sum())
            if not calibration.empty and "spacing_um_enabled" in calibration
            else 0,
            "missing_calibration_images": calibration.loc[~available, ["confocal_image_id", "filename"]].to_dict("records")
            if not calibration.empty
            else [],
            "warnings": calibration.loc[calibration["calibration_warning"].astype(str) != "ok", ["confocal_image_id", "calibration_warning"]].to_dict("records")
            if not calibration.empty
            else [],
            "manual_template_requested": bool(manual_template_requested),
            "spacing_policy": "Spacing in microns must use per-image confocal calibration only; widefield calibration is never used as fallback.",
            "widefield_calibration_used": False,
            "interpretation": [
                "Confocal pixel size is image-specific when metadata provides it.",
                "Images without valid pixel size remain uncalibrated for micron-scale spacing.",
                "Per-image calibration is required before any confocal spacing in microns is reported.",
                "This audit does not change widefield outputs or production algorithms.",
            ],
        }
    )


def unique_pixel_sizes(calibration: pd.DataFrame) -> list[dict[str, float]]:
    if calibration.empty:
        return []
    available = calibration.loc[calibration["pixel_size_available"].fillna(False).astype(bool)].copy()
    if available.empty:
        return []
    rounded = available.assign(
        pixel_size_x_um=available["pixel_size_x_um"].astype(float).round(9),
        pixel_size_y_um=available["pixel_size_y_um"].astype(float).round(9),
    )
    grouped = rounded.groupby(["pixel_size_x_um", "pixel_size_y_um"], dropna=False).size().reset_index(name="image_count")
    return [
        {
            "pixel_size_x_um": float(row["pixel_size_x_um"]),
            "pixel_size_y_um": float(row["pixel_size_y_um"]),
            "image_count": int(row["image_count"]),
        }
        for _, row in grouped.iterrows()
    ]


def write_confocal_metadata_outputs(
    calibration: pd.DataFrame,
    summary: dict[str, Any],
    paths: dict[str, Path],
    write_manual_template: bool = False,
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    calibration.to_csv(paths["calibration"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths["summary_txt"].write_text(render_confocal_metadata_summary_text(summary), encoding="utf-8")
    if write_manual_template:
        manual_template_from_calibration(calibration).to_csv(paths["manual_template"], index=False)


def manual_template_from_calibration(calibration: pd.DataFrame) -> pd.DataFrame:
    if calibration.empty:
        return pd.DataFrame(columns=MANUAL_TEMPLATE_COLUMNS)
    missing = calibration.loc[~calibration["pixel_size_available"].fillna(False).astype(bool)]
    rows = [
        {
            "confocal_image_id": str(row["confocal_image_id"]),
            "filename": str(row["filename"]),
            "pixel_size_x_um": "",
            "pixel_size_y_um": "",
            "pixel_size_unit": "µm",
            "notes": "Fill from FIJI Image > Properties; do not use widefield calibration.",
        }
        for _, row in missing.iterrows()
    ]
    return pd.DataFrame(rows, columns=MANUAL_TEMPLATE_COLUMNS)


def render_confocal_metadata_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "Confocal metadata calibration audit",
        f"image_count: {summary['image_count']}",
        f"pixel_size_available_count: {summary['pixel_size_available_count']}",
        f"pixel_size_missing_count: {summary['pixel_size_missing_count']}",
        f"unique_pixel_sizes_um: {summary['unique_pixel_sizes_um']}",
        f"pixel_sizes_differ_across_images: {summary['pixel_sizes_differ_across_images']}",
        f"anisotropic_pixel_count: {summary['anisotropic_pixel_count']}",
        f"spacing_um_enabled_count: {summary['spacing_um_enabled_count']}",
        f"widefield_calibration_used: {summary['widefield_calibration_used']}",
        f"spacing_policy: {summary['spacing_policy']}",
        "",
        "Warnings:",
    ]
    lines.extend(f"- {warning}" for warning in summary["warnings"])
    lines.append("")
    lines.extend(summary["interpretation"])
    return "\n".join(lines) + "\n"


def rational_tag_value(tag: Any) -> float | None:
    if tag is None:
        return None
    value = tag.value
    if isinstance(value, tuple) and len(value) == 2:
        denominator = float(value[1])
        if denominator == 0:
            return None
        return float(value[0]) / denominator
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolution_unit_value(tag: Any) -> str:
    if tag is None:
        return ""
    value = tag.value
    if hasattr(value, "name"):
        return str(value.name).lower()
    if isinstance(value, bytes):
        return value.decode(errors="ignore").lower()
    text = str(value).lower()
    mapping = {"1": "none", "2": "inch", "3": "centimeter"}
    return mapping.get(text, text)


def unit_to_microns(unit: str) -> float | None:
    normalized = unit.strip().lower().replace("µ", "u")
    mapping = {
        "um": 1.0,
        "micron": 1.0,
        "microns": 1.0,
        "micrometer": 1.0,
        "micrometers": 1.0,
        "micrometre": 1.0,
        "micrometres": 1.0,
        "nm": 0.001,
        "nanometer": 0.001,
        "nanometers": 0.001,
        "nanometre": 0.001,
        "nanometres": 0.001,
        "mm": 1000.0,
        "millimeter": 1000.0,
        "millimeters": 1000.0,
        "millimetre": 1000.0,
        "millimetres": 1000.0,
    }
    return mapping.get(normalized)
