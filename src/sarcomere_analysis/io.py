from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from .config import get_calibration, raw_tiff_dir
from .schemas import MANIFEST_COLUMNS, stabilize_columns


def discover_tiffs(config: dict[str, Any]) -> list[Path]:
    directory = raw_tiff_dir(config)
    extensions = {ext.lower() for ext in config["run"].get("include_extensions", [".tif", ".tiff"])}
    recursive = bool(config["run"].get("recursive", False))
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files)


def parse_image_filename(path: str | Path, filename_regex: str) -> dict[str, str]:
    path = Path(path)
    match = re.match(filename_regex, path.stem)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")
    parsed = {key: str(value) for key, value in match.groupdict().items()}
    parsed["image_id"] = path.stem
    return parsed


def build_manifest(config: dict[str, Any]) -> pd.DataFrame:
    regex = str(config["filename_pattern"]["regex"])
    calibration = get_calibration(config)
    rows: list[dict[str, object]] = []
    for path in discover_tiffs(config):
        parsed = parse_image_filename(path, regex)
        rows.append(
            {
                "image_id": parsed["image_id"],
                "donor_id": parsed["donor_id"],
                "region_id": parsed["region_id"],
                "filename": path.name,
                "image_path": str(path),
                "pixel_size_um": calibration.pixel_size_um,
                "expected_spacing_px_min": calibration.expected_spacing_px_min,
                "expected_spacing_px_max": calibration.expected_spacing_px_max,
            }
        )
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest = manifest.sort_values(["donor_id", "region_id", "image_id"]).reset_index(drop=True)
    return stabilize_columns(manifest, MANIFEST_COLUMNS)


def load_tiff(path: str | Path) -> np.ndarray:
    image = np.squeeze(tifffile.imread(path))
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeeze, got shape {image.shape}: {path}")
    return image


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
