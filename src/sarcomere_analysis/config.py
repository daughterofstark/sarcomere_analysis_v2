from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Calibration:
    pixel_size_um: float
    expected_spacing_um_min: float
    expected_spacing_um_max: float

    @property
    def pixels_per_um(self) -> float:
        return 1.0 / self.pixel_size_um

    @property
    def expected_spacing_px_min(self) -> float:
        return self.expected_spacing_um_min / self.pixel_size_um

    @property
    def expected_spacing_px_max(self) -> float:
        return self.expected_spacing_um_max / self.pixel_size_um


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config did not load as a mapping: {path}")
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    required_top_level = ["paths", "outputs", "calibration", "filename_pattern", "run"]
    missing = [key for key in required_top_level if key not in config]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    calibration = config["calibration"]
    pixel_size = float(calibration["pixel_size_um"])
    if pixel_size <= 0:
        raise ValueError("calibration.pixel_size_um must be positive")

    band = calibration["expected_sarcomere_spacing_um"]
    spacing_min = float(band["min"])
    spacing_max = float(band["max"])
    if spacing_min <= 0 or spacing_max <= 0 or spacing_min >= spacing_max:
        raise ValueError("Expected sarcomere spacing band must be positive and min < max")


def get_calibration(config: dict[str, Any]) -> Calibration:
    calibration = config["calibration"]
    band = calibration["expected_sarcomere_spacing_um"]
    return Calibration(
        pixel_size_um=float(calibration["pixel_size_um"]),
        expected_spacing_um_min=float(band["min"]),
        expected_spacing_um_max=float(band["max"]),
    )


def raw_tiff_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["raw_tiff_dir"])


def output_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["output_dir"])


def manifest_csv_path(config: dict[str, Any]) -> Path:
    return Path(config["outputs"]["manifest_csv"])
