from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PatchWindow:
    image_id: str
    patch_id: str
    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0


def patch_params(config: dict[str, Any]) -> dict[str, int]:
    defaults = {"patch_size_px": 256, "stride_px": 128, "margin_px": 0}
    params = dict(defaults)
    params.update(config.get("patches", {}))
    parsed = {key: int(value) for key, value in params.items()}
    if parsed["patch_size_px"] <= 0:
        raise ValueError("patches.patch_size_px must be positive")
    if parsed["stride_px"] <= 0:
        raise ValueError("patches.stride_px must be positive")
    if parsed["margin_px"] < 0:
        raise ValueError("patches.margin_px must be non-negative")
    return parsed


def generate_patch_grid(image_shape: tuple[int, int], image_id: str, config: dict[str, Any]) -> list[PatchWindow]:
    height, width = image_shape
    params = patch_params(config)
    size = params["patch_size_px"]
    stride = params["stride_px"]
    margin = params["margin_px"]

    y_start = margin
    x_start = margin
    y_stop = height - margin
    x_stop = width - margin
    if y_stop - y_start < size or x_stop - x_start < size:
        return []

    windows: list[PatchWindow] = []
    patch_index = 0
    for y0 in range(y_start, y_stop - size + 1, stride):
        for x0 in range(x_start, x_stop - size + 1, stride):
            y1 = y0 + size
            x1 = x0 + size
            windows.append(
                PatchWindow(
                    image_id=image_id,
                    patch_id=f"{image_id}_p{patch_index:05d}",
                    y0=y0,
                    x0=x0,
                    y1=y1,
                    x1=x1,
                )
            )
            patch_index += 1
    return windows


def patch_grid_table(image_shape: tuple[int, int], image_id: str, config: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "image_id": patch.image_id,
            "patch_id": patch.patch_id,
            "y0": patch.y0,
            "x0": patch.x0,
            "y1": patch.y1,
            "x1": patch.x1,
            "center_y": patch.center_y,
            "center_x": patch.center_x,
        }
        for patch in generate_patch_grid(image_shape, image_id, config)
    ]
    return pd.DataFrame(rows)
