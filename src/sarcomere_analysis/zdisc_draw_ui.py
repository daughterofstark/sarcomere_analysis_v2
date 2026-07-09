from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import output_dir
from .zdisc_annotation import ZDISC_LABELS, safe_name


DRAW_INDEX_REQUIRED_COLUMNS = [
    "annotation_id",
    "image_id",
    "donor_id",
    "patch_id",
    "annotation_image_path",
    "mask_path",
]


@dataclass
class DrawState:
    position: int
    current_label: int
    brush_radius: int
    is_drawing: bool = False
    changed: bool = False


def default_zdisc_draw_index_path(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "zdisc_annotation" / "zdisc_annotation_index.csv"


def default_zdisc_draw_progress_path(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "zdisc_annotation" / "zdisc_draw_progress.json"


def load_draw_index(index_path: str | Path) -> pd.DataFrame:
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Z-disc draw index not found: {path}")
    index = pd.read_csv(
        path,
        dtype={"annotation_id": str, "image_id": str, "donor_id": str, "patch_id": str},
    )
    require_columns(index, DRAW_INDEX_REQUIRED_COLUMNS, "Z-disc draw index")
    for column in ["annotation_id", "image_id", "donor_id", "patch_id"]:
        index[column] = index[column].astype(str)
    for column in ["annotation_image_path", "mask_path", "overlay_path"]:
        if column in index.columns:
            index[column] = index[column].astype(str)
    if "overlay_path" not in index.columns:
        index["overlay_path"] = [
            str(Path(row["mask_path"]).parents[1] / "overlays" / f"{safe_name(str(row['annotation_id']))}_overlay.png")
            for _, row in index.iterrows()
        ]
    return index


def load_crop_image(path: str | Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if image.size == 0:
        raise ValueError(f"Empty crop image: {path}")
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def load_mask(path: str | Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    mask_path = Path(path)
    if not mask_path.exists():
        if expected_shape is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        return np.zeros(expected_shape, dtype=np.uint8)
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    normalized = normalize_mask_labels(mask)
    if expected_shape is not None and tuple(normalized.shape) != tuple(expected_shape):
        raise ValueError(f"Mask shape {normalized.shape} does not match image shape {expected_shape}: {mask_path}")
    return normalized


def normalize_mask_labels(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {values.shape}")
    normalized = values.astype(np.uint16, copy=True)
    normalized[normalized == 255] = 1
    unique = set(int(value) for value in np.unique(normalized))
    invalid = sorted(unique.difference(ZDISC_LABELS.keys()))
    if invalid:
        raise ValueError(f"Mask contains unsupported labels {invalid}; allowed labels are 0, 1, 2, with 255 interpreted as 1.")
    return normalized.astype(np.uint8)


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    normalized = normalize_mask_labels(mask)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalized, mode="L").save(out)
    return out


def brush_pixel_indices(shape: tuple[int, int], x: float, y: float, radius: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = int(shape[0]), int(shape[1])
    r = max(int(radius), 1)
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    x0 = max(cx - r, 0)
    x1 = min(cx + r + 1, width)
    y0 = max(cy - r, 0)
    y1 = min(cy + r + 1, height)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    keep = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    return yy[keep], xx[keep]


def paint_mask(mask: np.ndarray, x: float, y: float, label: int, radius: int) -> np.ndarray:
    if label not in ZDISC_LABELS:
        raise ValueError(f"Invalid drawing label {label}; expected one of {sorted(ZDISC_LABELS)}.")
    result = normalize_mask_labels(mask).copy()
    ys, xs = brush_pixel_indices(result.shape, x, y, radius)
    result[ys, xs] = int(label)
    return result


def build_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    display = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    rgb = np.dstack([display, display, display])
    label_mask = normalize_mask_labels(mask)
    colors = {
        1: np.array([1.0, 0.05, 0.05], dtype=np.float32),
        2: np.array([0.05, 0.25, 1.0], dtype=np.float32),
    }
    blend_alpha = float(np.clip(alpha, 0.0, 1.0))
    for label, color in colors.items():
        pixels = label_mask == label
        if np.any(pixels):
            rgb[pixels] = (1.0 - blend_alpha) * rgb[pixels] + blend_alpha * color
    return np.clip(rgb, 0.0, 1.0)


def write_overlay_png(image: np.ndarray, mask: np.ndarray, path: str | Path, alpha: float = 0.45) -> Path:
    overlay = build_overlay(image, mask, alpha=alpha)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((overlay * 255).astype(np.uint8), mode="RGB").save(out)
    return out


def write_progress(
    progress_path: str | Path,
    index: pd.DataFrame,
    position: int,
    current_label: int,
    brush_radius: int,
) -> Path:
    row = index.iloc[int(position)]
    progress = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "position": int(position),
        "total": int(len(index)),
        "annotation_id": str(row["annotation_id"]),
        "image_id": str(row["image_id"]),
        "donor_id": str(row["donor_id"]),
        "patch_id": str(row["patch_id"]),
        "current_label": int(current_label),
        "brush_radius": int(brush_radius),
    }
    path = Path(progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    return path


def load_progress(progress_path: str | Path) -> dict[str, Any] | None:
    path = Path(progress_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def headless_check(
    cfg: dict[str, Any],
    index_path: str | Path | None = None,
) -> dict[str, Any]:
    index_file = Path(index_path) if index_path else default_zdisc_draw_index_path(cfg)
    index = load_draw_index(index_file)
    missing_images: list[str] = []
    missing_masks: list[str] = []
    shape_mismatches: list[str] = []
    invalid_masks: list[str] = []
    for _, row in index.iterrows():
        image_path = Path(str(row["annotation_image_path"]))
        mask_path = Path(str(row["mask_path"]))
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue
        image = load_crop_image(image_path)
        if not mask_path.exists():
            missing_masks.append(str(mask_path))
            continue
        try:
            _ = load_mask(mask_path, expected_shape=image.shape)
        except ValueError as exc:
            message = str(exc)
            if "does not match" in message:
                shape_mismatches.append(f"{mask_path}: {message}")
            else:
                invalid_masks.append(f"{mask_path}: {message}")
    return {
        "index_path": str(index_file),
        "rows": int(len(index)),
        "missing_image_count": int(len(missing_images)),
        "missing_mask_count": int(len(missing_masks)),
        "shape_mismatch_count": int(len(shape_mismatches)),
        "invalid_mask_count": int(len(invalid_masks)),
        "missing_images": missing_images[:10],
        "missing_masks": missing_masks[:10],
        "shape_mismatches": shape_mismatches[:10],
        "invalid_masks": invalid_masks[:10],
    }


def starting_position(index: pd.DataFrame, start_annotation_id: str | None, progress: dict[str, Any] | None = None) -> int:
    if start_annotation_id:
        matches = index.index[index["annotation_id"].astype(str) == str(start_annotation_id)].tolist()
        if not matches:
            raise ValueError(f"Unknown annotation_id for --start: {start_annotation_id}")
        return int(matches[0])
    if progress and "annotation_id" in progress:
        matches = index.index[index["annotation_id"].astype(str) == str(progress["annotation_id"])].tolist()
        if matches:
            return int(matches[0])
    return 0


def run_draw_ui(
    cfg: dict[str, Any],
    index_path: str | Path | None = None,
    start_annotation_id: str | None = None,
    brush_size: int = 2,
    alpha: float = 0.45,
    overwrite_progress: bool = False,
) -> Path:
    import matplotlib.pyplot as plt

    index_file = Path(index_path) if index_path else default_zdisc_draw_index_path(cfg)
    index = load_draw_index(index_file)
    progress_path = default_zdisc_draw_progress_path(cfg)
    progress = None if overwrite_progress else load_progress(progress_path)
    state = DrawState(
        position=starting_position(index, start_annotation_id, progress),
        current_label=1,
        brush_radius=max(int(brush_size), 1),
    )
    current = {"image": None, "mask": None}

    fig, ax = plt.subplots(figsize=(7, 7))

    def load_current() -> None:
        row = index.iloc[state.position]
        image = load_crop_image(row["annotation_image_path"])
        mask = load_mask(row["mask_path"], expected_shape=image.shape)
        current["image"] = image
        current["mask"] = mask

    def save_current(write_overlay: bool = False) -> None:
        row = index.iloc[state.position]
        save_mask(current["mask"], row["mask_path"])
        if write_overlay:
            write_overlay_png(current["image"], current["mask"], row["overlay_path"], alpha=alpha)
        write_progress(progress_path, index, state.position, state.current_label, state.brush_radius)
        state.changed = False

    def draw() -> None:
        row = index.iloc[state.position]
        ax.clear()
        ax.imshow(build_overlay(current["image"], current["mask"], alpha=alpha))
        ax.axis("off")
        ax.set_title(
            f"{state.position + 1}/{len(index)}  {row['annotation_id']}  image={row['image_id']} patch={row['patch_id']}\n"
            f"label={state.current_label} ({ZDISC_LABELS[state.current_label]})  brush radius={state.brush_radius}",
            fontsize=9,
        )
        fig.canvas.draw_idle()

    def paint_event(event: Any) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        current["mask"] = paint_mask(current["mask"], event.xdata, event.ydata, state.current_label, state.brush_radius)
        state.changed = True
        draw()

    def move(delta: int) -> None:
        save_current()
        state.position = min(max(state.position + delta, 0), len(index) - 1)
        load_current()
        draw()

    def on_press(event: Any) -> None:
        if event.button == 1:
            state.is_drawing = True
            paint_event(event)

    def on_motion(event: Any) -> None:
        if state.is_drawing:
            paint_event(event)

    def on_release(event: Any) -> None:
        if event.button == 1 and state.is_drawing:
            state.is_drawing = False
            save_current()

    def on_key(event: Any) -> None:
        key = str(event.key or "").lower()
        if key == "1":
            state.current_label = 1
        elif key == "2":
            state.current_label = 2
        elif key in {"0", "e"}:
            state.current_label = 0
        elif key == "[":
            state.brush_radius = max(state.brush_radius - 1, 1)
        elif key == "]":
            state.brush_radius += 1
        elif key in {"n", "right"}:
            move(1)
            return
        elif key in {"b", "left"}:
            move(-1)
            return
        elif key == "s":
            save_current()
            print(f"Saved mask: {index.iloc[state.position]['mask_path']}")
        elif key == "c":
            current["mask"] = np.zeros_like(current["mask"], dtype=np.uint8)
            save_current()
        elif key == "o":
            save_current(write_overlay=True)
            print(f"Wrote overlay: {index.iloc[state.position]['overlay_path']}")
        elif key == "h":
            print_controls()
        elif key == "q":
            save_current()
            plt.close(fig)
            return
        draw()

    print_controls()
    load_current()
    write_progress(progress_path, index, state.position, state.current_label, state.brush_radius)
    draw()
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return progress_path


def print_controls() -> None:
    print("Z-disc drawing controls:")
    print("  Left-click drag: paint current label")
    print("  1: visible Z-disc/striation, 2: ignore/uncertain, 0/e: eraser")
    print("  [ / ]: decrease/increase brush size")
    print("  n/right: next, b/left: previous, s: save, c: clear mask, o: write overlay, q: save and quit, h: help")


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required {label} columns: {missing}")
