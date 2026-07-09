from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import output_dir


ANNOTATION_UI_COLUMNS = [
    "annotation_id",
    "image_id",
    "donor_id",
    "patch_id",
    "manual_dominant_orientation_deg",
    "manual_organisation_score",
    "manual_organisation_label",
    "visible_striations_yes_no",
    "manual_sarcomere_length_um_optional",
    "confidence_score",
    "annotator_id",
    "notes",
]

ANNOTATION_INDEX_REQUIRED_COLUMNS = [
    "annotation_id",
    "image_id",
    "donor_id",
    "patch_id",
]

ORGANISATION_LABELS = {
    1: "disorganised / no coherent striation orientation",
    2: "weakly organised",
    3: "moderately organised",
    4: "strongly organised",
    5: "highly organised",
}

VISIBILITY_KEYS = {
    "y": "yes",
    "u": "yes_unclear",
    "n": "no",
}


@dataclass(frozen=True)
class AnnotationPaths:
    pack_dir: Path
    index_path: Path
    template_path: Path
    output_csv: Path
    autosave_csv: Path


def default_annotation_paths(cfg: dict[str, Any], output_csv: str | Path | None = None) -> AnnotationPaths:
    pack_dir = output_dir(cfg) / "annotation_pack"
    output_path = Path(output_csv) if output_csv else pack_dir / "annotation_filled.csv"
    return AnnotationPaths(
        pack_dir=pack_dir,
        index_path=pack_dir / "annotation_patch_index.csv",
        template_path=pack_dir / "annotation_template.csv",
        output_csv=output_path,
        autosave_csv=autosave_path_for(output_path),
    )


def autosave_path_for(output_csv: str | Path) -> Path:
    path = Path(output_csv)
    return path.with_name(f"{path.stem}.autosave{path.suffix}")


def read_annotation_index(index_path: str | Path) -> pd.DataFrame:
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation index not found: {path}")
    index = pd.read_csv(path, dtype={"annotation_id": str, "image_id": str, "donor_id": str, "patch_id": str})
    require_columns(index, ANNOTATION_INDEX_REQUIRED_COLUMNS, "annotation index")
    for column in ANNOTATION_INDEX_REQUIRED_COLUMNS:
        index[column] = index[column].astype(str)
    if "crop_path" not in index.columns:
        index["crop_path"] = [resolve_crop_path(row, path) for _, row in index.iterrows()]
    else:
        index["crop_path"] = [resolve_crop_path(row, path) for _, row in index.iterrows()]
    return index


def read_annotation_template(template_path: str | Path) -> pd.DataFrame:
    path = Path(template_path)
    if not path.exists():
        return pd.DataFrame(columns=ANNOTATION_UI_COLUMNS)
    template = pd.read_csv(path, dtype={"annotation_id": str, "image_id": str, "donor_id": str, "patch_id": str})
    return stabilize_annotation_table(template)


def resolve_crop_path(row: pd.Series, index_path: Path) -> str:
    raw_path = row.get("crop_path", "")
    if raw_path is not None and not pd.isna(raw_path) and str(raw_path).strip():
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = index_path.parent / path
        return str(path)
    annotation_id = str(row["annotation_id"])
    crops_dir = index_path.parent / "crops"
    matches = sorted(crops_dir.glob(f"{safe_glob_token(annotation_id)}*.png"))
    if matches:
        return str(matches[0])
    return str(crops_dir / f"{annotation_id}.png")


def initialize_annotation_table(index: pd.DataFrame, template: pd.DataFrame | None = None) -> pd.DataFrame:
    base = pd.DataFrame(columns=ANNOTATION_UI_COLUMNS)
    for column in ["annotation_id", "image_id", "donor_id", "patch_id"]:
        base[column] = index[column].astype(str).to_numpy()
    for column in ANNOTATION_UI_COLUMNS:
        if column not in base.columns:
            base[column] = ""

    if template is not None and not template.empty:
        template_stable = stabilize_annotation_table(template)
        update_columns = [column for column in ANNOTATION_UI_COLUMNS if column not in {"annotation_id", "image_id", "donor_id", "patch_id"}]
        template_by_id = template_stable.drop_duplicates("annotation_id", keep="last").set_index("annotation_id")
        for idx, annotation_id in base["annotation_id"].astype(str).items():
            if annotation_id in template_by_id.index:
                for column in update_columns:
                    value = template_by_id.at[annotation_id, column]
                    if not is_blank(value):
                        base.at[idx, column] = value
    return stabilize_annotation_table(base)


def load_or_initialize_annotations(
    index: pd.DataFrame,
    template_path: str | Path | None,
    output_csv: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    template = read_annotation_template(template_path) if template_path is not None else pd.DataFrame(columns=ANNOTATION_UI_COLUMNS)
    base = initialize_annotation_table(index, template)
    path = Path(output_csv)
    if path.exists() and not overwrite:
        existing = pd.read_csv(path, dtype={"annotation_id": str, "image_id": str, "donor_id": str, "patch_id": str})
        existing = stabilize_annotation_table(existing)
        return merge_existing_annotations(base, existing)
    return base


def merge_existing_annotations(base: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    result = base.copy(deep=True)
    existing_by_id = existing.drop_duplicates("annotation_id", keep="last").set_index("annotation_id")
    for idx, annotation_id in result["annotation_id"].astype(str).items():
        if annotation_id not in existing_by_id.index:
            continue
        for column in ANNOTATION_UI_COLUMNS:
            if column == "annotation_id":
                continue
            value = existing_by_id.at[annotation_id, column]
            if not is_blank(value):
                result.at[idx, column] = value
    extra = existing.loc[~existing["annotation_id"].astype(str).isin(result["annotation_id"].astype(str))].copy()
    if not extra.empty:
        result = pd.concat([result, extra[ANNOTATION_UI_COLUMNS]], ignore_index=True)
    return stabilize_annotation_table(result)


def stabilize_annotation_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy(deep=True)
    for column in ANNOTATION_UI_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    for column in ["annotation_id", "image_id", "donor_id", "patch_id"]:
        result[column] = result[column].fillna("").astype(str)
    return result[ANNOTATION_UI_COLUMNS]


def save_annotations(table: pd.DataFrame, output_csv: str | Path) -> Path:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    stable = stabilize_annotation_table(table)
    stable.to_csv(path, index=False)
    return path


def autosave_annotations(table: pd.DataFrame, output_csv: str | Path) -> Path:
    return save_annotations(table, autosave_path_for(output_csv))


def update_annotation_row(table: pd.DataFrame, annotation_id: str, updates: dict[str, object]) -> pd.DataFrame:
    result = table.copy(deep=True)
    mask = result["annotation_id"].astype(str) == str(annotation_id)
    if not mask.any():
        raise ValueError(f"Unknown annotation_id: {annotation_id}")
    row_index = result.index[mask][0]
    for column, value in updates.items():
        if column not in ANNOTATION_UI_COLUMNS:
            raise ValueError(f"Unknown annotation column: {column}")
        result.at[row_index, column] = value
    return stabilize_annotation_table(result)


def apply_key_to_annotation(row: pd.Series | dict[str, object], key: str) -> dict[str, object]:
    values = dict(row)
    normalized = key.lower()
    if normalized in {"1", "2", "3", "4", "5"}:
        score = int(normalized)
        values["manual_organisation_score"] = score
        values["manual_organisation_label"] = ORGANISATION_LABELS[score]
    elif normalized in VISIBILITY_KEYS:
        values["visible_striations_yes_no"] = VISIBILITY_KEYS[normalized]
    elif normalized == "r":
        values["manual_dominant_orientation_deg"] = np.nan
    return values


def validate_orientation_angle(value: str) -> float:
    if value.strip() == "":
        return float("nan")
    angle = float(value)
    if not np.isfinite(angle) or angle < 0 or angle > 180:
        raise ValueError("Manual orientation angle must be blank/NaN or between 0 and 180 degrees.")
    return angle


def headless_check(
    cfg: dict[str, Any],
    index_path: str | Path | None = None,
    template_path: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    paths = default_annotation_paths(cfg, output_csv)
    index_file = Path(index_path) if index_path else paths.index_path
    template_file = Path(template_path) if template_path else paths.template_path
    index = read_annotation_index(index_file)
    template = read_annotation_template(template_file)
    annotations = load_or_initialize_annotations(index, template_file, paths.output_csv, overwrite=False)
    missing_crops = [path for path in index["crop_path"].astype(str) if not Path(path).exists()]
    return {
        "index_path": str(index_file),
        "template_path": str(template_file),
        "output_csv": str(paths.output_csv),
        "autosave_csv": str(paths.autosave_csv),
        "annotation_rows": int(len(annotations)),
        "index_rows": int(len(index)),
        "template_rows": int(len(template)),
        "crop_count": int(len(index) - len(missing_crops)),
        "missing_crop_count": int(len(missing_crops)),
        "missing_crops": missing_crops[:10],
    }


def run_annotation_ui(
    cfg: dict[str, Any],
    start_annotation_id: str | None = None,
    output_csv: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    paths = default_annotation_paths(cfg, output_csv)
    index = read_annotation_index(paths.index_path)
    annotations = load_or_initialize_annotations(index, paths.template_path, paths.output_csv, overwrite=overwrite)
    if overwrite or not paths.output_csv.exists():
        save_annotations(annotations, paths.output_csv)

    state = {"position": starting_position(annotations, start_annotation_id), "annotations": annotations, "closed": False}

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))

    def draw() -> None:
        ax.clear()
        current = current_index_row(index, state["annotations"], state["position"])
        crop = load_display_crop(current["crop_path"])
        ax.imshow(crop, cmap="gray")
        ax.axis("off")
        ann = state["annotations"].iloc[state["position"]]
        title = annotation_title(current, ann, state["position"], len(state["annotations"]))
        ax.set_title(title, fontsize=9)
        fig.canvas.draw_idle()

    def persist() -> None:
        save_annotations(state["annotations"], paths.output_csv)
        autosave_annotations(state["annotations"], paths.output_csv)

    def prompt_and_update_current() -> None:
        ann = state["annotations"].iloc[state["position"]]
        annotation_id = str(ann["annotation_id"])
        print(f"\nAnnotation {annotation_id}")
        angle_text = input("Manual dominant orientation deg [0-180, blank = NaN, keep = unchanged]: ").strip()
        updates: dict[str, object] = {}
        if angle_text.lower() != "keep":
            updates["manual_dominant_orientation_deg"] = validate_orientation_angle(angle_text)
        label_text = input("Manual organisation label [blank = keep]: ").strip()
        if label_text:
            updates["manual_organisation_label"] = label_text
        confidence_text = input("Confidence score [1-5, blank = keep]: ").strip()
        if confidence_text:
            confidence = int(confidence_text)
            if confidence < 1 or confidence > 5:
                raise ValueError("Confidence score must be 1-5.")
            updates["confidence_score"] = confidence
        notes_text = input("Notes [blank = keep]: ").strip()
        if notes_text:
            updates["notes"] = notes_text
        if updates:
            state["annotations"] = update_annotation_row(state["annotations"], annotation_id, updates)
        persist()

    def advance(delta: int) -> None:
        state["position"] = min(max(state["position"] + delta, 0), len(state["annotations"]) - 1)
        draw()

    def on_key(event: Any) -> None:
        key = str(event.key or "").lower()
        if key in {"1", "2", "3", "4", "5", "y", "u", "n", "r"}:
            ann = state["annotations"].iloc[state["position"]]
            updated = apply_key_to_annotation(ann, key)
            state["annotations"] = update_annotation_row(state["annotations"], str(ann["annotation_id"]), updated)
            persist()
            draw()
        elif key in {"enter", "return"}:
            prompt_and_update_current()
            advance(1)
        elif key == "b":
            advance(-1)
        elif key == "s":
            persist()
            print(f"Saved annotations: {paths.output_csv}")
        elif key == "q":
            persist()
            print(f"Saved annotations and quitting: {paths.output_csv}")
            state["closed"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    print_keyboard_help(paths.output_csv, paths.autosave_csv)
    draw()
    plt.show()
    if not state["closed"]:
        persist()
    return paths.output_csv


def starting_position(annotations: pd.DataFrame, start_annotation_id: str | None = None) -> int:
    if start_annotation_id:
        matches = annotations.index[annotations["annotation_id"].astype(str) == str(start_annotation_id)].tolist()
        if not matches:
            raise ValueError(f"Unknown start annotation_id: {start_annotation_id}")
        return int(matches[0])
    incomplete = annotations.index[annotations["manual_organisation_score"].map(is_blank)].tolist()
    return int(incomplete[0]) if incomplete else 0


def current_index_row(index: pd.DataFrame, annotations: pd.DataFrame, position: int) -> pd.Series:
    annotation_id = str(annotations.iloc[position]["annotation_id"])
    match = index.loc[index["annotation_id"].astype(str) == annotation_id]
    if match.empty:
        raise ValueError(f"Annotation index missing annotation_id: {annotation_id}")
    return match.iloc[0]


def load_display_crop(crop_path: str | Path) -> np.ndarray:
    path = Path(crop_path)
    if not path.exists():
        raise FileNotFoundError(f"Crop PNG not found: {path}")
    image = Image.open(path)
    return np.asarray(image)


def annotation_title(index_row: pd.Series, annotation_row: pd.Series, position: int, total: int) -> str:
    return (
        f"{position + 1}/{total}  {annotation_row['annotation_id']}  "
        f"image={annotation_row['image_id']} donor={annotation_row['donor_id']} patch={annotation_row['patch_id']}\n"
        f"auto OOP={index_row.get('patch_oop', '')}  auto orientation deg={index_row.get('patch_mean_orientation_deg', '')}  "
        f"score={annotation_row.get('manual_organisation_score', '')}  visible={annotation_row.get('visible_striations_yes_no', '')}"
    )


def print_keyboard_help(output_csv: Path, autosave_csv: Path) -> None:
    print("Local OOP/orientation annotation UI")
    print(f"Output: {output_csv}")
    print(f"Autosave: {autosave_csv}")
    print("Keys: 1-5 score, y yes, u unclear, n no, r orientation NaN, enter prompt+next, b back, s save, q save+quit")


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required {label} columns: {missing}")


def is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip() == ""


def safe_glob_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "?" for char in value)
