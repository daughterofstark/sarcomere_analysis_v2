from __future__ import annotations

from html import escape
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .config import manifest_csv_path, output_dir
from .io import build_manifest
from .outputs import ensure_output_dirs


GALLERY_COLUMNS = [
    "image_id",
    "donor_id",
    "status",
    "tissue_fraction",
    "image_oop",
    "image_oop_heterogeneity",
    "n_spacing_valid_patches",
    "spacing_valid_fraction",
    "image_spacing_median_um",
    "tissue_mask_overlay_path",
    "orientation_preview_path",
    "coherence_preview_path",
    "oop_heatmap_path",
    "spacing_heatmap_path",
    "provenance_path",
    "qc_flag_summary",
]

PREVIEW_SUFFIXES = {
    "tissue_mask_overlay_path": "tissue_mask_overlay.png",
    "orientation_preview_path": "orientation.png",
    "coherence_preview_path": "coherence.png",
    "oop_heatmap_path": "oop_heatmap.png",
    "spacing_heatmap_path": "spacing_heatmap.png",
}


def build_qc_gallery_index(
    cfg: dict[str, Any],
    limit: int | None = None,
    sort_by: str = "spacing_valid_fraction",
    ascending: bool = False,
    require_existing_previews: bool = False,
) -> pd.DataFrame:
    manifest = read_manifest(cfg)
    per_image = read_optional_csv(output_dir(cfg) / "tables" / "per_image_metrics.csv")
    batch_summary = read_optional_csv(output_dir(cfg) / "tables" / "batch_run_summary.csv")

    table = manifest[["image_id", "donor_id"]].copy()
    table = table.merge(per_image, on=["image_id", "donor_id"], how="left", suffixes=("", "_metric"))
    status_columns = [column for column in ["image_id", "donor_id", "status"] if column in batch_summary.columns]
    if status_columns:
        table = table.merge(batch_summary[status_columns], on=["image_id", "donor_id"], how="left")
    else:
        table["status"] = "missing_batch_summary"

    previews = output_dir(cfg) / "previews"
    provenance = output_dir(cfg) / "provenance"
    missing_counts = []
    for key, suffix in PREVIEW_SUFFIXES.items():
        paths = []
        missing = []
        for image_id in table["image_id"].astype(str):
            path = previews / f"{image_id}_{suffix}"
            paths.append(str(path) if path.exists() else "")
            missing.append(not path.exists())
        table[key] = paths
        missing_counts.append(pd.Series(missing, index=table.index, name=key))

    provenance_paths = []
    for image_id in table["image_id"].astype(str):
        path = provenance / f"{image_id}_run_provenance.json"
        provenance_paths.append(str(path) if path.exists() else "")
    table["provenance_path"] = provenance_paths

    missing_preview_count = sum(series.astype(int) for series in missing_counts)
    table["qc_flag_summary"] = [
        "ok" if count == 0 else f"missing_previews:{int(count)}"
        for count in missing_preview_count
    ]
    if require_existing_previews and int(missing_preview_count.sum()) > 0:
        missing_images = table.loc[missing_preview_count > 0, "image_id"].astype(str).head(10).tolist()
        raise FileNotFoundError(f"Missing preview files for {int((missing_preview_count > 0).sum())} images; examples: {missing_images}")

    for column in GALLERY_COLUMNS:
        if column not in table.columns:
            table[column] = pd.NA
    table = table[GALLERY_COLUMNS]

    if sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=ascending, na_position="last")
    if limit is not None:
        table = table.head(limit)
    return table.reset_index(drop=True)


def write_gallery_index(index: pd.DataFrame, cfg: dict[str, Any]) -> Path:
    path = ensure_output_dirs(cfg)["tables"] / "qc_gallery_index.csv"
    index.to_csv(path, index=False)
    return path


def write_gallery_html(index: pd.DataFrame, cfg: dict[str, Any]) -> Path:
    gallery_dir = output_dir(cfg) / "qc_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    path = gallery_dir / "index.html"
    path.write_text(render_gallery_html(index, path.parent), encoding="utf-8")
    return path


def render_gallery_html(index: pd.DataFrame, html_dir: Path) -> str:
    cards = "\n".join(render_card(row, html_dir) for _, row in index.iterrows())
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sarcomere QC Gallery</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f4; color: #222; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    .metrics {{ font-size: 13px; line-height: 1.45; margin-bottom: 10px; }}
    .thumbs {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; height: 180px; object-fit: contain; background: #eee; border: 1px solid #ddd; }}
    figcaption {{ font-size: 12px; color: #555; }}
    .missing {{ height: 180px; display: grid; place-items: center; background: #eee; border: 1px dashed #aaa; color: #777; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Sarcomere QC Gallery</h1>
  <p>Diagnostic gallery for existing pipeline preview PNGs. Not a publication figure.</p>
  <div class="grid">
{cards}
  </div>
</body>
</html>
"""


def render_card(row: pd.Series, html_dir: Path) -> str:
    image_id = escape(str(row["image_id"]))
    donor_id = escape(str(row["donor_id"]))
    metric_lines = [
        f"status: {escape(str(row['status']))}",
        f"tissue_fraction: {format_value(row['tissue_fraction'])}",
        f"image_oop: {format_value(row['image_oop'])}",
        f"oop_heterogeneity: {format_value(row['image_oop_heterogeneity'])}",
        f"n_spacing_valid_patches: {format_value(row['n_spacing_valid_patches'])}",
        f"spacing_valid_fraction: {format_value(row['spacing_valid_fraction'])}",
        f"spacing_median_um: {format_value(row['image_spacing_median_um'])}",
        f"qc: {escape(str(row['qc_flag_summary']))}",
    ]
    figures = "\n".join(
        render_figure(row[column], label, html_dir)
        for column, label in [
            ("tissue_mask_overlay_path", "Tissue mask"),
            ("orientation_preview_path", "Orientation"),
            ("coherence_preview_path", "Coherence"),
            ("oop_heatmap_path", "OOP"),
            ("spacing_heatmap_path", "Spacing"),
        ]
    )
    metrics = "<br>".join(metric_lines)
    return f"""    <section class="card">
      <h2>{image_id}</h2>
      <div class="metrics">donor_id: {donor_id}<br>{metrics}</div>
      <div class="thumbs">
{figures}
      </div>
    </section>"""


def render_figure(path_value: object, label: str, html_dir: Path) -> str:
    path = str(path_value) if pd.notna(path_value) else ""
    if not path:
        return f'        <figure><div class="missing">Missing {escape(label)}</div><figcaption>{escape(label)}</figcaption></figure>'
    src = escape(relative_path_for_html(Path(path), html_dir))
    return f'        <figure><a href="{src}"><img src="{src}" alt="{escape(label)}"></a><figcaption>{escape(label)}</figcaption></figure>'


def relative_path_for_html(path: Path, html_dir: Path) -> str:
    return os.path.relpath(path, html_dir)


def format_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4g}"
    return escape(str(value))


def read_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    path = manifest_csv_path(cfg)
    if path.exists():
        return pd.read_csv(path, dtype={"image_id": str, "donor_id": str, "region_id": str})
    return build_manifest(cfg)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"image_id": str, "donor_id": str})
