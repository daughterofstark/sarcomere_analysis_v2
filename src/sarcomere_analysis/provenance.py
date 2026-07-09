from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from .config import output_dir


def collect_run_provenance(
    cfg: dict[str, Any],
    image_path: str | Path,
    image_id: str,
    stage_versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(stage_versions or {})
    config_path = metadata.get("config_path")
    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_version": _package_version(),
        "git_commit": _git_commit(),
        "git_dirty_status": _git_dirty_status(),
        "config_path": str(config_path) if config_path is not None else None,
        "config_hash": config_hash(cfg),
        "effective_config": effective_config(cfg),
        "input_image_shape": _json_safe(metadata.get("input_image_shape")),
        "input_image_dtype": metadata.get("input_image_dtype"),
        "preprocessing_metadata": _json_safe(metadata.get("preprocessing_metadata", {})),
        "tissue_mask_metadata": _json_safe(metadata.get("tissue_mask_metadata", {})),
        "counts": _json_safe(metadata.get("counts", {})),
        "output_file_paths": _json_safe(metadata.get("output_file_paths", {})),
    }


def write_run_provenance(provenance: Mapping[str, Any], image_id: str, cfg: dict[str, Any]) -> Path:
    out_dir = output_dir(cfg) / "provenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{image_id}_run_provenance.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(provenance)), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def config_hash(cfg: dict[str, Any]) -> str:
    encoded = json.dumps(_json_safe(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def effective_config(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "paths",
        "outputs",
        "calibration",
        "filename_pattern",
        "run",
        "preprocessing",
        "masking",
        "patches",
        "qc",
        "orientation",
        "spacing",
    ]
    return {key: cfg.get(key) for key in keys if key in cfg}


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("sarcomere-analysis")
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return completed.stdout.strip() or None


def _git_dirty_status() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return completed.stdout


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
