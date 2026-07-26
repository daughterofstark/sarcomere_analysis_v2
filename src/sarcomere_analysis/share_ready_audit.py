from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .zdisc_annotation import json_safe


LARGE_FILE_THRESHOLD_BYTES = 25 * 1024 * 1024
RAW_IMAGE_SUFFIXES = {".tif", ".tiff", ".czi", ".lif", ".nd2", ".lsm"}
ZIP_SUFFIXES = {".zip"}
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
LOCAL_PATH_MARKERS = ["/Users/" + "medhasharma", "/mnt/data", "/private/tmp"]
PRIVATE_MARKERS = ["One" + "Drive", "donor-level raw path" + " leakage"]
REQUIRED_GITIGNORE_PATTERNS = [
    "*.tif",
    "*.tiff",
    "*.czi",
    "*.lif",
    "*.nd2",
    "*.lsm",
    "results/",
    ".venv/",
    "venv/",
    "env/",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    ".DS_Store",
    "*.npz",
    "*.zip",
]


def run_share_ready_audit(
    repo_root: str | Path,
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
    large_threshold_bytes: int = LARGE_FILE_THRESHOLD_BYTES,
) -> tuple[dict[str, Any], dict[str, Path]]:
    root = Path(repo_root).resolve()
    paths = default_share_ready_paths(root, output_directory, docs_directory)
    audit = build_share_ready_audit(root, large_threshold_bytes=large_threshold_bytes)
    write_share_ready_outputs(audit, paths)
    return audit, paths


def default_share_ready_paths(
    repo_root: Path,
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> dict[str, Path]:
    out_dir = Path(output_directory) if output_directory else repo_root / "results" / "share_ready_audit"
    docs_dir = Path(docs_directory) if docs_directory else repo_root / "docs"
    return {
        "json": out_dir / "share_ready_audit_summary.json",
        "txt": out_dir / "share_ready_audit_summary.txt",
        "markdown": docs_dir / "SHARE_READY_AUDIT.md",
    }


def build_share_ready_audit(
    repo_root: str | Path,
    large_threshold_bytes: int = LARGE_FILE_THRESHOLD_BYTES,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    files = list(iter_repo_files(root))
    tracked = tracked_files(root)
    gitignore_text = read_text(root / ".gitignore")

    large_files = summarize_files([path for path in files if safe_size(path) > large_threshold_bytes], root)
    raw_present = summarize_files([path for path in files if path.suffix.lower() in RAW_IMAGE_SUFFIXES], root)
    raw_tracked = summarize_files([root / rel for rel in tracked if (root / rel).suffix.lower() in RAW_IMAGE_SUFFIXES], root)
    zip_files = summarize_files([path for path in files if path.suffix.lower() in ZIP_SUFFIXES], root)
    marker_search_files = [path for path in files if not is_share_audit_output(path, root)]
    absolute_paths = search_text_markers(root, marker_search_files, LOCAL_PATH_MARKERS)
    private_markers = search_private_markers(root, marker_search_files)
    gitignore_check = check_gitignore_patterns(gitignore_text)

    safe_to_push = (
        not large_files
        and not raw_tracked
        and not absolute_paths
        and not private_markers
        and not gitignore_check["missing_patterns"]
    )
    recommendations = build_recommendations(
        large_files=large_files,
        raw_present=raw_present,
        raw_tracked=raw_tracked,
        zip_files=zip_files,
        absolute_paths=absolute_paths,
        private_markers=private_markers,
        gitignore_check=gitignore_check,
    )
    return json_safe(
        {
            "mode": "share_ready_audit",
            "repo_root": str(root),
            "large_file_threshold_bytes": large_threshold_bytes,
            "large_files_over_threshold": large_files,
            "large_file_count": len(large_files),
            "raw_microscopy_files_present": raw_present,
            "raw_microscopy_file_count": len(raw_present),
            "raw_microscopy_files_tracked": raw_tracked,
            "raw_microscopy_tracked_count": len(raw_tracked),
            "zip_files_present": zip_files,
            "zip_file_count": len(zip_files),
            "absolute_path_hits": absolute_paths,
            "absolute_path_hit_count": len(absolute_paths),
            "private_marker_hits": private_markers,
            "private_marker_hit_count": len(private_markers),
            "gitignore_check": gitignore_check,
            "safe_to_push_as_is": bool(safe_to_push),
            "recommended_files_to_exclude": recommendations,
            "safe_to_commit_summary": [
                "Source code under scripts/ and src/.",
                "Tests under tests/.",
                "Documentation under docs/ after reviewing path references.",
                "Configuration files that do not expose private/local paths.",
            ],
            "should_stay_local_summary": [
                "Raw microscopy images.",
                "Generated results, previews, review packs, zips, and validation outputs.",
                "Local environments and cache directories.",
                "Files containing local absolute paths or private collaborator details.",
            ],
        }
    )


def iter_repo_files(repo_root: Path) -> list[Path]:
    ignored_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv", "env"}
    output: list[Path] = []
    for current_root, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [name for name in dirnames if name not in ignored_dirs]
        for filename in filenames:
            path = Path(current_root) / filename
            if path.is_file():
                output.append(path)
    return output


def is_share_audit_output(path: Path, repo_root: Path) -> bool:
    relative = relative_path(path, repo_root)
    parts = relative.parts
    return len(parts) >= 2 and parts[0] == "results" and parts[1] == "share_ready_audit"


def tracked_files(repo_root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def summarize_files(paths: list[Path], repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: str(relative_path(item, repo_root))):
        rows.append(
            {
                "path": str(relative_path(path, repo_root)),
                "suffix": path.suffix.lower(),
                "size_bytes": safe_size(path),
            }
        )
    return rows


def search_text_markers(repo_root: Path, files: list[Path], markers: list[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = read_text(path)
        if not text:
            continue
        for marker in markers:
            if marker in text:
                hits.append({"path": str(relative_path(path, repo_root)), "marker": marker})
    return hits


def search_private_markers(repo_root: Path, files: list[Path]) -> list[dict[str, Any]]:
    hits = search_text_markers(repo_root, files, PRIVATE_MARKERS)
    donor_path_terms = ["source_path", "image_path", "raw_tiff_dir", "confocal-root", "confocal_root"]
    for path in files:
        if path.suffix.lower() not in {".csv", ".json", ".md", ".yaml", ".yml"}:
            continue
        text = read_text(path)
        if not text:
            continue
        if any(term in text for term in donor_path_terms) and LOCAL_PATH_MARKERS[0] in text:
            hits.append({"path": str(relative_path(path, repo_root)), "marker": PRIVATE_MARKERS[1]})
    return hits


def check_gitignore_patterns(gitignore_text: str) -> dict[str, Any]:
    present = []
    missing = []
    lines = {line.strip() for line in gitignore_text.splitlines() if line.strip() and not line.strip().startswith("#")}
    for pattern in REQUIRED_GITIGNORE_PATTERNS:
        if pattern in lines:
            present.append(pattern)
        else:
            missing.append(pattern)
    return {"present_patterns": present, "missing_patterns": missing, "is_complete": not missing}


def build_recommendations(
    large_files: list[dict[str, Any]],
    raw_present: list[dict[str, Any]],
    raw_tracked: list[dict[str, Any]],
    zip_files: list[dict[str, Any]],
    absolute_paths: list[dict[str, Any]],
    private_markers: list[dict[str, Any]],
    gitignore_check: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    if large_files:
        recommendations.append("Keep large files over 25 MB out of GitHub unless intentionally using Git LFS.")
    if raw_present:
        recommendations.append("Do not commit raw microscopy image files; keep them local or in controlled data storage.")
    if raw_tracked:
        recommendations.append("Remove any tracked raw microscopy files from git history/index before sharing.")
    if zip_files:
        recommendations.append("Keep generated zip/review packs local unless intentionally shared outside the code repository.")
    if absolute_paths:
        recommendations.append("Replace or document local absolute paths before publishing public documentation.")
    if private_markers:
        recommendations.append("Review private marker hits before sharing.")
    missing = gitignore_check.get("missing_patterns", [])
    if missing:
        recommendations.append(f"Add missing .gitignore patterns: {', '.join(missing)}")
    if not recommendations:
        recommendations.append("No blocking share-readiness issues detected by this audit.")
    return recommendations


def write_share_ready_outputs(audit: dict[str, Any], paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(json_safe(audit), indent=2) + "\n", encoding="utf-8")
    paths["txt"].write_text(render_share_ready_text(audit), encoding="utf-8")
    paths["markdown"].write_text(render_share_ready_markdown(audit), encoding="utf-8")


def render_share_ready_text(audit: dict[str, Any]) -> str:
    lines = [
        "Share-readiness audit",
        f"repo_root: {audit['repo_root']}",
        f"safe_to_push_as_is: {audit['safe_to_push_as_is']}",
        f"large_file_count: {audit['large_file_count']}",
        f"raw_microscopy_file_count: {audit['raw_microscopy_file_count']}",
        f"raw_microscopy_tracked_count: {audit['raw_microscopy_tracked_count']}",
        f"zip_file_count: {audit['zip_file_count']}",
        f"absolute_path_hit_count: {audit['absolute_path_hit_count']}",
        f"private_marker_hit_count: {audit['private_marker_hit_count']}",
        f"gitignore_missing_patterns: {audit['gitignore_check']['missing_patterns']}",
        "",
        "Recommended exclusions/actions:",
    ]
    lines.extend(f"- {item}" for item in audit["recommended_files_to_exclude"])
    lines.extend(["", "Large files:"])
    lines.extend(format_file_rows(audit["large_files_over_threshold"]))
    lines.extend(["", "Raw microscopy files present:"])
    lines.extend(format_file_rows(audit["raw_microscopy_files_present"]))
    lines.extend(["", "Zip files present:"])
    lines.extend(format_file_rows(audit["zip_files_present"]))
    lines.extend(["", "Absolute path hits:"])
    lines.extend(format_hit_rows(audit["absolute_path_hits"]))
    lines.extend(["", "Private marker hits:"])
    lines.extend(format_hit_rows(audit["private_marker_hits"]))
    return "\n".join(lines) + "\n"


def render_share_ready_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Share-Ready Audit",
        "",
        "This audit checks repository hygiene before GitHub sharing. It does not delete files, rerun analyses, change algorithms, or alter scientific outputs.",
        "",
        "## Summary",
        "",
        f"- Safe to push as-is: `{audit['safe_to_push_as_is']}`",
        f"- Large files >25 MB: `{audit['large_file_count']}`",
        f"- Raw microscopy files present under repo: `{audit['raw_microscopy_file_count']}`",
        f"- Raw microscopy files tracked by git: `{audit['raw_microscopy_tracked_count']}`",
        f"- Zip files present: `{audit['zip_file_count']}`",
        f"- Local absolute path hits: `{audit['absolute_path_hit_count']}`",
        f"- Private marker hits: `{audit['private_marker_hit_count']}`",
        "",
        "## Recommended Exclusions",
        "",
    ]
    lines.extend(f"- {item}" for item in audit["recommended_files_to_exclude"])
    lines.extend(["", "## Safe To Commit", ""])
    lines.extend(f"- {item}" for item in audit["safe_to_commit_summary"])
    lines.extend(["", "## Should Stay Local", ""])
    lines.extend(f"- {item}" for item in audit["should_stay_local_summary"])
    lines.extend(["", "## Gitignore Check", ""])
    lines.append(f"- Missing patterns: `{audit['gitignore_check']['missing_patterns']}`")
    lines.extend(["", "## Details", ""])
    lines.append("See `results/share_ready_audit/share_ready_audit_summary.json` and `.txt` for full file/hit lists.")
    return "\n".join(lines) + "\n"


def format_file_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [f"- {row['path']} ({row['size_bytes']} bytes)" for row in rows]


def format_hit_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    return [f"- {row['path']}: {row['marker']}" for row in rows]


def relative_path(path: Path, repo_root: Path) -> Path:
    try:
        return path.resolve().relative_to(repo_root)
    except ValueError:
        return path


def safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
