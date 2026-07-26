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
LOCAL_PATH_MARKERS = ["/Users/" + "medhasharma", "/mnt" + "/data", "/private" + "/tmp"]
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
    tracked_paths = [root / rel for rel in tracked]
    tracked_set = {path.resolve() for path in tracked_paths if path.exists()}
    untracked_paths = [path for path in files if path.resolve() not in tracked_set]
    gitignore_text = read_text(root / ".gitignore")

    large_files = summarize_files([path for path in files if safe_size(path) > large_threshold_bytes], root)
    large_tracked = summarize_files([path for path in tracked_paths if path.exists() and safe_size(path) > large_threshold_bytes], root)
    raw_present = summarize_files([path for path in files if path.suffix.lower() in RAW_IMAGE_SUFFIXES], root)
    raw_tracked = summarize_files([path for path in tracked_paths if path.suffix.lower() in RAW_IMAGE_SUFFIXES], root)
    zip_files = summarize_files([path for path in files if path.suffix.lower() in ZIP_SUFFIXES], root)
    zip_tracked = summarize_files([path for path in tracked_paths if path.suffix.lower() in ZIP_SUFFIXES], root)
    results_tracked = summarize_files([path for path in tracked_paths if relative_path(path, root).parts[:1] == ("results",)], root)
    marker_search_files = [path for path in files if not is_share_audit_output(path, root)]
    absolute_paths = search_text_markers(root, marker_search_files, LOCAL_PATH_MARKERS)
    private_markers = search_private_markers(root, marker_search_files)
    tracked_marker_search_files = [
        path for path in tracked_paths if path.exists() and not is_share_audit_output(path, root)
    ]
    tracked_absolute_paths = search_text_markers(root, tracked_marker_search_files, LOCAL_PATH_MARKERS)
    tracked_private_markers = search_private_markers(root, tracked_marker_search_files)
    untracked_local_result_archive_leakage = summarize_untracked_local_leakage(
        root,
        untracked_paths,
        large_threshold_bytes=large_threshold_bytes,
    )
    gitignore_check = check_gitignore_patterns(gitignore_text)

    safe_to_push_git = (
        not large_tracked
        and not raw_tracked
        and not zip_tracked
        and not results_tracked
        and not tracked_absolute_paths
        and not tracked_private_markers
        and not gitignore_check["missing_patterns"]
    )
    safe_to_share_folder_archive = (
        safe_to_push_git
        and not large_files
        and not raw_present
        and not zip_files
        and not absolute_paths
        and not private_markers
        and not untracked_local_result_archive_leakage
    )
    recommendations = build_recommendations(
        large_files=large_files,
        large_tracked=large_tracked,
        raw_present=raw_present,
        raw_tracked=raw_tracked,
        zip_files=zip_files,
        zip_tracked=zip_tracked,
        results_tracked=results_tracked,
        absolute_paths=absolute_paths,
        tracked_absolute_paths=tracked_absolute_paths,
        private_markers=private_markers,
        tracked_private_markers=tracked_private_markers,
        gitignore_check=gitignore_check,
    )
    return json_safe(
        {
            "mode": "share_ready_audit",
            "repo_root": str(root),
            "tracked_file_count": len(tracked),
            "large_file_threshold_bytes": large_threshold_bytes,
            "large_files_over_threshold": large_files,
            "large_file_count": len(large_files),
            "tracked_large_files_over_threshold": large_tracked,
            "tracked_large_file_count": len(large_tracked),
            "raw_microscopy_files_present": raw_present,
            "raw_microscopy_file_count": len(raw_present),
            "raw_microscopy_files_tracked": raw_tracked,
            "raw_microscopy_tracked_count": len(raw_tracked),
            "zip_files_present": zip_files,
            "zip_file_count": len(zip_files),
            "zip_files_tracked": zip_tracked,
            "zip_tracked_count": len(zip_tracked),
            "results_files_tracked": results_tracked,
            "results_tracked_count": len(results_tracked),
            "absolute_path_hits": absolute_paths,
            "absolute_path_hit_count": len(absolute_paths),
            "tracked_local_absolute_path_hits": tracked_absolute_paths,
            "tracked_local_absolute_path_hit_count": len(tracked_absolute_paths),
            "private_marker_hits": private_markers,
            "private_marker_hit_count": len(private_markers),
            "tracked_private_path_leakage_hits": tracked_private_markers,
            "tracked_private_path_leakage_hit_count": len(tracked_private_markers),
            "untracked_local_result_archive_leakage_hits": untracked_local_result_archive_leakage,
            "untracked_local_result_archive_leakage_count": len(untracked_local_result_archive_leakage),
            "gitignore_check": gitignore_check,
            "safe_to_push_git": bool(safe_to_push_git),
            "safe_to_share_folder_archive": bool(safe_to_share_folder_archive),
            "safe_to_push_as_is": bool(safe_to_push_git),
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


def summarize_untracked_local_leakage(
    repo_root: Path,
    untracked_paths: list[Path],
    large_threshold_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(untracked_paths, key=lambda item: str(relative_path(item, repo_root))):
        relative = relative_path(path, repo_root)
        is_result = relative.parts[:1] == ("results",)
        is_archive = path.suffix.lower() in ZIP_SUFFIXES
        is_large = safe_size(path) > large_threshold_bytes
        if is_result or is_archive or is_large:
            rows.append(
                {
                    "path": str(relative),
                    "suffix": path.suffix.lower(),
                    "size_bytes": safe_size(path),
                    "reason": ";".join(
                        reason
                        for reason, present in [
                            ("untracked_result", is_result),
                            ("archive", is_archive),
                            ("large_file", is_large),
                        ]
                        if present
                    ),
                }
            )
    return rows


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
    large_tracked: list[dict[str, Any]],
    raw_present: list[dict[str, Any]],
    raw_tracked: list[dict[str, Any]],
    zip_files: list[dict[str, Any]],
    zip_tracked: list[dict[str, Any]],
    results_tracked: list[dict[str, Any]],
    absolute_paths: list[dict[str, Any]],
    tracked_absolute_paths: list[dict[str, Any]],
    private_markers: list[dict[str, Any]],
    tracked_private_markers: list[dict[str, Any]],
    gitignore_check: dict[str, Any],
) -> list[str]:
    recommendations: list[str] = []
    if large_tracked:
        recommendations.append("Remove tracked large files over 25 MB or use Git LFS intentionally.")
    if raw_present:
        recommendations.append("Do not commit raw microscopy image files; keep them local or in controlled data storage.")
    if raw_tracked:
        recommendations.append("Remove any tracked raw microscopy files from git history/index before sharing.")
    if zip_tracked:
        recommendations.append("Remove tracked zip/review packs from git before sharing.")
    elif zip_files:
        recommendations.append("Keep generated zip/review packs local and ignored unless intentionally shared outside the code repository.")
    if results_tracked:
        recommendations.append("Remove tracked generated results from git; results should stay local.")
    if tracked_absolute_paths:
        recommendations.append("Replace tracked local absolute paths with placeholders or relative paths before publishing.")
    elif absolute_paths:
        recommendations.append("Local absolute paths remain in ignored/local outputs; do not share a folder archive as-is.")
    if tracked_private_markers:
        recommendations.append("Review tracked private/path-leakage hits before sharing.")
    elif private_markers:
        recommendations.append("Private/path-leakage hits remain in ignored/local outputs; keep those outputs out of GitHub.")
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
        f"safe_to_push_git: {audit['safe_to_push_git']}",
        f"safe_to_share_folder_archive: {audit['safe_to_share_folder_archive']}",
        f"tracked_file_count: {audit['tracked_file_count']}",
        f"large_file_count: {audit['large_file_count']}",
        f"tracked_large_file_count: {audit['tracked_large_file_count']}",
        f"raw_microscopy_file_count: {audit['raw_microscopy_file_count']}",
        f"raw_microscopy_tracked_count: {audit['raw_microscopy_tracked_count']}",
        f"zip_file_count: {audit['zip_file_count']}",
        f"zip_tracked_count: {audit['zip_tracked_count']}",
        f"results_tracked_count: {audit['results_tracked_count']}",
        f"absolute_path_hit_count: {audit['absolute_path_hit_count']}",
        f"tracked_local_absolute_path_hit_count: {audit['tracked_local_absolute_path_hit_count']}",
        f"private_marker_hit_count: {audit['private_marker_hit_count']}",
        f"tracked_private_path_leakage_hit_count: {audit['tracked_private_path_leakage_hit_count']}",
        f"untracked_local_result_archive_leakage_count: {audit['untracked_local_result_archive_leakage_count']}",
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
    lines.extend(["", "Tracked absolute path hits:"])
    lines.extend(format_hit_rows(audit["tracked_local_absolute_path_hits"]))
    lines.extend(["", "Private marker hits:"])
    lines.extend(format_hit_rows(audit["private_marker_hits"]))
    lines.extend(["", "Tracked private/path leakage hits:"])
    lines.extend(format_hit_rows(audit["tracked_private_path_leakage_hits"]))
    return "\n".join(lines) + "\n"


def render_share_ready_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Share-Ready Audit",
        "",
        "This audit checks repository hygiene before GitHub sharing. It does not delete files, rerun analyses, change algorithms, or alter scientific outputs.",
        "",
        "## Summary",
        "",
        f"- Safe to push Git repository: `{audit['safe_to_push_git']}`",
        f"- Safe to share local folder/archive as-is: `{audit['safe_to_share_folder_archive']}`",
        f"- Tracked files inspected: `{audit['tracked_file_count']}`",
        f"- Large files >25 MB: `{audit['large_file_count']}`",
        f"- Tracked large files >25 MB: `{audit['tracked_large_file_count']}`",
        f"- Raw microscopy files present under repo: `{audit['raw_microscopy_file_count']}`",
        f"- Raw microscopy files tracked by git: `{audit['raw_microscopy_tracked_count']}`",
        f"- Zip files present: `{audit['zip_file_count']}`",
        f"- Zip files tracked by git: `{audit['zip_tracked_count']}`",
        f"- Results files tracked by git: `{audit['results_tracked_count']}`",
        f"- Local absolute path hits: `{audit['absolute_path_hit_count']}`",
        f"- Tracked local absolute path hits: `{audit['tracked_local_absolute_path_hit_count']}`",
        f"- Private marker hits: `{audit['private_marker_hit_count']}`",
        f"- Tracked private/path leakage hits: `{audit['tracked_private_path_leakage_hit_count']}`",
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
