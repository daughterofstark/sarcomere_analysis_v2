from __future__ import annotations

import json
from pathlib import Path

from sarcomere_analysis.share_ready_audit import (
    build_share_ready_audit,
    check_gitignore_patterns,
    run_share_ready_audit,
)


def write_gitignore(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
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
        )
        + "\n",
        encoding="utf-8",
    )


def test_audit_collects_large_files(tmp_path: Path) -> None:
    write_gitignore(tmp_path / ".gitignore")
    large = tmp_path / "large.bin"
    large.write_bytes(b"0" * 2048)

    audit = build_share_ready_audit(tmp_path, large_threshold_bytes=1024)

    assert audit["large_file_count"] == 1
    assert audit["large_files_over_threshold"][0]["path"] == "large.bin"


def test_audit_detects_raw_image_files(tmp_path: Path) -> None:
    write_gitignore(tmp_path / ".gitignore")
    (tmp_path / "example.czi").write_bytes(b"raw")

    audit = build_share_ready_audit(tmp_path)

    assert audit["raw_microscopy_file_count"] == 1
    assert audit["raw_microscopy_files_present"][0]["path"] == "example.czi"


def test_audit_detects_zip_files(tmp_path: Path) -> None:
    write_gitignore(tmp_path / ".gitignore")
    (tmp_path / "pack.zip").write_bytes(b"zip")

    audit = build_share_ready_audit(tmp_path)

    assert audit["zip_file_count"] == 1
    assert audit["zip_files_present"][0]["path"] == "pack.zip"


def test_audit_detects_local_absolute_paths(tmp_path: Path) -> None:
    write_gitignore(tmp_path / ".gitignore")
    marker = "/Users/" + "medhasharma"
    (tmp_path / "README.md").write_text(f"{marker}/private/path\n", encoding="utf-8")

    audit = build_share_ready_audit(tmp_path)

    assert audit["absolute_path_hit_count"] == 1
    assert audit["absolute_path_hits"][0]["marker"] == marker


def test_audit_detects_private_markers(tmp_path: Path) -> None:
    write_gitignore(tmp_path / ".gitignore")
    marker = "One" + "Drive"
    (tmp_path / "notes.md").write_text(f"Stored in {marker} during analysis.\n", encoding="utf-8")

    audit = build_share_ready_audit(tmp_path)

    assert audit["private_marker_hit_count"] == 1
    assert audit["private_marker_hits"][0]["marker"] == marker


def test_gitignore_missing_patterns_are_reported() -> None:
    result = check_gitignore_patterns("*.tif\nresults/\n")

    assert "*.czi" in result["missing_patterns"]
    assert result["is_complete"] is False


def test_audit_recommendations_include_exclusions(tmp_path: Path) -> None:
    write_gitignore(tmp_path / ".gitignore")
    (tmp_path / "image.tif").write_bytes(b"raw")

    audit = build_share_ready_audit(tmp_path)

    assert any("raw microscopy" in item.lower() for item in audit["recommended_files_to_exclude"])


def test_summary_json_serializable(tmp_path: Path) -> None:
    write_gitignore(tmp_path / ".gitignore")

    audit = build_share_ready_audit(tmp_path)

    json.dumps(audit)


def test_script_can_write_outputs_and_markdown(tmp_path: Path) -> None:
    write_gitignore(tmp_path / ".gitignore")

    audit, paths = run_share_ready_audit(tmp_path, output_directory=tmp_path / "out", docs_directory=tmp_path / "docs")

    assert paths["json"].exists()
    assert paths["txt"].exists()
    assert paths["markdown"].exists()
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["mode"] == "share_ready_audit"
    assert "Safe to push as-is" in paths["markdown"].read_text(encoding="utf-8")
    assert audit["safe_to_push_as_is"] is True


def test_audit_does_not_modify_existing_files(tmp_path: Path) -> None:
    write_gitignore(tmp_path / ".gitignore")
    table = tmp_path / "results" / "tables" / "features_per_image.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("image_id,image_oop\n2.007-1,0.1\n", encoding="utf-8")
    before = table.read_bytes()

    run_share_ready_audit(tmp_path, output_directory=tmp_path / "out", docs_directory=tmp_path / "docs")

    assert table.read_bytes() == before
