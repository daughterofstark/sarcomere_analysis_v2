# Share-Ready Audit

This audit checks repository hygiene before GitHub sharing. It does not delete files, rerun analyses, change algorithms, or alter scientific outputs.

## Summary

- Safe to push Git repository: `True`
- Safe to share local folder/archive as-is: `False`
- Tracked files inspected: `218`
- Large files >25 MB: `2`
- Tracked large files >25 MB: `0`
- Raw microscopy files present under repo: `0`
- Raw microscopy files tracked by git: `0`
- Zip files present: `3`
- Zip files tracked by git: `0`
- Results files tracked by git: `0`
- Local absolute path hits: `185`
- Tracked local absolute path hits: `0`
- Private marker hits: `158`
- Tracked private/path leakage hits: `0`

## Recommended Exclusions

- Keep generated zip/review packs local and ignored unless intentionally shared outside the code repository.
- Local absolute paths remain in ignored/local outputs; do not share a folder archive as-is.
- Private/path-leakage hits remain in ignored/local outputs; keep those outputs out of GitHub.

## Safe To Commit

- Source code under scripts/ and src/.
- Tests under tests/.
- Documentation under docs/ after reviewing path references.
- Configuration files that do not expose private/local paths.

## Should Stay Local

- Raw microscopy images.
- Generated results, previews, review packs, zips, and validation outputs.
- Local environments and cache directories.
- Files containing local absolute paths or private collaborator details.

## Gitignore Check

- Missing patterns: `[]`

## Details

See `results/share_ready_audit/share_ready_audit_summary.json` and `.txt` for full file/hit lists.
