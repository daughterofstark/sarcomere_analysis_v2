# Share-Ready Audit

This audit checks repository hygiene before GitHub sharing. It does not delete files, rerun analyses, change algorithms, or alter scientific outputs.

## Summary

- Safe to push as-is: `False`
- Large files >25 MB: `2`
- Raw microscopy files present under repo: `0`
- Raw microscopy files tracked by git: `0`
- Zip files present: `3`
- Local absolute path hits: `199`
- Private marker hits: `160`

## Recommended Exclusions

- Keep large files over 25 MB out of GitHub unless intentionally using Git LFS.
- Keep generated zip/review packs local unless intentionally shared outside the code repository.
- Replace or document local absolute paths before publishing public documentation.
- Review private marker hits before sharing.

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
