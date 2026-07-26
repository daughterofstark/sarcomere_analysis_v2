# Confocal Gate Decision

This decision record documents the final visual-review decision for the confocal confident-striation gate.

It does not change algorithms, thresholds, default config, widefield outputs, confocal analysis outputs, or spacing outputs.

## Decision

- Primary gate: `moderate`
- Secondary/sensitivity gate: `moderate_relaxed_combined`

Use the moderate gate as the primary exploratory confocal gate. Keep moderate_relaxed_combined only as a secondary sensitivity/review option.

## Rationale

- The moderate gate is conservative and visually safer across the reviewed images.
- moderate_relaxed_combined improves recall in 5138.
- moderate_relaxed_combined is too broad in 6052-CLEAR_STRIPES and 7028.
- moderate_relaxed_combined is risky in 3112 because it may include Z-disc-like structures that do not form clear striations.

## Image-Specific Notes

- `5138`: moderate_relaxed_combined is acceptable as a sensitivity option because it improves coverage.
- `6052-CLEAR_STRIPES`: Use moderate; moderate_relaxed_combined becomes too broad.
- `3112`: Use moderate; relaxed selection is exploratory only for short or fragmented structures and risks including non-striated Z-disc-like signal.
- `7028`: Use moderate; moderate_relaxed_combined has broad-selection risk.

## Analysis Implication

- The calibrated spacing audit remains based on the moderate gate.
- Do not report moderate_relaxed_combined spacing unless a separate refreshed audit is explicitly run and visually reviewed.
- moderate_relaxed_combined may be used only for sensitivity figures or review, not primary summaries.

## Allowed Claims

- The moderate gate is the primary exploratory confocal gate.
- The relaxed gate may recover additional regions but is not adopted globally.
- Confocal selected-region spacing remains promising but exploratory.

## Not Allowed Claims

- moderate_relaxed_combined is a final validated segmentation.
- moderate_relaxed_combined should replace moderate globally.
- Relaxed-gate spacing results exist.
- Spacing is biologically validated.

## Scope

This is a documentation-only decision record. It is not a refreshed spacing audit, not threshold tuning, and not a biological validation claim.
