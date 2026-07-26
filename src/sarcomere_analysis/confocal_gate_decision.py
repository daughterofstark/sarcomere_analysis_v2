from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import output_dir
from .zdisc_annotation import json_safe


PRIMARY_GATE = "moderate"
SECONDARY_GATE = "moderate_relaxed_combined"


def default_confocal_gate_decision_paths(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(output_directory) if output_directory else output_dir(cfg) / "confocal_gate_refinement"
    docs_dir = Path(docs_directory) if docs_directory else Path("docs")
    return {
        "json": root / "confocal_gate_decision_summary.json",
        "txt": root / "confocal_gate_decision_summary.txt",
        "markdown": docs_dir / "CONFOCAL_GATE_DECISION.md",
    }


def build_confocal_gate_decision() -> dict[str, Any]:
    return json_safe(
        {
            "mode": "confocal_gate_decision_record",
            "primary_gate": PRIMARY_GATE,
            "secondary_sensitivity_gate": SECONDARY_GATE,
            "final_decision": (
                "Use the moderate gate as the primary exploratory confocal gate. "
                "Keep moderate_relaxed_combined only as a secondary sensitivity/review option."
            ),
            "rationale": [
                "The moderate gate is conservative and visually safer across the reviewed images.",
                "moderate_relaxed_combined improves recall in 5138.",
                "moderate_relaxed_combined is too broad in 6052-CLEAR_STRIPES and 7028.",
                (
                    "moderate_relaxed_combined is risky in 3112 because it may include "
                    "Z-disc-like structures that do not form clear striations."
                ),
            ],
            "image_specific_notes": {
                "5138": "moderate_relaxed_combined is acceptable as a sensitivity option because it improves coverage.",
                "6052-CLEAR_STRIPES": "Use moderate; moderate_relaxed_combined becomes too broad.",
                "3112": (
                    "Use moderate; relaxed selection is exploratory only for short or fragmented structures "
                    "and risks including non-striated Z-disc-like signal."
                ),
                "7028": "Use moderate; moderate_relaxed_combined has broad-selection risk.",
            },
            "analysis_implications": [
                "The calibrated spacing audit remains based on the moderate gate.",
                (
                    "Do not report moderate_relaxed_combined spacing unless a separate refreshed audit "
                    "is explicitly run and visually reviewed."
                ),
                "moderate_relaxed_combined may be used only for sensitivity figures or review, not primary summaries.",
            ],
            "allowed_claims": [
                "The moderate gate is the primary exploratory confocal gate.",
                "The relaxed gate may recover additional regions but is not adopted globally.",
                "Confocal selected-region spacing remains promising but exploratory.",
            ],
            "not_allowed_claims": [
                "moderate_relaxed_combined is a final validated segmentation.",
                "moderate_relaxed_combined should replace moderate globally.",
                "Relaxed-gate spacing results exist.",
                "Spacing is biologically validated.",
            ],
            "no_changes_made_to": [
                "widefield outputs",
                "production algorithms",
                "default config",
                "existing confocal analysis outputs",
                "spacing outputs",
            ],
        }
    )


def write_confocal_gate_decision(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
    docs_directory: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    decision = build_confocal_gate_decision()
    paths = default_confocal_gate_decision_paths(cfg, output_directory, docs_directory)
    write_confocal_gate_decision_outputs(decision, paths)
    return decision, paths


def write_confocal_gate_decision_outputs(decision: dict[str, Any], paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(json_safe(decision), indent=2) + "\n", encoding="utf-8")
    paths["txt"].write_text(render_confocal_gate_decision_text(decision), encoding="utf-8")
    paths["markdown"].write_text(render_confocal_gate_decision_markdown(decision), encoding="utf-8")


def render_confocal_gate_decision_text(decision: dict[str, Any]) -> str:
    lines = [
        "Confocal gate decision",
        f"primary_gate: {decision['primary_gate']}",
        f"secondary_sensitivity_gate: {decision['secondary_sensitivity_gate']}",
        f"final_decision: {decision['final_decision']}",
        "",
        "Rationale:",
    ]
    lines.extend(f"- {item}" for item in decision["rationale"])
    lines.extend(["", "Image-specific notes:"])
    for image_id, note in decision["image_specific_notes"].items():
        lines.append(f"- {image_id}: {note}")
    lines.extend(["", "Analysis implications:"])
    lines.extend(f"- {item}" for item in decision["analysis_implications"])
    lines.extend(["", "Allowed claims:"])
    lines.extend(f"- {item}" for item in decision["allowed_claims"])
    lines.extend(["", "Not allowed claims:"])
    lines.extend(f"- {item}" for item in decision["not_allowed_claims"])
    return "\n".join(lines) + "\n"


def render_confocal_gate_decision_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# Confocal Gate Decision",
        "",
        "This decision record documents the final visual-review decision for the confocal confident-striation gate.",
        "",
        "It does not change algorithms, thresholds, default config, widefield outputs, confocal analysis outputs, or spacing outputs.",
        "",
        "## Decision",
        "",
        f"- Primary gate: `{decision['primary_gate']}`",
        f"- Secondary/sensitivity gate: `{decision['secondary_sensitivity_gate']}`",
        "",
        decision["final_decision"],
        "",
        "## Rationale",
        "",
    ]
    lines.extend(f"- {item}" for item in decision["rationale"])
    lines.extend(["", "## Image-Specific Notes", ""])
    for image_id, note in decision["image_specific_notes"].items():
        lines.append(f"- `{image_id}`: {note}")
    lines.extend(["", "## Analysis Implication", ""])
    lines.extend(f"- {item}" for item in decision["analysis_implications"])
    lines.extend(["", "## Allowed Claims", ""])
    lines.extend(f"- {item}" for item in decision["allowed_claims"])
    lines.extend(["", "## Not Allowed Claims", ""])
    lines.extend(f"- {item}" for item in decision["not_allowed_claims"])
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This is a documentation-only decision record. It is not a refreshed spacing audit, not threshold tuning, and not a biological validation claim.",
            "",
        ]
    )
    return "\n".join(lines)
