from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sarcomere_analysis.diagnostics.spacing_review_pack import (
    REVIEW_INDEX_COLUMNS,
    ReviewImageContext,
    class_mask,
    export_review_pack,
    select_review_candidates,
    stabilize_review_index,
    write_review_outputs,
    write_spacing_review_panel,
)
from test_step_6_spacing import oriented_stripe_patch, single_patch_metrics, step6_config


def synthetic_candidates() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(4):
        rows.append(candidate_row("accepted_current", f"img{i}", f"p{i}", final_valid_for_spacing=True, final_invalid_reason="ok", selected_lag_px=12.0, selected_lag_um=1.2, peak_confidence=0.25, best_in_band_lag_px=12.0, best_global_lag_px=12.0))
    for i in range(4):
        rows.append(candidate_row("no_local_peak", f"img{i}", f"n{i}", rejected_reason_diagnostic="no_local_peak", final_invalid_reason="no_local_peak"))
    for i in range(4):
        rows.append(candidate_row("low_periodicity_confidence", f"img{i}", f"l{i}", rejected_reason_diagnostic="low_periodicity_confidence", final_invalid_reason="low_periodicity_confidence", peak_confidence=0.10, best_in_band_lag_px=14.0, best_in_band_peak_value=0.3, best_global_lag_px=14.0, best_global_peak_value=0.3))
    for i in range(4):
        rows.append(candidate_row("global_out_of_band", f"img{i}", f"g{i}", rejected_reason_diagnostic="best_global_peak_outside_expected_band", best_global_lag_px=4.0, best_global_peak_value=0.9))
    for i in range(4):
        rows.append(candidate_row("borderline_in_band", f"img{i}", f"b{i}", rejected_reason_diagnostic="low_periodicity_confidence", final_invalid_reason="low_periodicity_confidence", peak_confidence=0.14, best_in_band_lag_px=15.0, best_in_band_peak_value=0.35, best_global_lag_px=15.0, best_global_peak_value=0.35))
    return pd.DataFrame(rows)


def candidate_row(_class: str, image_id: str, patch_id: str, **overrides) -> dict[str, object]:
    _ = _class
    row = {
        "image_id": image_id,
        "donor_id": f"d{image_id}",
        "patch_id": patch_id,
        "y0": 0,
        "x0": 0,
        "y1": 64,
        "x1": 64,
        "valid_for_spacing_qc": True,
        "final_valid_for_spacing": False,
        "final_invalid_reason": "best_global_peak_outside_expected_band",
        "expected_min_lag_px": 10.0,
        "expected_max_lag_px": 16.0,
        "selected_lag_px": np.nan,
        "selected_lag_um": np.nan,
        "selected_peak_value": np.nan,
        "baseline_value": np.nan,
        "peak_prominence": np.nan,
        "peak_confidence": 0.0,
        "n_local_peaks_total": 1,
        "n_local_peaks_in_band": 0,
        "best_in_band_lag_px": np.nan,
        "best_in_band_peak_value": np.nan,
        "best_global_lag_px": np.nan,
        "best_global_peak_value": np.nan,
        "rejected_reason_diagnostic": "best_global_peak_outside_expected_band",
    }
    row.update(overrides)
    return row


def test_stratified_sampling_respects_max_per_class() -> None:
    selected, summary = select_review_candidates(synthetic_candidates(), max_per_class=2, seed=5)
    counts = selected["review_class"].value_counts().to_dict()
    assert all(count <= 2 for count in counts.values())
    assert summary["selected_counts_by_class"]["accepted_current"] == 2


def test_sampling_is_deterministic_with_seed() -> None:
    first, _ = select_review_candidates(synthetic_candidates(), max_per_class=2, seed=7)
    second, _ = select_review_candidates(synthetic_candidates(), max_per_class=2, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_missing_class_does_not_crash() -> None:
    table = synthetic_candidates().loc[~class_mask(synthetic_candidates(), "accepted_current")].copy()
    selected, summary = select_review_candidates(table, classes=["accepted_current", "no_local_peak"], max_per_class=2, seed=1)
    assert summary["available_counts_by_class"]["accepted_current"] == 0
    assert "accepted_current" in summary["missing_classes"]
    assert set(selected["review_class"]) == {"no_local_peak"}


def test_review_index_has_required_columns() -> None:
    index = stabilize_review_index(pd.DataFrame([{"image_id": "img1", "patch_id": "p1"}]))
    assert list(index.columns) == REVIEW_INDEX_COLUMNS


def test_summary_json_is_serializable(tmp_path: Path) -> None:
    index = stabilize_review_index(pd.DataFrame([{"image_id": "img1", "patch_id": "p1", "render_status": "ok"}]))
    summary = {"selected_counts_by_class": {"accepted_current": 1}, "limited": True}
    paths = write_review_outputs(index, summary, tmp_path)
    loaded = json.loads(paths["review_summary_json"].read_text(encoding="utf-8"))
    assert loaded["limited"]
    assert paths["review_index"].exists()


def test_output_directory_creation_is_safe(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "review"
    paths = write_review_outputs(pd.DataFrame(), {"selected_rows": 0}, out)
    assert out.exists()
    assert paths["review_index"].parent == out


def test_plotting_function_can_render_one_synthetic_panel(tmp_path: Path) -> None:
    cfg = step6_config(tmp_path)
    image = oriented_stripe_patch(12.0, np.pi / 2, shape=(64, 64))
    patch_row = single_patch_metrics(valid_for_spacing=True, orientation=np.pi / 2).iloc[0]
    candidate = pd.Series(candidate_row("accepted_current", "synthetic", "synthetic_p00000", final_valid_for_spacing=True, final_invalid_reason="ok", selected_lag_px=12.0, selected_lag_um=1.2, peak_confidence=0.25, best_in_band_lag_px=12.0, best_global_lag_px=12.0))
    candidate["review_class"] = "accepted_current"
    context = ReviewImageContext(
        image_id="synthetic",
        donor_id="d1",
        preprocessed_image=image,
        tissue_mask=np.ones_like(image, dtype=bool),
        patch_metrics=pd.DataFrame([patch_row]),
    )
    path = write_spacing_review_panel(context, patch_row, candidate, tmp_path / "panel.png", cfg)
    assert path.exists()
    assert path.stat().st_size > 0


def test_no_input_csv_is_modified(tmp_path: Path) -> None:
    cfg = step6_config(tmp_path)
    candidates = synthetic_candidates().head(0)
    candidate_path = tmp_path / "spacing_candidates.csv"
    patch_path = tmp_path / "per_patch_metrics.csv"
    manifest_path = tmp_path / "manifest.csv"
    candidates.to_csv(candidate_path, index=False)
    pd.DataFrame(columns=["image_id", "donor_id", "patch_id"]).to_csv(patch_path, index=False)
    pd.DataFrame(columns=["image_id", "donor_id", "image_path"]).to_csv(manifest_path, index=False)
    cfg["outputs"]["manifest_csv"] = str(manifest_path)
    before = candidate_path.read_bytes()

    export_review_pack(cfg, candidate_table=candidate_path, patch_table=patch_path, output_directory=tmp_path / "review")

    assert candidate_path.read_bytes() == before
