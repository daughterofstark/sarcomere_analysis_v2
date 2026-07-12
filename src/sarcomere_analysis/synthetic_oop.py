from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from .config import output_dir
from .orientation import compute_orientation_analysis
from .validation_zdisc_masks import axial_angular_error_deg, iqr
from .zdisc_annotation import json_safe


SYNTHETIC_RESULT_COLUMNS = [
    "synthetic_id",
    "true_orientation_deg",
    "true_period_px",
    "disorder_level",
    "noise_level",
    "noise_sigma",
    "blur_level",
    "blur_sigma",
    "contrast",
    "background_gradient",
    "seed",
    "expected_oop_rank",
    "image_oop",
    "image_mean_orientation_deg",
    "axial_orientation_error_deg",
    "image_oop_heterogeneity",
    "n_orientation_valid_patches",
]

DISORDER_RANK = {"low": 3, "medium": 2, "high": 1}
DISORDER_STRENGTH = {"low": 0.0, "medium": 0.45, "high": 0.9}
NOISE_SIGMA = {"none": 0.0, "moderate": 0.08, "high": 0.18}
BLUR_SIGMA = {"none": 0.0, "moderate": 1.2}


def default_synthetic_oop_paths(cfg: dict[str, Any], output_directory: str | Path | None = None) -> dict[str, Path]:
    out_dir = Path(output_directory) if output_directory else output_dir(cfg) / "validation"
    return {
        "results_csv": out_dir / "synthetic_oop_validation_results.csv",
        "summary_json": out_dir / "synthetic_oop_validation_summary.json",
        "summary_txt": out_dir / "synthetic_oop_validation_summary.txt",
        "example_dir": out_dir / "synthetic_oop_examples",
    }


def generate_synthetic_striated_image(
    size: int | tuple[int, int] = 256,
    orientation_deg: float = 0.0,
    period_px: float = 16.0,
    disorder_level: str = "low",
    noise_sigma: float = 0.0,
    blur_sigma: float = 0.0,
    contrast: float = 1.0,
    background_gradient: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    shape = (int(size), int(size)) if isinstance(size, int) else (int(size[0]), int(size[1]))
    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    cx = xx - (shape[1] - 1) / 2.0
    cy = yy - (shape[0] - 1) / 2.0
    theta = np.deg2rad(float(orientation_deg))
    projection = cx * np.cos(theta) + cy * np.sin(theta)
    base = np.sin(2.0 * np.pi * projection / float(period_px))

    level = str(disorder_level).lower()
    if level not in DISORDER_STRENGTH:
        raise ValueError(f"Unsupported disorder_level {disorder_level!r}; expected one of {sorted(DISORDER_STRENGTH)}")
    strength = float(DISORDER_STRENGTH[level])
    if strength > 0:
        random_orientation = float(rng.uniform(0.0, 180.0))
        random_theta = np.deg2rad(random_orientation)
        random_projection = cx * np.cos(random_theta) + cy * np.sin(random_theta)
        secondary = np.sin(2.0 * np.pi * random_projection / float(period_px) + float(rng.uniform(0.0, 2.0 * np.pi)))
        displacement = rng.normal(0.0, strength * period_px, size=shape).astype(np.float32)
        displacement = ndi.gaussian_filter(displacement, sigma=max(float(period_px) / 2.0, 1.0))
        warped = np.sin(2.0 * np.pi * (projection + displacement) / float(period_px))
        base = (1.0 - strength) * base + 0.65 * strength * secondary + 0.35 * strength * warped

    image = 0.5 + 0.5 * float(contrast) * base
    if float(background_gradient) != 0.0:
        gradient = (xx / max(shape[1] - 1, 1)) - 0.5
        image = image + float(background_gradient) * gradient
    if float(noise_sigma) > 0:
        image = image + rng.normal(0.0, float(noise_sigma), size=shape)
    if float(blur_sigma) > 0:
        image = ndi.gaussian_filter(image, sigma=float(blur_sigma))
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def synthetic_patch_qc(size: int | tuple[int, int], synthetic_id: str) -> pd.DataFrame:
    shape = (int(size), int(size)) if isinstance(size, int) else (int(size[0]), int(size[1]))
    return pd.DataFrame(
        [
            {
                "image_id": str(synthetic_id),
                "patch_id": f"{synthetic_id}_p00000",
                "y0": 0,
                "x0": 0,
                "y1": shape[0],
                "x1": shape[1],
                "center_y": shape[0] / 2.0,
                "center_x": shape[1] / 2.0,
                "tissue_fraction": 1.0,
                "intensity_mean": np.nan,
                "intensity_std": np.nan,
                "rms_contrast": np.nan,
                "gradient_energy": np.nan,
                "valid_for_orientation": True,
                "valid_for_periodicity": True,
                "valid_for_spacing": True,
                "invalid_reason": "ok",
            }
        ]
    )


def run_orientation_on_synthetic(image: np.ndarray, synthetic_id: str, cfg: dict[str, Any]) -> dict[str, float | int]:
    mask = np.ones_like(image, dtype=bool)
    patch_qc = synthetic_patch_qc(image.shape, synthetic_id)
    result = compute_orientation_analysis(image, mask, patch_qc, cfg)
    return result.image_metrics


def scenario_grid(n_replicates: int = 1, seed: int = 123) -> list[dict[str, Any]]:
    orientations = [0, 30, 60, 90, 120, 150]
    noise_blur_pairs = [
        ("none", "none"),
        ("moderate", "none"),
        ("high", "none"),
        ("none", "moderate"),
    ]
    scenarios: list[dict[str, Any]] = []
    counter = 0
    for orientation in orientations:
        for disorder in ["low", "medium", "high"]:
            for noise_level, blur_level in noise_blur_pairs:
                for replicate in range(int(n_replicates)):
                    counter += 1
                    scenarios.append(
                        {
                            "synthetic_id": f"SYN_{counter:04d}",
                            "true_orientation_deg": float(orientation),
                            "true_period_px": 16.0,
                            "disorder_level": disorder,
                            "noise_level": noise_level,
                            "noise_sigma": NOISE_SIGMA[noise_level],
                            "blur_level": blur_level,
                            "blur_sigma": BLUR_SIGMA[blur_level],
                            "contrast": 1.0,
                            "background_gradient": 0.0,
                            "seed": int(seed + counter + replicate * 1000),
                            "expected_oop_rank": DISORDER_RANK[disorder],
                        }
                    )
    return scenarios


def validate_synthetic_oop(
    cfg: dict[str, Any],
    output_directory: str | Path | None = None,
    seed: int = 123,
    n_replicates: int = 1,
    size: int = 256,
    write_example_images: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    scenarios = scenario_grid(n_replicates=n_replicates, seed=seed)
    if len(scenarios) > 100:
        raise ValueError("Synthetic OOP validation is intentionally modest; reduce n_replicates to keep <=100 examples.")
    rows: list[dict[str, Any]] = []
    paths = default_synthetic_oop_paths(cfg, output_directory)
    if write_example_images:
        paths["example_dir"].mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        image = generate_synthetic_striated_image(
            size=size,
            orientation_deg=scenario["true_orientation_deg"],
            period_px=scenario["true_period_px"],
            disorder_level=scenario["disorder_level"],
            noise_sigma=scenario["noise_sigma"],
            blur_sigma=scenario["blur_sigma"],
            contrast=scenario["contrast"],
            background_gradient=scenario["background_gradient"],
            seed=scenario["seed"],
        )
        metrics = run_orientation_on_synthetic(image, scenario["synthetic_id"], cfg)
        recovered_orientation = float(metrics["image_mean_orientation_deg"])
        error = (
            axial_angular_error_deg(float(scenario["true_orientation_deg"]), recovered_orientation)
            if np.isfinite(recovered_orientation)
            else np.nan
        )
        rows.append(
            {
                **scenario,
                "image_oop": metrics["image_oop"],
                "image_mean_orientation_deg": recovered_orientation,
                "axial_orientation_error_deg": error,
                "image_oop_heterogeneity": metrics["image_oop_heterogeneity"],
                "n_orientation_valid_patches": metrics["n_orientation_valid_patches"],
            }
        )
        if write_example_images and scenario["disorder_level"] == "low" and scenario["noise_level"] == "none":
            write_example_image(image, paths["example_dir"] / f"{scenario['synthetic_id']}.png")
    results = stabilize_synthetic_results(pd.DataFrame(rows))
    summary = build_synthetic_oop_summary(results)
    write_synthetic_oop_outputs(results, summary, paths)
    return results, summary, paths


def stabilize_synthetic_results(results: pd.DataFrame) -> pd.DataFrame:
    table = results.copy(deep=True)
    for column in SYNTHETIC_RESULT_COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    return table[SYNTHETIC_RESULT_COLUMNS]


def build_synthetic_oop_summary(results: pd.DataFrame) -> dict[str, Any]:
    clean = results.loc[
        (results["disorder_level"] == "low")
        & (results["noise_level"] == "none")
        & (results["blur_level"] == "none")
    ].copy()
    errors = pd.to_numeric(clean["axial_orientation_error_deg"], errors="coerce").dropna()
    medians_by_disorder = recovered_oop_by_group(results, "disorder_level")
    monotonic = monotonic_oop_check(medians_by_disorder)
    return json_safe(
        {
            "mode": "synthetic_oop_orientation_validation",
            "synthetic_examples": int(len(results)),
            "clean_low_disorder_cases": int(len(clean)),
            "clean_case_median_angular_error_deg": float(np.nanmedian(errors)) if len(errors) else np.nan,
            "clean_case_max_angular_error_deg": float(np.nanmax(errors)) if len(errors) else np.nan,
            "clean_case_n": int(len(errors)),
            "recovered_oop_median_by_disorder_level": medians_by_disorder,
            "oop_monotonicity_low_gt_medium_gt_high": bool(monotonic),
            "recovered_oop_by_noise_level": recovered_oop_by_group(results, "noise_level"),
            "angular_error_by_noise_level": angular_error_by_group(results, "noise_level"),
            "recovered_oop_by_blur_level": recovered_oop_by_group(results, "blur_level"),
            "angular_error_by_blur_level": angular_error_by_group(results, "blur_level"),
            "degradation_failure_modes": degradation_failure_modes(results),
            "orientation_convention": "true_orientation_deg is the axial orientation of the sinusoidal intensity gradient estimated by the frozen structure-tensor implementation.",
            "interpretation_flags": [
                "controlled_synthetic_implementation_sanity_check_only",
                "does_not_prove_real_tissue_oop_is_biologically_valid",
                "manual_mask_validation_remains_pilot_and_did_not_validate_oop",
                "real_expert_validation_still_needed_for_publication_claims",
                "spacing_remains_exploratory_low_yield",
                "no_production_algorithm_or_threshold_changes",
            ],
        }
    )


def recovered_oop_by_group(results: pd.DataFrame, group_column: str) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for value, group in results.groupby(group_column, sort=True):
        values = pd.to_numeric(group["image_oop"], errors="coerce").dropna()
        output[str(value)] = None if values.empty else float(np.nanmedian(values))
    return output


def angular_error_by_group(results: pd.DataFrame, group_column: str) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for value, group in results.groupby(group_column, sort=True):
        errors = pd.to_numeric(group["axial_orientation_error_deg"], errors="coerce").dropna()
        output[str(value)] = {
            "n": int(len(errors)),
            "median": None if errors.empty else float(np.nanmedian(errors)),
            "iqr": None if errors.empty else iqr(errors),
        }
    return output


def monotonic_oop_check(medians_by_disorder: dict[str, float | None]) -> bool:
    low = medians_by_disorder.get("low")
    medium = medians_by_disorder.get("medium")
    high = medians_by_disorder.get("high")
    if low is None or medium is None or high is None:
        return False
    return bool(float(low) > float(medium) > float(high))


def degradation_failure_modes(results: pd.DataFrame) -> list[str]:
    modes: list[str] = []
    grouped = angular_error_by_group(results, "noise_level")
    for level, stats in grouped.items():
        median = stats["median"]
        if median is not None and float(median) > 30.0:
            modes.append(f"noise_level={level}: median angular error > 30 deg")
    grouped_blur = angular_error_by_group(results, "blur_level")
    for level, stats in grouped_blur.items():
        median = stats["median"]
        if median is not None and float(median) > 30.0:
            modes.append(f"blur_level={level}: median angular error > 30 deg")
    if not modes:
        modes.append("No median angular-error failure above 30 deg in this modest synthetic grid.")
    return modes


def write_example_image(image: np.ndarray, path: Path) -> Path:
    values = np.asarray(np.clip(image, 0.0, 1.0) * 255.0, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values, mode="L").save(path)
    return path


def write_synthetic_oop_outputs(results: pd.DataFrame, summary: dict[str, Any], paths: dict[str, Path]) -> None:
    paths["results_csv"].parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(paths["results_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
    lines = [f"{key}: {value}" for key, value in summary.items()]
    paths["summary_txt"].write_text("\n".join(lines) + "\n", encoding="utf-8")
