from __future__ import annotations

import inspect

import numpy as np

import sarcomere_analysis.preprocessing as preprocessing
from sarcomere_analysis.preprocessing import preprocess_image


def base_config() -> dict:
    return {
        "preprocessing": {
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "background_method": "gaussian",
            "background_sigma_px": 2.0,
            "enable_denoise": False,
            "denoise_sigma_px": 0.5,
            "output_dtype": "float32",
            "cache_preprocessed": False,
        }
    }


def test_preprocessing_preserves_shape_and_dtype() -> None:
    raw = np.arange(100, dtype=np.uint16).reshape(10, 10)
    result = preprocess_image(raw, base_config())
    assert result.image.shape == raw.shape
    assert result.image.dtype == np.float32


def test_preprocessing_outputs_finite_unit_range() -> None:
    raw = np.linspace(0, 5000, 256, dtype=np.float32).reshape(16, 16)
    result = preprocess_image(raw, base_config())
    assert np.isfinite(result.image).all()
    assert float(result.image.min()) >= -1e-6
    assert float(result.image.max()) <= 1.0 + 1e-6


def test_constant_image_does_not_crash() -> None:
    raw = np.full((12, 12), 42, dtype=np.uint16)
    result = preprocess_image(raw, base_config())
    assert np.isfinite(result.image).all()
    assert result.image.shape == raw.shape


def test_noisy_image_does_not_produce_nans() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(loc=100, scale=20, size=(32, 32)).astype(np.float32)
    result = preprocess_image(raw, base_config())
    assert not np.isnan(result.image).any()


def test_config_drives_percentiles_and_denoising() -> None:
    config = base_config()
    config["preprocessing"]["lower_percentile"] = 5.0
    config["preprocessing"]["upper_percentile"] = 95.0
    config["preprocessing"]["enable_denoise"] = True
    config["preprocessing"]["denoise_sigma_px"] = 1.25
    raw = np.arange(400, dtype=np.float32).reshape(20, 20)
    result = preprocess_image(raw, config)
    assert result.metadata["lower_percentile"] == 5.0
    assert result.metadata["upper_percentile"] == 95.0
    assert result.metadata["enable_denoise"] is True
    assert result.metadata["denoise_sigma_px"] == 1.25


def test_measurement_preprocessing_does_not_import_adaptive_equalization() -> None:
    source = inspect.getsource(preprocessing)
    assert "equalize_adapthist" not in source
    assert "createCLAHE" not in source
