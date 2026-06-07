from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multidisciplinary_analysis.analysis import (
    AnalysisConfig,
    generate_synthetic_data,
    run_analysis,
    summarize_text,
)


def fast_config(**overrides: object) -> AnalysisConfig:
    values = {
        "sample_size": 400,
        "seed": 17,
        "bootstrap_resamples": 300,
        "cv_repeats": 2,
    }
    values.update(overrides)
    return AnalysisConfig(**values)


def test_synthetic_generation_is_reproducible_and_zero_mean() -> None:
    config = fast_config(sample_size=2_000)
    first = generate_synthetic_data(config)
    second = generate_synthetic_data(config)

    pd.testing.assert_frame_equal(first, second)
    noise = first["y"] - (config.true_intercept + config.true_slope * first["X"])
    assert abs(noise.mean()) < 0.03


def test_analysis_recovers_signal_and_reports_uncertainty() -> None:
    config = fast_config()
    result = run_analysis(generate_synthetic_data(config), config)

    holdout = result.metrics["holdout"]
    slope_interval = result.metrics["model"]["slope_interval"]
    prediction_interval = result.metrics["prediction_interval"]

    assert holdout["rmse"] < holdout["baseline_rmse"]
    assert holdout["r2"] > 0.75
    assert slope_interval["lower"] < config.true_slope < slope_interval["upper"]
    assert prediction_interval["mean_width"] > 0
    assert 0 <= prediction_interval["empirical_coverage"] <= 1
    assert result.predictions["X"].is_monotonic_increasing


def test_analysis_is_deterministic() -> None:
    config = fast_config()
    data = generate_synthetic_data(config)
    first = run_analysis(data, config)
    second = run_analysis(data, config)

    assert first.metrics == second.metrics
    pd.testing.assert_frame_equal(first.predictions, second.predictions)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"X": range(30), "y": [np.nan] * 30}), "missing"),
        (pd.DataFrame({"X": ["text"] * 30, "y": range(30)}), "numeric"),
        (pd.DataFrame({"X": [1.0] * 30, "y": range(30)}), "distinct"),
    ],
)
def test_invalid_data_is_rejected(frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_analysis(frame, fast_config(sample_size=30))


def test_text_summary_handles_case_and_punctuation() -> None:
    summary = summarize_text("Data, data! Evidence-driven analysis.")

    assert summary["token_count"] == 5
    assert summary["unique_token_count"] == 4
    assert summary["top_terms"][0] == {"term": "data", "count": 2}
