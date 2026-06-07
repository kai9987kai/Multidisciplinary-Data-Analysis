"""Core statistical workflow with deterministic uncertainty estimates."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split

DEFAULT_SENTENCE = (
    "Reproducible analysis connects numerical evidence, statistical uncertainty, "
    "visual diagnostics, and clear language."
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for synthetic generation and model evaluation."""

    sample_size: int = 300
    seed: int = 42
    true_intercept: float = 2.0
    true_slope: float = 3.0
    noise_scale: float = 0.35
    test_fraction: float = 0.2
    calibration_fraction: float = 0.25
    confidence_level: float = 0.9
    cv_folds: int = 5
    cv_repeats: int = 5
    bootstrap_resamples: int = 2_000

    def validate(self) -> None:
        if self.sample_size < 30:
            raise ValueError("sample_size must be at least 30")
        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be positive")
        if not 0.1 <= self.test_fraction <= 0.4:
            raise ValueError("test_fraction must be between 0.1 and 0.4")
        if not 0.1 <= self.calibration_fraction <= 0.5:
            raise ValueError("calibration_fraction must be between 0.1 and 0.5")
        if not 0.5 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0.5 and 1")
        if self.cv_folds < 2 or self.cv_repeats < 1:
            raise ValueError("cross-validation requires at least 2 folds and 1 repeat")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")


@dataclass
class AnalysisResult:
    """All data required to inspect, serialize, or visualize one run."""

    config: AnalysisConfig
    source: str
    feature_name: str
    target_name: str
    data: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    model: LinearRegression
    conformal_radius: float


def generate_synthetic_data(config: AnalysisConfig) -> pd.DataFrame:
    """Generate a linear dataset with explicit zero-mean Gaussian noise."""

    config.validate()
    rng = np.random.default_rng(config.seed)
    feature = rng.uniform(0.0, 1.0, size=config.sample_size)
    noise = rng.normal(0.0, config.noise_scale, size=config.sample_size)
    target = config.true_intercept + config.true_slope * feature + noise
    return pd.DataFrame({"X": feature, "y": target})


def summarize_text(text: str, top_n: int = 5) -> dict[str, Any]:
    """Tokenize text without external model downloads and summarize frequencies."""

    if top_n < 1:
        raise ValueError("top_n must be positive")
    tokens = [token.casefold() for token in re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)]
    counts = Counter(tokens)
    return {
        "token_count": len(tokens),
        "unique_token_count": len(counts),
        "lexical_diversity": len(counts) / len(tokens) if tokens else 0.0,
        "top_terms": [{"term": term, "count": count} for term, count in counts.most_common(top_n)],
        "tokens": tokens,
    }


def run_analysis(
    data: pd.DataFrame,
    config: AnalysisConfig,
    *,
    feature_name: str = "X",
    target_name: str = "y",
    sentence: str = DEFAULT_SENTENCE,
    source: str = "synthetic",
) -> AnalysisResult:
    """Fit, calibrate, and evaluate a regression workflow without data leakage."""

    config.validate()
    frame = _validated_frame(data, feature_name, target_name, config)
    indices = np.arange(len(frame))

    development_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_fraction,
        random_state=config.seed,
        shuffle=True,
    )
    train_idx, calibration_idx = train_test_split(
        development_idx,
        test_size=config.calibration_fraction,
        random_state=config.seed + 1,
        shuffle=True,
    )

    x = frame[[feature_name]].to_numpy()
    y = frame[target_name].to_numpy()
    model = LinearRegression().fit(x[train_idx], y[train_idx])
    baseline = DummyRegressor(strategy="mean").fit(x[train_idx], y[train_idx])

    calibration_predictions = model.predict(x[calibration_idx])
    calibration_errors = np.abs(y[calibration_idx] - calibration_predictions)
    conformal_radius = _conformal_quantile(calibration_errors, config.confidence_level)

    test_predictions = model.predict(x[test_idx])
    lower = test_predictions - conformal_radius
    upper = test_predictions + conformal_radius
    residuals = y[test_idx] - test_predictions
    baseline_predictions = baseline.predict(x[test_idx])

    cv_metrics = _cross_validation_metrics(x[train_idx], y[train_idx], config)
    coefficient_intervals = _coefficient_intervals(
        x[train_idx, 0],
        y[train_idx],
        config,
    )

    rmse = float(root_mean_squared_error(y[test_idx], test_predictions))
    baseline_rmse = float(root_mean_squared_error(y[test_idx], baseline_predictions))
    interval_coverage = float(np.mean((y[test_idx] >= lower) & (y[test_idx] <= upper)))

    predictions = pd.DataFrame(
        {
            "row_index": test_idx,
            feature_name: x[test_idx, 0],
            "actual": y[test_idx],
            "predicted": test_predictions,
            "lower_bound": lower,
            "upper_bound": upper,
            "residual": residuals,
        }
    ).sort_values(feature_name, ignore_index=True)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    target_std = float(np.std(y, ddof=1))
    metrics: dict[str, Any] = {
        "dataset": {
            "source": source,
            "rows": len(frame),
            "feature": feature_name,
            "target": target_name,
            "feature_mean": float(np.mean(x)),
            "feature_std": float(np.std(x, ddof=1)),
            "target_mean": float(np.mean(y)),
            "target_std": target_std,
            "pearson_correlation": float(np.corrcoef(x[:, 0], y)[0, 1]),
            "missing_values": int(frame[[feature_name, target_name]].isna().sum().sum()),
            "duplicate_rows": int(frame[[feature_name, target_name]].duplicated().sum()),
        },
        "split": {
            "training_rows": len(train_idx),
            "calibration_rows": len(calibration_idx),
            "test_rows": len(test_idx),
        },
        "model": {
            "type": "ordinary_least_squares",
            "intercept": intercept,
            "slope": slope,
            "intercept_interval": coefficient_intervals["intercept"],
            "slope_interval": coefficient_intervals["slope"],
        },
        "holdout": {
            "mae": float(mean_absolute_error(y[test_idx], test_predictions)),
            "rmse": rmse,
            "r2": float(r2_score(y[test_idx], test_predictions)),
            "baseline_rmse": baseline_rmse,
            "rmse_improvement_percent": float(100 * (baseline_rmse - rmse) / baseline_rmse),
        },
        "cross_validation": cv_metrics,
        "prediction_interval": {
            "method": "split_conformal_absolute_residual",
            "target_coverage": config.confidence_level,
            "empirical_coverage": interval_coverage,
            "radius": conformal_radius,
            "mean_width": float(np.mean(upper - lower)),
        },
        "diagnostics": {
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals, ddof=1)),
            "normalized_rmse": rmse / target_std if target_std else math.nan,
        },
        "text_analysis": summarize_text(sentence),
        "configuration": asdict(config),
    }

    if source == "synthetic":
        metrics["known_truth"] = {
            "intercept": config.true_intercept,
            "slope": config.true_slope,
            "noise_scale": config.noise_scale,
            "intercept_error": intercept - config.true_intercept,
            "slope_error": slope - config.true_slope,
        }

    return AnalysisResult(
        config=config,
        source=source,
        feature_name=feature_name,
        target_name=target_name,
        data=frame,
        predictions=predictions,
        metrics=metrics,
        model=model,
        conformal_radius=conformal_radius,
    )


def _validated_frame(
    data: pd.DataFrame,
    feature_name: str,
    target_name: str,
    config: AnalysisConfig,
) -> pd.DataFrame:
    if feature_name == target_name:
        raise ValueError("feature and target columns must be different")
    missing_columns = [name for name in (feature_name, target_name) if name not in data.columns]
    if missing_columns:
        raise ValueError(f"missing required column(s): {', '.join(missing_columns)}")

    frame = data.copy().reset_index(drop=True)
    selected = frame[[feature_name, target_name]]
    if not all(pd.api.types.is_numeric_dtype(selected[column]) for column in selected):
        raise ValueError("feature and target columns must be numeric")
    if selected.isna().any().any():
        raise ValueError("feature and target columns must not contain missing values")
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError("feature and target columns must contain only finite values")
    if len(frame) < 30:
        raise ValueError("at least 30 rows are required")
    if selected[feature_name].nunique() < 2:
        raise ValueError("feature column must contain at least two distinct values")

    estimated_training_rows = int(
        len(frame) * (1 - config.test_fraction) * (1 - config.calibration_fraction)
    )
    if estimated_training_rows < config.cv_folds:
        raise ValueError("not enough training rows for the configured cross-validation folds")
    return frame


def _cross_validation_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, float | int]:
    splitter = RepeatedKFold(
        n_splits=config.cv_folds,
        n_repeats=config.cv_repeats,
        random_state=config.seed,
    )
    scores = cross_validate(
        LinearRegression(),
        x_train,
        y_train,
        cv=splitter,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
    )
    mae = -scores["test_mae"]
    rmse = -scores["test_rmse"]
    r2 = scores["test_r2"]
    return {
        "folds": config.cv_folds,
        "repeats": config.cv_repeats,
        "evaluations": len(rmse),
        "mae_mean": float(np.mean(mae)),
        "mae_std": float(np.std(mae, ddof=1)),
        "rmse_mean": float(np.mean(rmse)),
        "rmse_std": float(np.std(rmse, ddof=1)),
        "r2_mean": float(np.mean(r2)),
        "r2_std": float(np.std(r2, ddof=1)),
    }


def _conformal_quantile(absolute_errors: np.ndarray, confidence_level: float) -> float:
    sample_count = len(absolute_errors)
    rank = min(sample_count, math.ceil((sample_count + 1) * confidence_level))
    return float(np.partition(absolute_errors, rank - 1)[rank - 1])


def _coefficient_intervals(
    feature: np.ndarray,
    target: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, dict[str, float]]:
    common = {
        "paired": True,
        "vectorized": True,
        "n_resamples": config.bootstrap_resamples,
        "confidence_level": config.confidence_level,
        "method": "BCa",
    }
    slope_result = bootstrap(
        (feature, target),
        _slope_statistic,
        rng=np.random.default_rng(config.seed + 101),
        **common,
    )
    intercept_result = bootstrap(
        (feature, target),
        _intercept_statistic,
        rng=np.random.default_rng(config.seed + 102),
        **common,
    )
    return {
        "slope": {
            "confidence_level": config.confidence_level,
            "lower": float(slope_result.confidence_interval.low),
            "upper": float(slope_result.confidence_interval.high),
        },
        "intercept": {
            "confidence_level": config.confidence_level,
            "lower": float(intercept_result.confidence_interval.low),
            "upper": float(intercept_result.confidence_interval.high),
        },
    }


def _slope_statistic(
    feature: np.ndarray,
    target: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    feature_centered = feature - np.mean(feature, axis=axis, keepdims=True)
    target_centered = target - np.mean(target, axis=axis, keepdims=True)
    numerator = np.sum(feature_centered * target_centered, axis=axis)
    denominator = np.sum(feature_centered**2, axis=axis)
    return numerator / denominator


def _intercept_statistic(
    feature: np.ndarray,
    target: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    slope = _slope_statistic(feature, target, axis=axis)
    return np.mean(target, axis=axis) - slope * np.mean(feature, axis=axis)
