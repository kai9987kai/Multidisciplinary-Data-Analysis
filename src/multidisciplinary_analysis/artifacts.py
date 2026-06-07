"""Durable, inspectable artifacts for analysis runs."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from multidisciplinary_analysis.analysis import AnalysisResult

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def save_artifacts(result: AnalysisResult, output_dir: Path) -> dict[str, Path]:
    """Write the complete analysis contract and return its paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "dataset": output_dir / "dataset.csv",
        "predictions": output_dir / "predictions.csv",
        "metrics": output_dir / "metrics.json",
        "run": output_dir / "run.json",
        "plot": output_dir / "analysis.png",
        "report": output_dir / "report.md",
    }

    result.data.to_csv(paths["dataset"], index=False)
    result.predictions.to_csv(paths["predictions"], index=False)
    paths["metrics"].write_text(
        json.dumps(_json_ready(result.metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["run"].write_text(
        json.dumps(_run_metadata(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["report"].write_text(_render_report(result), encoding="utf-8")
    _save_plot(result, paths["plot"])
    return paths


def _save_plot(result: AnalysisResult, path: Path) -> None:
    feature = result.feature_name
    target = result.target_name
    data = result.data.sort_values(feature)
    predictions = result.predictions.sort_values(feature)
    x_grid = np.linspace(data[feature].min(), data[feature].max(), 250)
    center = result.model.predict(x_grid.reshape(-1, 1))
    lower = center - result.conformal_radius
    upper = center + result.conformal_radius

    figure, (fit_axis, residual_axis) = plt.subplots(1, 2, figsize=(12, 5))
    fit_axis.scatter(
        data[feature],
        data[target],
        color="#64748b",
        alpha=0.45,
        s=24,
        label="All observations",
    )
    fit_axis.scatter(
        predictions[feature],
        predictions["actual"],
        color="#2563eb",
        alpha=0.85,
        s=30,
        label="Test observations",
    )
    fit_axis.plot(x_grid, center, color="#dc2626", linewidth=2.2, label="Linear fit")
    fit_axis.fill_between(
        x_grid,
        lower,
        upper,
        color="#dc2626",
        alpha=0.14,
        label=f"{result.config.confidence_level:.0%} conformal interval",
    )
    fit_axis.set(title="Regression and calibrated uncertainty", xlabel=feature, ylabel=target)
    fit_axis.legend(frameon=False)
    fit_axis.grid(alpha=0.2)

    residual_axis.scatter(
        predictions["predicted"],
        predictions["residual"],
        color="#0f766e",
        alpha=0.8,
        s=32,
    )
    residual_axis.axhline(0, color="#111827", linewidth=1.2, linestyle="--")
    residual_axis.set(
        title="Holdout residual diagnostic",
        xlabel=f"Predicted {target}",
        ylabel="Actual - predicted",
    )
    residual_axis.grid(alpha=0.2)

    figure.suptitle("Multidisciplinary Data Analysis", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_report(result: AnalysisResult) -> str:
    model = result.metrics["model"]
    holdout = result.metrics["holdout"]
    interval = result.metrics["prediction_interval"]
    cross_validation = result.metrics["cross_validation"]
    text = result.metrics["text_analysis"]
    slope_interval = model["slope_interval"]
    top_terms = (
        ", ".join(f"{item['term']} ({item['count']})" for item in text["top_terms"]) or "none"
    )

    return (
        "# Analysis Report\n\n"
        f"- Source: `{result.source}`\n"
        f"- Rows: {result.metrics['dataset']['rows']}\n"
        f"- Model: `{result.target_name} = {model['intercept']:.4f} + "
        f"{model['slope']:.4f} * {result.feature_name}`\n"
        f"- Holdout RMSE: {holdout['rmse']:.4f}\n"
        f"- Holdout MAE: {holdout['mae']:.4f}\n"
        f"- Holdout R-squared: {holdout['r2']:.4f}\n"
        f"- RMSE improvement over mean baseline: "
        f"{holdout['rmse_improvement_percent']:.2f}%\n"
        f"- Repeated-CV RMSE: {cross_validation['rmse_mean']:.4f} "
        f"+/- {cross_validation['rmse_std']:.4f}\n"
        f"- {result.config.confidence_level:.0%} bootstrap slope interval: "
        f"[{slope_interval['lower']:.4f}, {slope_interval['upper']:.4f}]\n"
        f"- Conformal empirical coverage: {interval['empirical_coverage']:.2%} "
        f"(target {interval['target_coverage']:.0%})\n"
        f"- Text tokens: {text['token_count']} ({text['unique_token_count']} unique)\n"
        f"- Top terms: {top_terms}\n\n"
        "## Interpretation\n\n"
        "The holdout metrics estimate performance on unseen observations. The bootstrap "
        "interval quantifies coefficient sampling uncertainty, while the split-conformal "
        "interval targets marginal predictive coverage without assuming normally distributed "
        "residuals. These quantities are descriptive and predictive; they do not establish "
        "causal effects.\n"
    )


def _run_metadata(result: AnalysisResult) -> dict[str, Any]:
    packages = ["matplotlib", "numpy", "pandas", "scikit-learn", "scipy"]
    return {
        "project_version": "2.0.0",
        "source": result.source,
        "feature": result.feature_name,
        "target": result.target_name,
        "configuration": asdict(result.config),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "implementation": sys.implementation.name,
            "packages": {package: _package_version(package) for package in packages},
        },
    }


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
