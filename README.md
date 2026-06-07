# Multidisciplinary Python Data Analysis

A reproducible, testable data-analysis workflow combining numerical computing,
data manipulation, regression, uncertainty quantification, visualization, and
lightweight text analysis.

The original repository was a single interactive script. The project now
produces an auditable set of data, metrics, predictions, diagnostics, and a
human-readable report from either synthetic data or a user-provided CSV file.

## What It Demonstrates

- Deterministic synthetic data generation with an explicit data-generating process
- Leakage-resistant train/calibration/test separation
- Linear regression evaluated with MAE, RMSE, and R-squared
- Comparison with an intercept-only baseline
- Repeated cross-validation for performance stability
- Bootstrap confidence intervals for model coefficients
- Split-conformal prediction intervals with empirical coverage reporting
- Regression and residual diagnostics saved as a headless PNG
- Dependency-free tokenization and word-frequency analysis
- Machine-readable artifacts, tests, linting, packaging, and CI

## Quick Start

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python main.py
```

After installation, the equivalent console command is:

```powershell
mda --output-dir artifacts
```

Run the quality checks:

```powershell
ruff check .
pytest
python -m build
```

## Analyze a CSV

```powershell
mda --input data.csv --feature temperature --target energy_use `
    --output-dir artifacts/energy
```

The selected feature and target must be numeric, finite, non-missing columns.
At least 30 complete rows are required.

Useful options:

```text
--samples 500
--seed 42
--confidence 0.90
--bootstrap-resamples 2000
--sentence "Reliable analysis needs reproducible evidence."
```

## Output Contract

Each run creates:

| File | Purpose |
| --- | --- |
| `dataset.csv` | Exact input used by the workflow |
| `predictions.csv` | Holdout predictions, residuals, and conformal bounds |
| `metrics.json` | Dataset, model, evaluation, uncertainty, and text metrics |
| `run.json` | Configuration and runtime package versions |
| `analysis.png` | Regression fit, prediction interval, and residual diagnostics |
| `report.md` | Concise interpretation of the run |

Generated outputs are ignored by Git by default.

## Statistical Design

For synthetic runs, the data-generating process is:

```text
X ~ Uniform(0, 1)
epsilon ~ Normal(0, noise_scale)
y = intercept + slope * X + epsilon
```

The default values are an intercept of `2`, slope of `3`, and noise scale of
`0.35`. Randomness is controlled with NumPy's `Generator`; no global random
state is mutated.

The data is separated into training, calibration, and test partitions. The
regression model is fit only on training data. Calibration residuals determine
the split-conformal interval width, and the test partition is used once for
final metrics and empirical interval coverage. Repeated cross-validation is
reported as an additional stability estimate, not as a replacement for the
untouched holdout.

## Research and Practice Basis

- [scikit-learn: common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [SciPy: bootstrap confidence intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [Lei et al.: Distribution-Free Predictive Inference for Regression](https://arxiv.org/abs/1604.04173)
- [Python Packaging User Guide: `pyproject.toml`](https://packaging.python.org/en/latest/specifications/pyproject-toml/)

Split-conformal coverage is marginal and relies on exchangeable observations.
It does not guarantee coverage for every subgroup or under distribution shift.
The coefficient bootstrap also describes sampling uncertainty, not causality.

## Project Layout

```text
src/multidisciplinary_analysis/
  analysis.py   # Pure analysis and uncertainty functions
  artifacts.py  # CSV, JSON, Markdown, and PNG outputs
  cli.py        # Command-line orchestration
tests/          # Determinism, statistics, artifacts, and CLI tests
```
