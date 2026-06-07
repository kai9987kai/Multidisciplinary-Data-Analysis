"""Command-line interface for the analysis workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from multidisciplinary_analysis.analysis import (
    DEFAULT_SENTENCE,
    AnalysisConfig,
    generate_synthetic_data,
    run_analysis,
)
from multidisciplinary_analysis.artifacts import save_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mda",
        description="Run a reproducible regression, uncertainty, visualization, and text workflow.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional CSV file; synthetic data is the default",
    )
    parser.add_argument("--feature", default="X", help="Numeric feature column")
    parser.add_argument("--target", default="y", help="Numeric target column")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--samples", type=int, default=300, help="Synthetic sample count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = AnalysisConfig(
        sample_size=args.samples,
        seed=args.seed,
        confidence_level=args.confidence,
        bootstrap_resamples=args.bootstrap_resamples,
    )

    try:
        if args.input is None:
            data = generate_synthetic_data(config)
            source = "synthetic"
        else:
            data = pd.read_csv(args.input)
            source = str(args.input.resolve())

        result = run_analysis(
            data,
            config,
            feature_name=args.feature,
            target_name=args.target,
            sentence=args.sentence,
            source=source,
        )
        paths = save_artifacts(result, args.output_dir)
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))

    holdout = result.metrics["holdout"]
    interval = result.metrics["prediction_interval"]
    print(f"Analysis complete: {args.output_dir.resolve()}")
    print(
        f"Holdout RMSE={holdout['rmse']:.4f}, R2={holdout['r2']:.4f}, "
        f"conformal coverage={interval['empirical_coverage']:.1%}"
    )
    print(f"Report: {paths['report'].resolve()}")
    return 0


def entrypoint() -> None:
    raise SystemExit(main())
