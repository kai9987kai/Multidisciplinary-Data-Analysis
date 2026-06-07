"""Reproducible multidisciplinary data analysis."""

from multidisciplinary_analysis.analysis import (
    AnalysisConfig,
    AnalysisResult,
    generate_synthetic_data,
    run_analysis,
    summarize_text,
)

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "generate_synthetic_data",
    "run_analysis",
    "summarize_text",
]

__version__ = "2.0.0"
