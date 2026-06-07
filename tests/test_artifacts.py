from __future__ import annotations

import json
from pathlib import Path

from multidisciplinary_analysis.analysis import (
    AnalysisConfig,
    generate_synthetic_data,
    run_analysis,
)
from multidisciplinary_analysis.artifacts import save_artifacts


def test_artifact_contract(tmp_path: Path) -> None:
    config = AnalysisConfig(
        sample_size=120,
        seed=9,
        bootstrap_resamples=200,
        cv_repeats=1,
    )
    result = run_analysis(generate_synthetic_data(config), config)

    paths = save_artifacts(result, tmp_path)

    assert set(paths) == {"dataset", "predictions", "metrics", "run", "plot", "report"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    run = json.loads(paths["run"].read_text(encoding="utf-8"))
    assert metrics["dataset"]["rows"] == 120
    assert run["configuration"]["seed"] == 9
    assert paths["plot"].read_bytes().startswith(b"\x89PNG")
