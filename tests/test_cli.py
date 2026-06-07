from __future__ import annotations

from pathlib import Path

from multidisciplinary_analysis.cli import main


def test_cli_creates_outputs(tmp_path: Path, capsys: object) -> None:
    exit_code = main(
        [
            "--samples",
            "100",
            "--bootstrap-resamples",
            "200",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "analysis.png").exists()
    output = capsys.readouterr().out
    assert "Analysis complete" in output
