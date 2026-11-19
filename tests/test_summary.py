"""`reserve summary`: the headline table comes straight from a results file (M46)."""

import json

import pytest

from reserve import cli, summary


def _cov(x):
    return {"coverage": x, "ci_low": x - 0.05, "ci_high": x + 0.05}


RESULT = {
    "run_id": "20261119T000000Z_freddie_test_test_abcdef12",
    "config_hash": "abcdef1234567890",
    "role": "sealed test",
    "synthetic": True,
    "n_origins": 24,
    "origins": ["2013-03-01", "2018-12-01"],
    "sealed": {"run_number": 1},
    "provenance": {"git": {"commit": "0123456789", "dirty": False}},
    "scores": {
        "headline_horizon": 12,
        "contestants": {
            "B_age_curve": {
                "mean_quantile_loss": 0.002,
                "coverage_50": _cov(0.4),
                "coverage_90": _cov(0.8),
            },
            "D_tsfm_chronos_bolt_tiny": {
                "mean_quantile_loss": 0.003,
                "coverage_50": _cov(0.5),
                "coverage_90": _cov(0.7),
            },
        },
        "comparison": {
            "reference": "B_age_curve",
            "n_comparisons": 1,
            "comparisons": [
                {
                    "contestant": "D_tsfm_chronos_bolt_tiny",
                    "relative_error": 1.25,
                    "dm": {"p_value": 0.04},
                    "holm_p": 0.04,
                }
            ],
        },
    },
}


def test_table_carries_ids_labels_and_caveats():
    md = summary.headline_markdown(RESULT)
    assert md.startswith("**SYNTHETIC DATA")
    assert "abcdef123456" in md and "Sealed run number 1" in md
    assert "| B_age_curve (reference) | 0.002 | 1 |" in md
    assert "| D_tsfm_chronos_bolt_tiny | 0.003 | 1.25 | 0.04 | 0.04 |" in md
    assert "own 80% interval" in md and "1 comparisons, Holm-adjusted" in md


def test_experiments_are_refused_and_cli_reads_the_file(tmp_path, capsys):
    with pytest.raises(ValueError, match="standard runs"):
        summary.headline_markdown({"run_id": "x"})
    (tmp_path / "results.json").write_text(json.dumps(RESULT))
    assert cli.main(["summary", str(tmp_path)]) == 0
    assert "Reference (best classical" in capsys.readouterr().out
