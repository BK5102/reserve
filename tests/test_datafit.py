"""Data-fit checks on synthetic data and hand-made curves."""

import importlib.util
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl
import pytest

from reserve import datafit
from reserve.data import loans, synthetic

ROOT = Path(__file__).parents[1]
FIX = ROOT / "tests" / "fixtures" / "freddie"

CFG = synthetic.SimConfig(
    first_vintage="2005Q1",
    n_vintages=64,
    loans_per_vintage=150,
    data_end="2021-12",
    forbearance_start="2020-04",
    forbearance_end="2021-06",
    seed=2,
)


@pytest.fixture(scope="module")
def report():
    sim = synthetic.simulate(CFG)
    perf = sim.perf.with_columns(loans.parse_delinquency(pl.col("dlq_status")))
    return datafit.check(sim.orig, perf, "synthetic:test", macro_observations=sim.macro)


def test_report_is_labelled_and_complete(report):
    assert report["synthetic"] is True
    for key in (
        "latest",
        "loans_per_quarter",
        "events_at_check_age",
        "noise_floor",
        "recession",
        "covid",
        "reconciliation",
        "power",
        "gate",
        "macro",
    ):
        assert key in report
    assert report["loans_per_quarter"]["n_vintages"] == 64
    assert report["latest"]["last_performance_month"] == "2021-12-01"
    assert "SYNTHETIC DATA" in datafit.to_markdown(report)
    json.dumps(report, default=str)  # serializable


def test_recession_and_covid_checks_find_their_windows(report):
    assert report["recession"]["vintages_observed_in_window"] > 0
    assert report["recession"]["events_in_window"] > 0
    assert report["covid"]["events_in_window"] > 0
    assert report["covid"]["share_flagged_forbearance"] > 0


def test_power_arithmetic(report):
    p = report["power"]
    assert p["origins_with_targets"] == 24  # quarterly 2013-03 to 2018-12
    assert p["independent_windows"] == (69 + 12) // 12
    z = NormalDist().inv_cdf(1 - 0.05 / 6) + NormalDist().inv_cdf(0.8)
    assert p["mde_sd_units_all_origins"] == pytest.approx(z / np.sqrt(24))
    assert p["mde_sd_units_independent_windows"] == pytest.approx(z / np.sqrt(6))


def test_gate_logic(report):
    g = report["gate"]
    assert g["passes"] == (g["usable_vintages"] >= 60 and g["median_events_by_age"] >= 20)
    sim = synthetic.simulate(CFG)
    perf = sim.perf.with_columns(loans.parse_delinquency(pl.col("dlq_status")))
    lax = datafit.check(sim.orig, perf, "synthetic:lax", datafit.FitSettings(min_events=0))
    assert lax["gate"]["passes"] is True


def test_noise_floor_by_hand():
    """Identical vintages: the class-average forecast is exact, so the ratio is zero."""
    rows = []
    for v in ["2001Q1", "2001Q2", "2001Q3", "2001Q4", "2002Q1", "2002Q2"]:
        rows += [
            {"vintage": v, "age": 12, "n_orig": 1000, "cum_rate": 0.01},
            {"vintage": v, "age": 24, "n_orig": 1000, "cum_rate": 0.03},
        ]
    nf = datafit.noise_floor(pl.DataFrame(rows), datafit.FitSettings())
    assert nf["n_vintages"] == 2
    assert nf["class_average_rmse"] == pytest.approx(0.0)
    assert nf["binomial_noise_sd"] == pytest.approx(np.sqrt(0.02 * (1 - 0.02 - 0.01) / 1000))
    assert nf["ratio"] == pytest.approx(0.0)


def test_macro_coverage(report):
    un = report["macro"]["UNRATE"]
    assert un["n_vintages"] > 1 and un["missing_values"] == 0


def test_loads_freddie_layout_files(tmp_path):
    for name in ("sample_orig_2005.txt", "sample_perf_2005.txt"):
        (tmp_path / name).write_text((FIX / name).read_text())
    orig, perf, files = datafit.load_freddie(tmp_path)
    assert files[0]["orig_layout"] == "current" and files[0]["perf_rows"] == 22
    rep = datafit.check(orig, perf, f"freddie:{tmp_path}", file_report=files)
    assert rep["synthetic"] is False
    assert rep["reconciliation"]["loans_with_event"] == 1


def test_script_runs_on_a_synthetic_config(tmp_path, capsys):
    cfg = tmp_path / "fit.yaml"
    cfg.write_text(
        "synthetic:\n  first_vintage: '2008Q1'\n  n_vintages: 8\n  loans_per_vintage: 40\n"
        "  data_end: '2012-12'\n"
    )
    spec = importlib.util.spec_from_file_location("cdf", ROOT / "scripts" / "check_data_fit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "fit.json"
    assert mod.main(["synthetic", str(cfg), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["synthetic"] is True
    assert "SYNTHETIC DATA" in capsys.readouterr().out
