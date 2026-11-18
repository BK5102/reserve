"""The drift monitor, tested on synthetic drift with a known start (methodology M36)."""

from datetime import date

import numpy as np
import polars as pl
import pytest
from scipy import stats as sps

from reserve import months
from reserve.data import macro, synthetic
from reserve.evaluation.harness import qcol
from reserve.models.base import QUANTILES
from reserve.monitoring import drift

DRIFT_START = 24  # origin index where outcomes start to shift


def _preds(shift_sd: float, n_origins: int = 40, n_vintages: int = 30, seed: int = 0):
    """Calibrated N(0.02, 0.002) forecasts; outcomes shift by `shift_sd` from DRIFT_START."""
    rng = np.random.default_rng(seed)
    mu, sd = 0.02, 0.002
    z = sps.norm.ppf(QUANTILES)
    rows = []
    for i in range(n_origins):
        origin = months.add(date(2005, 3, 1), 3 * i)
        for v in range(n_vintages):
            y = rng.normal(mu + (shift_sd * sd if i >= DRIFT_START else 0.0), sd)
            rows.append(
                {
                    "origin": origin,
                    "vintage": f"v{v}",
                    "k": 12,
                    "y": y,
                    "contestant": "X",
                    **{qcol(q): mu + sd * zq for q, zq in zip(QUANTILES, z, strict=True)},
                }
            )
    return pl.DataFrame(rows)


REF_END = months.add(date(2005, 3, 1), 3 * 11)  # first 12 origins are the reference


def test_no_drift_stays_green_almost_always():
    table = drift.monitor_predictions(_preds(0.0), "X", REF_END, QUANTILES, 12)
    assert table.height == 40 - 12
    assert (table["status"] == drift.RED).sum() == 0
    assert (table["status"] == drift.GREEN).mean() >= 0.8


def test_a_shift_in_outcomes_turns_red_within_the_window():
    table = drift.monitor_predictions(_preds(2.0), "X", REF_END, QUANTILES, 12, window=4)
    before = table.filter(pl.col("origin") < months.add(date(2005, 3, 1), 3 * DRIFT_START))
    after = table.filter(pl.col("origin") >= months.add(date(2005, 3, 1), 3 * (DRIFT_START + 3)))
    assert (before["status"] != drift.RED).all()
    assert (after["coverage_status"] == drift.RED).all()
    assert (after["loss_status"] == drift.RED).all()
    first_red = table.filter(pl.col("status") == drift.RED)["origin"].min()
    assert first_red <= months.add(date(2005, 3, 1), 3 * (DRIFT_START + 3))
    row = table.row(0, named=True)
    assert row["known_at"] == months.add(row["origin"], 12)


def test_checks_by_hand():
    y = np.zeros(100)
    inside = drift.coverage_check(y, np.full(100, -1.0), np.full(100, 1.0))
    assert inside.status == drift.GREEN and inside.value == 1.0
    lo = np.where(np.arange(100) < 80, -1.0, 0.5)  # 80 of 100 inside
    assert drift.coverage_check(y, lo, np.full(100, 1.0)).status == drift.RED
    lo = np.where(np.arange(100) < 86, -1.0, 0.5)  # 86 of 100: within 5 points of 90%
    assert drift.coverage_check(y, lo, np.full(100, 1.0)).status == drift.GREEN
    # 16 of 20 inside: more than 5 points short, but too few items to rule out luck.
    lo = np.where(np.arange(20) < 16, -1.0, 0.5)
    assert drift.coverage_check(y[:20], lo, np.full(20, 1.0)).status == drift.AMBER
    assert drift.loss_ratio_check(1.2, 1.0).status == drift.GREEN
    assert drift.loss_ratio_check(1.7, 1.0).status == drift.AMBER
    assert drift.loss_ratio_check(2.5, 1.0).status == drift.RED
    assert drift.worst([drift.GREEN, drift.RED, drift.AMBER]) == drift.RED


def test_macro_range_flags_the_synthetic_recession():
    sim = synthetic.simulate(
        synthetic.SimConfig(
            first_vintage="2003Q1", n_vintages=8, loans_per_vintage=20, data_end="2011-12", seed=3
        )
    )
    ms = [months.add(date(2003, 1, 1), i) for i in range(108)]
    feats = macro.macro_features(sim.macro, ms, mode="asof")
    train_until = date(2006, 12, 1)
    calm = drift.macro_range_check(feats, train_until, date(2006, 6, 1), ["unemployment"])
    peak = drift.macro_range_check(feats, train_until, date(2009, 10, 1), ["unemployment"])
    assert calm[0].status != drift.RED
    assert peak[0].status == drift.RED and "training range" in peak[0].detail
    with pytest.raises(ValueError):
        drift.macro_range_check(feats, train_until, date(2030, 1, 1), ["unemployment"])


def test_monitor_cli(tmp_path, capsys):
    import json

    from reserve import cli

    run = tmp_path / "run"
    run.mkdir()
    _preds(2.0).write_parquet(run / "predictions.parquet")
    cfg = {"evaluation": {"quantiles": list(QUANTILES), "horizon": 12}}
    (run / "results.json").write_text(
        json.dumps({"run_id": "r", "synthetic": True, "config": cfg}), encoding="utf-8"
    )
    args = ["monitor", str(run), "--contestant", "X", "--reference-end", "2007-12"]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC DATA" in out and "red" in out
    assert (
        cli.main(["monitor", str(run), "--contestant", "nobody", "--reference-end", "2007-12"]) == 1
    )
