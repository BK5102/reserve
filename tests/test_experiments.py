"""Thin-history experiment and HYPOTHETICAL macro scenarios (methodology M34, M35).

Includes past-only leak tests for both options: corrupting anything after the origin
(curve cells, or macro values published later) must not change any forecast.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from reserve.data import loans, macro, synthetic, vintages
from reserve.evaluation import experiments, harness, origins
from reserve.models import baselines, hazard
from reserve.models.base import QUANTILES
from reserve.models.tsfm import TSFMContestant, make_forecaster
from reserve.stress import scenarios as sc

CFG = synthetic.SimConfig(
    first_vintage="2003Q1", n_vintages=36, loans_per_vintage=300, data_end="2012-12", seed=4
)
QS = [harness.qcol(q) for q in QUANTILES]


@pytest.fixture(scope="module")
def sim():
    return synthetic.simulate(CFG)


@pytest.fixture(scope="module")
def curves(sim):
    perf = sim.perf.with_columns(loans.parse_delinquency(pl.col("dlq_status")))
    return vintages.build(sim.orig, perf).curves


@pytest.fixture(scope="module")
def feats(sim, curves):
    return macro.macro_features(sim.macro, sorted(curves["period"].unique().to_list()), "asof")


@pytest.fixture(scope="module")
def origins_():
    return origins.monthly_origins(date(2008, 3, 1), date(2010, 6, 1), 3, 12)


def base_contestants(feats, scenario=None):
    return [
        baselines.LastIncrement(),
        baselines.AgeCurve(),
        hazard.HazardRegression(feats, draws=50, scenario=scenario),
        TSFMContestant(make_forecaster("fake")),
    ]


class Spy:
    """Records the history it is given and forecasts flat."""

    name = "spy"

    def __init__(self):
        self.seen: list[tuple[date, pl.DataFrame]] = []

    def describe(self):
        return {"name": self.name}

    def forecast(self, history, origin, vintages_, horizon, quantiles):
        self.seen.append((origin, history))
        state = history.filter(pl.col("period") == origin)
        c0 = dict(zip(state["vintage"], state["cum_rate"], strict=True))
        return {v: np.full((horizon, len(quantiles)), c0[v]) for v in vintages_}


# ----- thin history ---------------------------------------------------------------------


@pytest.mark.parametrize("months", [6, 12, 24])
def test_thin_history_shows_only_a_portfolio_that_young(curves, origins_, months):
    spy = Spy()
    preds = harness.run(curves, [spy], origins_, 12, QUANTILES, 6, history_months=months)
    for origin, hist in spy.seen:
        at = hist.filter(pl.col("period") == origin)
        assert at["age"].max() <= months  # no vintage older than the portfolio
        assert hist["period"].min() > date(origin.year - 3, origin.month, 1)
        assert (hist["period"] <= origin).all()
        # Every vintage seen has its whole curve inside the last `months` months.
        first_month = hist.group_by("vintage").agg(pl.col("age").min())["age"]
        assert (first_month == 1).all()
    assert preds["a0"].max() <= months


def test_thin_history_ignores_corrupted_future_cells(curves, feats):
    os_ = origins.monthly_origins(date(2008, 9, 1), date(2008, 9, 1), 3, 12)
    t = os_[0].month
    poisoned = curves.with_columns(
        pl.when(pl.col("period") > t).then(pl.col(c) * 7 + 3).otherwise(pl.col(c)).alias(c)
        for c in ("cum_rate", "increment", "new_events", "at_risk", "exits_prepaid")
    )
    kw = {"history_months": 12, "max_target_age": 12}
    a = harness.run(curves, base_contestants(feats), os_, 12, QUANTILES, 6, **kw)
    b = harness.run(poisoned, base_contestants(feats), os_, 12, QUANTILES, 6, **kw)
    cols = ["contestant", "vintage", "k", *QS]
    assert a.select(cols).equals(b.select(cols))


@pytest.fixture(scope="module")
def thin(curves, feats, origins_):
    cs = base_contestants(feats)
    return experiments.run_thin_history(curves, lambda _k: cs, origins_, 12, QUANTILES, 3, 6)


def test_thin_history_targets_are_the_same_at_every_level(thin):
    preds, payload = thin
    assert payload["levels"] == ["6", "12", "24", "full"]
    keys = {
        lvl: set(g.select("origin", "vintage").unique().iter_rows())
        for (lvl,), g in preds.group_by("level")
    }
    assert len({frozenset(v) for v in keys.values()}) == 1
    assert preds["a0"].max() == 6 and preds["a0"].min() == 6


def test_models_that_use_only_their_own_curve_do_not_move(thin):
    _, payload = thin
    curve = payload["learning_curve"]
    for name in ("A_last_increment", "D_tsfm_fake"):
        assert len({round(v, 12) for v in curve[name].values()}) == 1
    # B and C borrow from other vintages, so cutting history changes them.
    assert len({round(v, 12) for v in curve["B_age_curve"].values()}) > 1


def test_thin_history_corrects_across_every_level(thin):
    _, payload = thin
    joint = payload["holm_all_levels"]
    assert payload["n_comparisons_all_levels"] == len(joint) > 3
    assert payload["n_comparisons_tried_all_levels"] == 3 * 4
    for row in joint:
        assert row["holm_p_all_levels"] >= row["p"]


def test_thin_history_levels_below_min_history_are_refused(curves, feats, origins_):
    with pytest.raises(ValueError, match="min_history"):
        experiments.run_thin_history(
            curves, lambda _k: [], origins_, 12, QUANTILES, 3, 6, months=(3, 12)
        )


# ----- HYPOTHETICAL scenarios -----------------------------------------------------------


def test_scenario_paths_by_hand():
    s = sc.Scenario("x", unemployment_rise=3.0, hpi_growth_fall=0.12)
    d = s.deltas(macro.FEATURES, 15)
    assert d.shape == (15, 4)
    assert d[0] == pytest.approx([0.25, 0.25, 0.0, -0.01])  # month 1 of a 12-month ramp
    assert d[11] == pytest.approx([3.0, 3.0, 0.0, -0.12])
    assert d[14] == pytest.approx(d[11])  # flat after month 12
    assert not sc.get("baseline").deltas(macro.FEATURES, 12).any()
    assert all(
        s.hypothetical and s.describe()["label"] == "HYPOTHETICAL" for s in sc.SCENARIOS.values()
    )
    with pytest.raises(ValueError):
        s.deltas(("gdp",), 12)
    with pytest.raises(ValueError):
        sc.get("boom")


def test_baseline_scenario_equals_the_flat_macro_path(curves, feats, origins_):
    plain = harness.run(curves, [hazard.HazardRegression(feats, draws=50)], origins_, 12, QUANTILES)
    base = harness.run(
        curves,
        [hazard.HazardRegression(feats, draws=50, scenario=sc.get("baseline"))],
        origins_,
        12,
        QUANTILES,
    )
    assert plain.select(QS).equals(base.select(QS))


def test_rising_unemployment_raises_the_hazard_forecast(curves, feats, origins_):
    """The simulator's hazard rises with unemployment, so C's median must rise too."""
    jobs = sc.Scenario("jobs_only", unemployment_rise=4.0, hpi_growth_fall=0.0)
    run = {
        name: harness.run(
            curves,
            [hazard.HazardRegression(feats, features=("unemployment",), draws=50, scenario=s)],
            origins_,
            12,
            QUANTILES,
        ).filter(pl.col("k") == 12)
        for name, s in (("base", sc.get("baseline")), ("jobs", jobs))
    }
    assert (run["jobs"]["q0.5"] >= run["base"]["q0.5"]).all()
    assert run["jobs"]["q0.5"].mean() > run["base"]["q0.5"].mean()


def test_scenario_forecasts_ignore_macro_published_after_the_origin(sim, curves):
    t = date(2008, 9, 1)
    cut = macro.end_of_month(t)
    bad_obs = {
        sid: o.with_columns(
            pl.when(pl.col("realtime_start") > cut)
            .then(pl.col("value") * 50 + 9)
            .otherwise(pl.col("value"))
            .alias("value")
        )
        for sid, o in sim.macro.items()
    }
    ms = sorted(curves.filter(pl.col("period") <= t)["period"].unique().to_list())
    good = macro.macro_features(sim.macro, ms, mode="asof")
    bad = macro.macro_features(bad_obs, ms, mode="asof")
    os_ = origins.monthly_origins(t, t, 3, 12)
    severe = sc.get("severe")
    a = harness.run(
        curves, [hazard.HazardRegression(good, draws=50, scenario=severe)], os_, 12, QUANTILES
    )
    b = harness.run(
        curves, [hazard.HazardRegression(bad, draws=50, scenario=severe)], os_, 12, QUANTILES
    )
    assert a.select(QS).equals(b.select(QS))


def test_run_scenarios_labels_everything_and_finds_the_macro_blind(curves, feats, origins_):
    preds, payload = experiments.run_scenarios(
        curves,
        lambda name: base_contestants(feats, sc.get(name)),
        origins_,
        12,
        QUANTILES,
        6,
        names=("adverse", "severe"),
    )
    assert payload["hypothetical"] is True and payload["label"] == "HYPOTHETICAL"
    assert "not forecasts of the economy" in payload["disclaimer"]
    assert list(payload["scenarios"]) == ["baseline", "adverse", "severe"]  # baseline added
    assert set(preds["label"].unique()) == {"HYPOTHETICAL"}
    assert payload["responds_to_scenario"] == {
        "A_last_increment": False,
        "B_age_curve": False,
        "C_hazard_macro": True,
        "D_tsfm_fake": False,
    }
    s = payload["summary"]
    assert (
        s["severe"]["C_hazard_macro"]["mean_increase_q0.5"]
        > s["baseline"]["C_hazard_macro"]["mean_increase_q0.5"]
    )
    assert "loss" not in str(payload["summary"])  # never scored against outcomes
