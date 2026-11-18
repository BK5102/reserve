"""Origins, contestants A to C, the TSFM interface, the harness, and scoring.

Includes the past-only leak tests: no prediction at an origin may change when anything after
the origin (later curve cells, or macro values published later) is corrupted.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from reserve import months
from reserve.data import loans, macro, synthetic, vintages
from reserve.evaluation import harness, origins, scoring
from reserve.models import baselines, hazard
from reserve.models.base import QUANTILES, tidy
from reserve.models.tsfm import TSFMContestant, chronos_bolt, make_forecaster

CFG = synthetic.SimConfig(
    first_vintage="2003Q1", n_vintages=36, loans_per_vintage=400, data_end="2012-12", seed=4
)


@pytest.fixture(scope="module")
def sim():
    return synthetic.simulate(CFG)


@pytest.fixture(scope="module")
def curves(sim):
    perf = sim.perf.with_columns(loans.parse_delinquency(pl.col("dlq_status")))
    return vintages.build(sim.orig, perf).curves


@pytest.fixture(scope="module")
def feats(sim, curves):
    months_needed = sorted(curves["period"].unique().to_list())
    return macro.macro_features(sim.macro, months_needed, mode="asof")


def contestants(feats):
    return [
        baselines.LastIncrement(),
        baselines.AgeCurve(),
        hazard.HazardRegression(feats, draws=100),
        TSFMContestant(make_forecaster("fake")),
    ]


# ----- origins -------------------------------------------------------------------------


def test_month_axis_round_trip():
    for d in (date(1999, 1, 1), date(2008, 2, 1), date(2026, 12, 1)):
        assert origins.from_month_time(origins.month_time(d)) == d


def test_monthly_origins_are_cleanroom_splits():
    os_ = origins.monthly_origins(date(2010, 3, 1), date(2010, 12, 1), 3, 12)
    assert [o.month for o in os_] == [date(2010, m, 1) for m in (3, 6, 9, 12)]
    first = os_[0].split
    assert first.train.start is None  # expanding window
    assert origins.from_month_time(first.train.end) == date(2010, 4, 1)
    assert origins.from_month_time(first.test.start) == date(2010, 4, 1)
    assert origins.from_month_time(first.test.end) == date(2011, 4, 1)


def test_origin_apply_cuts_at_the_month(curves):
    o = origins.monthly_origins(date(2008, 6, 1), date(2008, 6, 1), 3, 12)[0]
    hist, fut = o.apply(origins.with_month_time(curves))
    assert hist["period"].max() == date(2008, 6, 1)
    assert fut["period"].min() == date(2008, 7, 1) and fut["period"].max() == date(2009, 6, 1)


# ----- clean-up and contestants by hand -------------------------------------------------


def test_tidy_enforces_coherence():
    raw = np.array([[0.02, 0.01], [0.005, 0.03], [2.0, 0.04]])
    out = tidy(raw, c0=0.01)
    assert (out >= 0.01).all() and (out <= 1).all()
    assert (np.diff(out, axis=0) >= 0).all() and (np.diff(out, axis=1) >= 0).all()


def _toy_history(increments_by_vintage: dict[str, list[float]], n: int = 1000) -> pl.DataFrame:
    rows = []
    for v, incs in increments_by_vintage.items():
        qend = months.index(months.quarter_end(v))
        cum = 0.0
        for a, inc in enumerate(incs, start=1):
            cum += inc
            rows.append(
                {
                    "vintage": v,
                    "age": a,
                    "period": months.from_index(qend + a),
                    "n_orig": n,
                    "at_risk": n,
                    "new_events": round(inc * n),
                    "exits_prepaid": 0,
                    "exits_other": 0,
                    "exits_unexplained": 0,
                    "cum_rate": cum,
                    "increment": inc,
                }
            )
    return pl.DataFrame(rows)


def test_last_increment_by_hand():
    hist = _toy_history({"2010Q1": [0.0, 0.001, 0.002, 0.003]})
    origin = hist["period"].max()
    out = baselines.LastIncrement(window=3).forecast(hist, origin, ["2010Q1"], 2, (0.5,))
    # d = 0.002, n = 1000: medians of Poisson(2) and Poisson(4) are 2 and 4 events.
    assert out["2010Q1"][:, 0] == pytest.approx([0.006 + 0.002, 0.006 + 0.004])


def test_age_curve_by_hand():
    older = {f"2009Q{q}": [0.001] * 12 for q in (1, 2, 3, 4)}
    hist = _toy_history({**older, "2010Q4": [0.001] * 3})
    origin = date(2011, 3, 1)  # 2010Q4 is at age 3; older vintages have ages up to 12 by then
    hist = hist.filter(pl.col("period") <= origin)
    out = baselines.AgeCurve(donors=8).forecast(hist, origin, ["2010Q4"], 4, QUANTILES)
    point = out["2010Q4"]
    assert point[:, 3] == pytest.approx(0.003 + 0.001 * np.arange(1, 5))
    assert np.allclose(point, point[:, [3]])  # identical donors: zero spread


def test_logistic_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    x = np.column_stack([np.ones(4000), rng.normal(size=(4000, 2))])
    beta = np.array([-4.0, 0.8, -0.5])
    trials = np.full(4000, 500)
    events = rng.binomial(trials, 1 / (1 + np.exp(-(x @ beta))))
    got, cov = hazard.fit_logistic(x, events, trials)
    assert got == pytest.approx(beta, abs=4 * np.sqrt(np.diag(cov)).max() + 0.02)


def test_hazard_regression_finds_the_macro_effect(curves, feats):
    c = hazard.HazardRegression(feats, features=("unemployment",), draws=50)
    c.fit(curves)
    # Coefficient on standardized unemployment (last column) must be positive and sizeable.
    assert c.last_fit["beta"][-1] > 0.2


def test_tsfm_contestant_shapes_and_coherence(curves):
    origin = date(2009, 6, 1)
    hist = curves.filter(pl.col("period") <= origin)
    vs = harness.eligible_vintages(hist, origin, 6)
    out = TSFMContestant(make_forecaster("fake")).forecast(hist, origin, vs, 12, QUANTILES)
    assert set(out) == set(vs)
    for arr in out.values():
        assert arr.shape == (12, len(QUANTILES))
        assert (np.diff(arr, axis=0) >= 0).all() and (np.diff(arr, axis=1) >= 0).all()


def test_tsfm_shape_mismatch_is_caught(curves):
    class Bad:
        name, version = "bad", "0"

        def forecast(self, contexts, horizon, quantiles):
            return np.zeros((len(contexts), horizon))

    origin = date(2009, 6, 1)
    hist = curves.filter(pl.col("period") <= origin)
    with pytest.raises(ValueError):
        TSFMContestant(Bad()).forecast(hist, origin, ["2005Q1"], 12, QUANTILES)


def test_chronos_bolt_stub_is_pinned_and_lazy():
    assert chronos_bolt.REVISION == "a0e552de83495b5c28c14c71c374f3e33280b340"
    assert chronos_bolt.MODEL_ID == "amazon/chronos-bolt-tiny"
    try:
        import chronos  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="--extra tsfm"):
            make_forecaster("chronos_bolt_tiny")
    else:
        pytest.skip("chronos is installed; constructing would download weights")


# ----- harness and leak tests ----------------------------------------------------------


@pytest.fixture(scope="module")
def run_preds(curves, feats):
    os_ = origins.monthly_origins(date(2007, 3, 1), date(2010, 12, 1), 3, 12)
    return harness.run(curves, contestants(feats), os_, 12, QUANTILES)


def test_harness_output(run_preds):
    assert run_preds["contestant"].n_unique() == 4
    assert run_preds["k"].max() == 12
    one = run_preds.filter(pl.col("contestant") == "B_age_curve")
    assert one.select("origin", "vintage", "k").is_duplicated().sum() == 0
    obs = one.drop_nulls("y")
    assert (obs["target_period"] > obs["origin"]).all()


def test_predictions_ignore_corrupted_future_cells(curves, feats):
    """Past-only, every contestant: corrupt all cells after the origin; forecasts do not change."""
    os_ = origins.monthly_origins(date(2008, 9, 1), date(2008, 9, 1), 3, 12)
    t = os_[0].month
    poisoned = curves.with_columns(
        pl.when(pl.col("period") > t).then(pl.col(c) * 7 + 3).otherwise(pl.col(c)).alias(c)
        for c in ("cum_rate", "increment", "new_events", "at_risk", "exits_prepaid")
    )
    qs = [harness.qcol(q) for q in QUANTILES]
    a = harness.run(curves, contestants(feats), os_, 12, QUANTILES).select(
        "contestant", "vintage", "k", *qs
    )
    b = harness.run(poisoned, contestants(feats), os_, 12, QUANTILES).select(
        "contestant", "vintage", "k", *qs
    )
    assert a.equals(b)


def test_hazard_ignores_macro_published_after_the_origin(sim, curves):
    """Corrupt every macro value published after the origin month; C's forecast is unchanged."""
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
    qs = [harness.qcol(q) for q in QUANTILES]
    a = harness.run(curves, [hazard.HazardRegression(good, draws=50)], os_, 12, QUANTILES).select(
        qs
    )
    b = harness.run(curves, [hazard.HazardRegression(bad, draws=50)], os_, 12, QUANTILES).select(qs)
    assert a.equals(b)


# ----- scoring -------------------------------------------------------------------------


def test_scoring_by_hand():
    rows = []
    for name, shift in (("A_last_increment", 0.002), ("B_age_curve", 0.0), ("D_tsfm_fake", 0.001)):
        for i in range(6):
            rows.append(
                {
                    "origin": date(2010, 1 + i, 1),
                    "vintage": "2009Q1",
                    "k": 12,
                    "target_period": date(2011, 1 + i, 1),
                    "c0": 0.01,
                    "y": 0.02 + 0.0001 * i,
                    "contestant": name,
                    **{
                        harness.qcol(q): 0.02 + 0.0001 * i + shift + (q - 0.5) * 0.001
                        for q in QUANTILES
                    },
                }
            )
    df = pl.DataFrame(rows)
    res = scoring.score(df, QUANTILES, headline_horizon=12, step_months=1)
    assert res["comparison"]["reference"] == "B_age_curve"
    assert res["comparison"]["n_comparisons"] == 2
    b = res["contestants"]["B_age_curve"]
    assert b["coverage_90"]["coverage"] == 1.0 and b["median_abs_error"] == pytest.approx(0.0)
    a_cmp = next(
        c for c in res["comparison"]["comparisons"] if c["contestant"] == "A_last_increment"
    )
    assert a_cmp["relative_error"] > 1
    excluded = scoring.score(df, QUANTILES, 12, 1, exclude=[(date(2011, 1, 1), date(2011, 3, 1))])
    assert excluded["n_rows_headline"] == df.height // 2


def test_full_scoring_on_synthetic(run_preds):
    res = scoring.score(run_preds, QUANTILES, headline_horizon=12, step_months=3)
    assert set(res["contestants"]) == {
        "A_last_increment",
        "B_age_curve",
        "C_hazard_macro",
        "D_tsfm_fake",
    }
    assert res["comparison"]["reference"] in scoring.CLASSICAL
    for v in res["contestants"].values():
        assert 0 <= v["coverage_90"]["coverage"] <= 1
