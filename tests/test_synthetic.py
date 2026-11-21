"""Synthetic portfolio: exact Freddie Mac layout, and the pipeline recovers its known answers."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from reserve.data import loans, macro, synthetic, vintages

SMALL = synthetic.SimConfig(
    first_vintage="2005Q1", n_vintages=16, loans_per_vintage=250, data_end="2011-12", seed=3
)


@pytest.fixture(scope="module")
def sim():
    return synthetic.simulate(SMALL)


@pytest.fixture(scope="module")
def written(sim, tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    synthetic.write(sim, out)
    return out


@pytest.fixture(scope="module")
def built(written):
    orig = pl.concat(
        [loans.read_origination(o) for _, o, _ in loans.find_sample_files(written / "freddie")]
    )
    perf = pl.concat(
        [loans.read_performance(p) for _, _, p in loans.find_sample_files(written / "freddie")]
    )
    return vintages.build(orig, perf)


def test_files_use_the_exact_layout(written):
    files = loans.find_sample_files(written / "freddie")
    assert [y for y, _, _ in files] == list(range(2005, 2009))
    for _, o, p in files:
        assert loans.detect_layout(o, "orig")[0] == "current"
        assert loans.detect_layout(p, "perf")[0] == "current"
        loans.check_loan_ids(loans.read_origination(o))


def test_builder_recovers_every_event_age_exactly(sim, built):
    got = built.loans.select("loan_id", "event_age").sort("loan_id")
    want = sim.loan_events.select("loan_id", "event_age").sort("loan_id")
    assert got.equals(want)


def test_observed_curves_match_the_true_expected_curves(sim, built):
    """Across vintages, observed events by age 24 are within binomial noise of the truth."""
    at24 = built.curves.filter(pl.col("age") == 24).join(
        sim.truth.select("vintage", "age", "expected_cum_rate"), on=["vintage", "age"]
    )
    observed = at24["cum_events"].sum()
    expected = (at24["expected_cum_rate"] * at24["n_orig"]).sum()
    sd = np.sqrt(
        (at24["expected_cum_rate"] * (1 - at24["expected_cum_rate"]) * at24["n_orig"]).sum()
    )
    assert abs(observed - expected) < 4 * sd


def test_recession_raises_the_true_hazard():
    m = synthetic.months
    month_idx = np.array([m.index(m.parse("2006-06")), m.index(m.parse("2009-10"))])
    h, _ = synthetic.true_hazard(SMALL, 0.0, np.array([30, 30]), month_idx)
    season = np.exp(
        SMALL.season_amplitude * (np.cos(2 * np.pi * 5 / 12) - np.cos(2 * np.pi * 9 / 12))
    )
    assert h[1] / h[0] == pytest.approx(np.exp(SMALL.macro_beta * SMALL.recession_rise) / season)


def test_no_events_before_age_three(sim):
    assert sim.loan_events["event_age"].drop_nulls().min() >= 3
    assert sim.truth.filter(pl.col("age") < 3)["hazard"].max() == 0


def test_same_seed_same_portfolio():
    a = synthetic.simulate(
        synthetic.SimConfig(n_vintages=3, loans_per_vintage=20, data_end="2003-12")
    )
    b = synthetic.simulate(
        synthetic.SimConfig(n_vintages=3, loans_per_vintage=20, data_end="2003-12")
    )
    assert a.perf.equals(b.perf) and a.orig.equals(b.orig)


def test_macro_tables_work_with_the_as_of_loader(sim):
    feats = macro.macro_features(sim.macro, [synthetic.months.parse("2009-10")], mode="asof").row(
        0, named=True
    )
    # As of end of October 2009 the newest unemployment value is September's first release.
    assert 9.0 < feats["unemployment"] < 10.5
    assert feats["unemployment_change_12m"] > 2
    assert feats["hpi_growth_12m"] < 0


def test_forbearance_window_flags_events():
    cfg = synthetic.SimConfig(
        first_vintage="2018Q1",
        n_vintages=4,
        loans_per_vintage=400,
        data_end="2021-12",
        recession_start="2030-01",
        recession_peak="2030-02",
        recession_end="2030-03",
        forbearance_start="2020-04",
        forbearance_end="2021-06",
        seed=1,
    )
    s = synthetic.simulate(cfg)
    data = vintages.build(
        s.orig, s.perf.with_columns(loans.parse_delinquency(pl.col("dlq_status")))
    )
    flagged = data.loans.filter(pl.col("event_in_forbearance"))
    assert flagged.height > 0
    assert flagged["event_period"].min() >= synthetic.months.parse("2020-04")


def test_per_year_summaries_give_the_same_tables(written, built):
    """Summarizing each year's file on its own, then stacking, changes nothing (M47)."""
    files = loans.find_sample_files(written / "freddie")
    orig = pl.concat([loans.read_origination(o) for _, o, _ in files])
    summary = pl.concat(
        [vintages.summarize_performance(loans.read_performance(p)) for _, _, p in files]
    )
    assert vintages.is_summary(summary) and summary.height < sum(
        loans.read_performance(p).height for _, _, p in files
    )
    stacked = vintages.build(orig, summary)
    assert stacked.cutoff == built.cutoff
    assert stacked.loans.equals(built.loans)
    assert stacked.curves.equals(built.curves)


def test_a_loan_in_two_summaries_is_refused(written):
    _, o, p = loans.find_sample_files(written / "freddie")[0]
    one = vintages.summarize_performance(loans.read_performance(p))
    with pytest.raises(vintages.ReconciliationError, match="more than one performance file"):
        vintages.build(loans.read_origination(o), pl.concat([one, one]))


def test_a_summary_cannot_be_cut_earlier(written, built):
    files = loans.find_sample_files(written / "freddie")
    orig = pl.concat([loans.read_origination(o) for _, o, _ in files])
    summary = pl.concat(
        [vintages.summarize_performance(loans.read_performance(p)) for _, _, p in files]
    )
    with pytest.raises(ValueError, match="cannot be cut earlier"):
        vintages.build(orig, summary, cutoff=date(2010, 1, 1))
