"""Vintage builder: values by hand, an independent brute-force count, and past-only cutoffs."""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from reserve import months
from reserve.data import loans, vintages

FIX = Path(__file__).parent / "fixtures" / "freddie"


@pytest.fixture(scope="module")
def fixture_data():
    orig = loans.read_origination(FIX / "sample_orig_2005.txt")
    perf = loans.read_performance(FIX / "sample_perf_2005.txt")
    return orig, perf


def test_loan_outcomes_by_hand(fixture_data):
    orig, perf = fixture_data
    out = vintages.loan_outcomes(orig, perf).sort("loan_id")
    rows = {r["loan_id"]: r for r in out.iter_rows(named=True)}
    one, two, three = rows["F05Q10000001"], rows["F05Q20000002"], rows["F05Q20000003"]
    assert one["exit_reason"] == "prepaid" and one["event_age"] is None
    assert one["last_obs_age"] == 5  # 2005-08 is five months after March 2005
    assert two["event_period"] == date(2005, 11, 1)  # first month at status 03
    assert two["event_age"] == 5 and two["exit_reason"] == "default"
    assert three["event_age"] is None and three["exit_reason"] == "active"


def test_curves_by_hand(fixture_data):
    orig, perf = fixture_data
    data = vintages.build(orig, perf)
    assert data.cutoff == date(2006, 3, 1)
    q2 = data.curves.filter(pl.col("vintage") == "2005Q2").sort("age")
    assert q2["age"].to_list() == list(range(1, 10))  # 2005-07 to 2006-03
    assert q2["period"][0] == date(2005, 7, 1)
    assert q2["n_orig"].unique().to_list() == [2]
    assert q2["at_risk"].to_list() == [2, 2, 2, 2, 2, 1, 0, 0, 0]
    assert q2["new_events"].to_list() == [0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert q2["cum_rate"].to_list() == [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5]
    # Loan 3 vanishes after 2005-12 without a zero balance code.
    assert q2["exits_unexplained"].to_list() == [0, 0, 0, 0, 0, 1, 0, 0, 0]
    q1 = data.curves.filter(pl.col("vintage") == "2005Q1").sort("age")
    assert q1.height == 12 and q1["cum_rate"].max() == 0
    assert q1["exits_prepaid"].to_list()[4] == 1


def _random_portfolio(seed: int, n_loans: int = 400):
    """Random loans in four vintages with random status paths; returns (orig, perf) frames."""
    rng = np.random.default_rng(seed)
    vint = ["2010Q1", "2010Q2", "2010Q3", "2010Q4"]
    orig_rows, perf_rows = [], []
    for i in range(n_loans):
        v = vint[i % 4]
        lid = f"F10Q{v[-1]}{i:07d}"
        orig_rows.append({"loan_id": lid, "vintage": v})
        start = months.index(months.quarter_end(v)) - int(rng.integers(0, 3))
        length = int(rng.integers(1, 30))
        dlq = 0
        for k in range(length):
            dlq = dlq + 1 if rng.random() < 0.1 else (0 if rng.random() < 0.5 else dlq)
            perf_rows.append(
                {
                    "loan_id": lid,
                    "period": months.from_index(start + k),
                    "dlq_months": dlq,
                    "reo_acquired": False,
                    "zero_balance_code": None,
                }
            )
        end = rng.choice(["active", "prepaid", "default", "defect"])
        code = {"prepaid": 1, "default": 3, "defect": 96}.get(end)
        if code is not None:
            perf_rows[-1]["zero_balance_code"] = code
    orig = pl.DataFrame(orig_rows)
    perf = pl.DataFrame(perf_rows, schema_overrides={"zero_balance_code": pl.Int64})
    return orig, perf


def _brute_force(orig: pl.DataFrame, perf: pl.DataFrame, cutoff: date) -> dict:
    """Cumulative events by (vintage, age), counted loan by loan in plain Python."""
    n = orig.group_by("vintage").len()
    n_orig = dict(zip(n["vintage"], n["len"], strict=True))
    vintage_of = dict(zip(orig["loan_id"], orig["vintage"], strict=True))
    first_event: dict[str, date] = {}
    for r in perf.sort("period").iter_rows(named=True):
        bad = (r["dlq_months"] or 0) >= 3 or r["zero_balance_code"] in (2, 3, 9, 15)
        if bad and r["period"] <= cutoff and r["loan_id"] not in first_event:
            first_event[r["loan_id"]] = r["period"]
    out = {}
    for v, total in n_orig.items():
        qend = months.index(months.quarter_end(v))
        for age in range(1, months.index(cutoff) - qend + 1):
            cell_month = qend + age
            k = sum(
                1
                for lid, p in first_event.items()
                if vintage_of[lid] == v and months.index(p) <= cell_month
            )
            out[(v, age)] = k / total
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_curves_match_brute_force_and_reconcile(seed):
    orig, perf = _random_portfolio(seed)
    data = vintages.build(orig, perf)  # build() runs reconcile(), which raises on any mismatch
    expected = _brute_force(orig, perf, data.cutoff)
    got = {(r["vintage"], r["age"]): r["cum_rate"] for r in data.curves.iter_rows(named=True)}
    assert got.keys() == expected.keys()
    for key, value in expected.items():
        assert got[key] == pytest.approx(value), key
    # Every loan is counted once: denominators equal the origination counts.
    assert data.curves.group_by("vintage").agg(pl.col("n_orig").first())["n_orig"].sum() == 400


@pytest.mark.parametrize("cut", [date(2010, 9, 1), date(2011, 3, 1), date(2011, 12, 1)])
def test_cutoff_is_past_only(cut):
    """Curves built from data up to month t equal the full curves restricted to months <= t."""
    orig, perf = _random_portfolio(5)
    full = vintages.build(orig, perf).curves
    early = vintages.build(orig, perf, cutoff=cut).curves
    expected = full.filter(pl.col("period") <= cut)
    cols = ["vintage", "age", "period", "n_orig", "at_risk", "new_events", "cum_events"]
    assert early.select(cols).sort("vintage", "age").equals(expected.select(cols))


def test_duplicate_rows_are_rejected(fixture_data):
    orig, perf = fixture_data
    with pytest.raises(vintages.ReconciliationError):
        vintages.build(orig, pl.concat([perf, perf.head(1)]))


def test_orphan_performance_is_rejected(fixture_data):
    orig, perf = fixture_data
    with pytest.raises(vintages.ReconciliationError):
        vintages.build(orig.filter(pl.col("loan_id") != "F05Q10000001"), perf)


def test_curves_never_decrease_and_stay_in_unit_interval():
    orig, perf = _random_portfolio(9)
    curves = vintages.build(orig, perf).curves
    assert curves["cum_rate"].min() >= 0 and curves["cum_rate"].max() <= 1
    diffs = curves.sort("vintage", "age").select(pl.col("cum_rate").diff().over("vintage"))
    assert diffs.drop_nulls()["cum_rate"].min() >= 0
