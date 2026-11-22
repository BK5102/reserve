"""Macro loader on a fake FRED cassette: parsing, paging, as-of vintages, and past-only features."""

import math
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from reserve.data import macro

CASSETTE = Path(__file__).parent / "fixtures" / "fred" / "cassette.json"


@pytest.fixture(scope="module")
def client():
    return macro.CassetteFredClient(CASSETTE)


@pytest.fixture(scope="module")
def observations(client):
    return {sid: macro.fetch_observations(client, sid) for sid in macro.SERIES}


def value_at(df: pl.DataFrame, d: date) -> float:
    return df.filter(pl.col("date") == d)["value"][0]


def test_parse_and_missing_value(observations):
    un = observations["UNRATE"]
    assert un.columns == ["series_id", "date", "value", "realtime_start", "realtime_end"]
    assert un.height == 27 + 1 + 1  # 27 months, one revision, one "." value
    assert un.filter(pl.col("date") == date(2020, 4, 1))["value"][0] is None
    assert un["realtime_end"].max() == date(9999, 12, 31)


def test_paging_reads_every_page(observations):
    assert observations["HPIPONM226S"].height == 27


def test_as_of_uses_the_vintage_published_by_then(observations):
    un = observations["UNRATE"]
    before_revision = macro.as_of(un, date(2020, 2, 29))
    after_revision = macro.as_of(un, date(2020, 3, 31))
    assert before_revision["date"].max() == date(2020, 1, 1)
    assert value_at(before_revision, date(2020, 1, 1)) == pytest.approx(6.4)
    assert value_at(after_revision, date(2020, 1, 1)) == pytest.approx(6.9)
    assert macro.as_of(un, date(2020, 2, 6))["date"].max() == date(2019, 12, 1)


def test_features_by_hand(observations):
    feats = macro.macro_features(observations, [date(2020, 2, 1)], mode="asof").row(0, named=True)
    assert feats["unemployment"] == pytest.approx(6.4)  # January, first release
    assert feats["unemployment_change_12m"] == pytest.approx(6.4 - 5.2)
    assert feats["hpi_growth_12m"] == pytest.approx(math.log(273.0 / 261.0))  # Dec over Dec
    thursdays = [date(2020, 2, d) for d in (6, 13, 20, 27)]
    mort = observations["MORTGAGE30US"].filter(pl.col("date").is_in(thursdays))["value"].mean()
    assert feats["mortgage_rate"] == pytest.approx(mort)
    assert feats["revisions_ignored"] is False


def test_features_are_past_only(observations):
    """Removing or poisoning everything published after month t leaves features at t unchanged."""
    feature_months = [date(2019, m, 1) for m in range(1, 13)] + [
        date(2020, m, 1) for m in (1, 2, 3)
    ]
    full = macro.macro_features(observations, feature_months, mode="asof")
    for i, month in enumerate(feature_months):
        known_by = macro.end_of_month(month)
        poisoned = {
            sid: obs.with_columns(
                pl.when(pl.col("realtime_start") > known_by)
                .then(pl.col("value") * 100 + 7)
                .otherwise(pl.col("value"))
                .alias("value")
            )
            for sid, obs in observations.items()
        }
        dropped = {
            sid: obs.filter(pl.col("realtime_start") <= known_by)
            for sid, obs in observations.items()
        }
        for variant in (poisoned, dropped):
            got = macro.macro_features(variant, [month], mode="asof")
            assert got.row(0) == full.row(i), month


def test_latest_mode_flags_ignored_revisions(observations):
    feats = macro.macro_features(observations, [date(2020, 2, 1)], mode="latest").row(0, named=True)
    assert feats["revisions_ignored"] is True
    # Today's revised January value is used as if it had been known in February 2020.
    assert feats["unemployment"] == pytest.approx(6.9)


def test_latest_request_without_realtime(client):
    latest = macro.fetch_observations(client, "UNRATE", realtime=False)
    assert latest["date"].n_unique() == latest.height
    assert value_at(latest, date(2020, 1, 1)) == pytest.approx(6.9)


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv(macro.KEY_ENV, raising=False)
    with pytest.raises(macro.MissingKeyError):
        macro.HttpFredClient()


def test_cassette_replay_and_record(tmp_path):
    with pytest.raises(FileNotFoundError):
        macro.CassetteFredClient(tmp_path / "none.json")
    replay = macro.CassetteFredClient(CASSETTE)
    with pytest.raises(KeyError):
        replay.get("series/observations", {"series_id": "NOPE"})

    class Backend:
        def get(self, endpoint, params):
            return {"echo": params["series_id"]}

    path = tmp_path / "rec.json"
    rec = macro.CassetteFredClient(path, mode="record", backend=Backend())
    rec.get("series/observations", {"series_id": "X", "api_key": "secret-not-stored"})
    text = path.read_text()
    assert "secret-not-stored" not in text and "series_id=X" in text


def test_download_and_load_round_trip(tmp_path, client):
    macro.download_all(client, tmp_path)
    loaded = macro.load_observations(tmp_path)
    assert set(loaded) == set(macro.SERIES)
    assert loaded["UNRATE"].height == 29


def test_fallback_only_before_a_series_first_vintage(observations):
    """asof_fallback = asof wherever ALFRED has a vintage; latest values only before it (M48)."""
    # HPIPONM226S's first fixture vintage is 2018-03-07, so February 2018 has no record.
    early = macro.macro_features(observations, [date(2018, 2, 1)], mode="asof_fallback")
    strict = macro.macro_features(observations, [date(2018, 2, 1)], mode="asof")
    row = early.row(0, named=True)
    assert row["latest_fallback"] == "HPIPONM226S" and row["revisions_ignored"] is True
    assert row["unemployment"] == strict["unemployment"][0]  # UNRATE had a vintage: as-of
    assert row["mortgage_rate"] == strict["mortgage_rate"][0]
    later = [date(2019, m, 1) for m in range(1, 13)] + [date(2020, 2, 1)]
    a = macro.macro_features(observations, later, mode="asof")
    f = macro.macro_features(observations, later, mode="asof_fallback")
    assert f["latest_fallback"].null_count() == f.height
    assert not f["revisions_ignored"].any()
    assert f.drop("latest_fallback").equals(a.drop("latest_fallback"))


def test_fallback_fills_a_missing_house_price_record(observations):
    """No vintage for a month: strict as-of leaves the feature empty; fallback fills it."""
    month = [date(2019, 3, 1)]
    hpi_late = {
        **observations,
        "HPIPONM226S": observations["HPIPONM226S"].with_columns(
            pl.lit(date(2019, 6, 1)).alias("realtime_start")
        ),
    }
    assert macro.macro_features(hpi_late, month, mode="asof")["hpi_growth_12m"][0] is None
    filled = macro.macro_features(hpi_late, month, mode="asof_fallback").row(0, named=True)
    latest = macro.macro_features(hpi_late, month, mode="latest").row(0, named=True)
    assert filled["hpi_growth_12m"] == pytest.approx(latest["hpi_growth_12m"])
    assert filled["latest_fallback"] == "HPIPONM226S"


def test_unknown_mode_is_refused(observations):
    with pytest.raises(ValueError, match="mode must be one of"):
        macro.macro_features(observations, [date(2019, 3, 1)], mode="asoff")
