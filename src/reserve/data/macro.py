"""Macro series from FRED, with an as-of (real-time) mode built on ALFRED vintages.

FRED is the St. Louis Fed's free economic database; ALFRED is its archive of earlier
versions of each series. Real-time ("as-of") mode asks the FRED API for every vintage of
a series (`realtime_start=1776-07-04`, `realtime_end=9999-12-31`; see
https://fred.stlouisfed.org/docs/api/fred/series_observations.html) and then, for each
forecast date, keeps only the version that was published by that date. Latest mode uses
today's values with a fixed publication lag per series and marks every output
`revisions_ignored=True` (CLAUDE.md rule 1; methodology M13 to M16).

The API key is read from the FRED_API_KEY environment variable and never written to disk.
Requests go through a `FredClient`, so tests replay JSON cassettes and never touch the network
(the same pattern as Tells' `agent/llm.py`).
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl
from cleanroom.asof import join_as_known

from reserve import months

API_ROOT = "https://api.stlouisfed.org/fred/"
KEY_ENV = "FRED_API_KEY"
ALL_REALTIME = {"realtime_start": "1776-07-04", "realtime_end": "9999-12-31"}
PAGE = 100000  # the API's maximum `limit`


@dataclass(frozen=True)
class Series:
    series_id: str
    title: str
    frequency: str  # "m" monthly or "w" weekly
    publication_lag_months: int  # latest mode only: month m is treated as known at end of m+lag
    source: str


# Ids, titles, units, and frequencies checked on the FRED series pages on 2026-11-17 (M13).
SERIES: dict[str, Series] = {
    "UNRATE": Series(
        "UNRATE",
        "Unemployment Rate, percent, monthly, seasonally adjusted (BLS)",
        "m",
        1,
        "https://fred.stlouisfed.org/series/UNRATE",
    ),
    "MORTGAGE30US": Series(
        "MORTGAGE30US",
        "30-Year Fixed Rate Mortgage Average in the United States, percent, weekly ending "
        "Thursday, not seasonally adjusted (Freddie Mac PMMS)",
        "w",
        0,
        "https://fred.stlouisfed.org/series/MORTGAGE30US",
    ),
    "HPIPONM226S": Series(
        "HPIPONM226S",
        "Purchase Only House Price Index for the United States, index Jan 1991=100, monthly, "
        "seasonally adjusted (FHFA)",
        "m",
        2,
        "https://fred.stlouisfed.org/series/HPIPONM226S",
    ),
}


class MissingKeyError(RuntimeError):
    """FRED_API_KEY is not set."""


class FredClient(Protocol):
    def get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]: ...


def request_key(endpoint: str, params: dict[str, Any]) -> str:
    """A stable name for a request, without the API key."""
    clean = {k: str(v) for k, v in sorted(params.items()) if k != "api_key"}
    return endpoint + "?" + urllib.parse.urlencode(clean)


class HttpFredClient:
    """Real requests to the FRED API. Needs FRED_API_KEY in the environment."""

    def __init__(self, timeout: float = 60.0):
        key = os.environ.get(KEY_ENV, "").strip()
        if not key:
            raise MissingKeyError(
                f"set {KEY_ENV} to a free FRED API key (https://fred.stlouisfed.org/docs/api/"
                "api_key.html); it is read from the environment only"
            )
        self._key = key
        self.timeout = timeout

    def get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode({**params, "api_key": self._key, "file_type": "json"})
        with urllib.request.urlopen(f"{API_ROOT}{endpoint}?{query}", timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))


class CassetteFredClient:
    """Replay recorded responses (default), or record new ones through a real client."""

    def __init__(self, path: str | Path, mode: str = "replay", backend: FredClient | None = None):
        if mode not in ("replay", "record"):
            raise ValueError("mode must be 'replay' or 'record'")
        if mode == "record" and backend is None:
            raise ValueError("record mode needs a real client")
        self.path = Path(path)
        self.mode = mode
        self.backend = backend
        self._recorded: dict[str, Any] = (
            json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        )
        if mode == "replay" and not self._recorded:
            raise FileNotFoundError(f"no cassette at {self.path}")

    def get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        key = request_key(endpoint, params)
        if self.mode == "record":
            assert self.backend is not None
            response = self.backend.get(endpoint, params)
            self._recorded[key] = response
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._recorded, indent=1), encoding="utf-8")
            return response
        if key not in self._recorded:
            raise KeyError(f"cassette {self.path.name} has no response for {key}")
        return self._recorded[key]


def _parse_date(s: pl.Expr) -> pl.Expr:
    return s.str.strptime(pl.Date, "%Y-%m-%d", strict=True)


def fetch_observations(client: FredClient, series_id: str, realtime: bool = True) -> pl.DataFrame:
    """All observations of a series: every ALFRED vintage if `realtime`, else today's values.

    Columns: series_id, date, value (null where FRED reports "."), realtime_start,
    realtime_end. Pages through the API's 100,000-row limit.
    """
    params: dict[str, Any] = {"series_id": series_id, "limit": PAGE, "offset": 0}
    if realtime:
        params.update(ALL_REALTIME)
    rows: list[dict[str, Any]] = []
    while True:
        payload = client.get("series/observations", params)
        batch = payload.get("observations", [])
        rows.extend(batch)
        count = int(payload.get("count", len(rows)))
        if not batch or len(rows) >= count:
            break
        params = {**params, "offset": len(rows)}
    if not rows:
        raise ValueError(f"FRED returned no observations for {series_id}")
    df = pl.DataFrame(rows).select("date", "value", "realtime_start", "realtime_end")
    value = pl.col("value")
    return df.select(
        pl.lit(series_id).alias("series_id"),
        _parse_date(pl.col("date")).alias("date"),
        # FRED marks a missing value with "." (not stated on the API page; methodology M14).
        pl.when(value == ".").then(None).otherwise(value).cast(pl.Float64).alias("value"),
        _parse_date(pl.col("realtime_start")).alias("realtime_start"),
        _parse_date(pl.col("realtime_end")).alias("realtime_end"),
    )


def as_of(obs: pl.DataFrame, when: date) -> pl.DataFrame:
    """The series as it was published on date `when`: one row per observation date.

    Uses the shared `cleanroom.asof.join_as_known`: for each observation date, the latest
    release whose `realtime_start` is on or before `when` (`inclusive=True`, because a
    FRED vintage date is the day the value was available). A latest release of "." means
    the value was not available then, so the date is left out.
    """
    known = join_as_known(
        obs, "date", "value", "realtime_start", when, at_col="as_of", inclusive=True
    )
    return known.filter(pl.col("value").is_not_null()).select("date", "value").sort("date")


def latest_with_lag(obs: pl.DataFrame, series: Series, month: date) -> pl.DataFrame:
    """Latest-mode stand-in for `as_of`: today's values, observation month m known at m + lag.

    Revisions are ignored: a value revised later is used as if it had been known then.
    """
    newest = obs.group_by("date").agg(pl.all().sort_by("realtime_start").last())
    last_known = months.add(month, -series.publication_lag_months)
    keep = months.expr_index("date") <= months.index(last_known)
    return newest.filter(keep & pl.col("value").is_not_null()).select("date", "value").sort("date")


def to_monthly(df: pl.DataFrame, frequency: str) -> pl.DataFrame:
    """Monthly values: monthly series as is, weekly series averaged within each calendar month."""
    if frequency == "m":
        return df.with_columns(pl.col("date").dt.truncate("1mo")).sort("date")
    return df.group_by(pl.col("date").dt.truncate("1mo")).agg(pl.col("value").mean()).sort("date")


def end_of_month(d: date) -> date:
    nxt = months.add(d, 1)
    return date.fromordinal(nxt.toordinal() - 1)


FEATURES = ("unemployment", "unemployment_change_12m", "mortgage_rate", "hpi_growth_12m")


def _known_series(obs: pl.DataFrame, series: Series, month: date, mode: str) -> pl.DataFrame:
    if mode == "asof":
        known = as_of(obs, end_of_month(month))
    elif mode == "latest":
        known = latest_with_lag(obs, series, month)
    else:
        raise ValueError("mode must be 'asof' or 'latest'")
    known = to_monthly(known, series.frequency)
    return known.filter(pl.col("date") <= month)


def _last_and_lag(known: pl.DataFrame, lag: int) -> tuple[float, float]:
    """The newest known value and the value `lag` months before it (NaN when missing)."""
    if known.height == 0:
        return np.nan, np.nan
    last_date = known["date"][-1]
    last = known["value"][-1]
    earlier = known.filter(pl.col("date") == months.add(last_date, -lag))
    return float(last), float(earlier["value"][0]) if earlier.height else np.nan


def macro_features(
    observations: dict[str, pl.DataFrame], feature_months: list[date], mode: str = "asof"
) -> pl.DataFrame:
    """Past-only macro features for each month: what an analyst knew at the end of that month.

    - unemployment: newest known UNRATE.
    - unemployment_change_12m: that value minus the value 12 months before it (same vintage).
    - mortgage_rate: newest known monthly average of MORTGAGE30US.
    - hpi_growth_12m: log change of HPIPONM226S over the 12 months to its newest known month.

    `mode="asof"` uses ALFRED vintages; `mode="latest"` uses today's values with each
    series' publication lag and sets `revisions_ignored` to true.
    """
    rows = []
    for month in feature_months:
        known = {
            sid: _known_series(observations[sid], SERIES[sid], month, mode)
            for sid in ("UNRATE", "MORTGAGE30US", "HPIPONM226S")
        }
        u, u12 = _last_and_lag(known["UNRATE"], 12)
        r, _ = _last_and_lag(known["MORTGAGE30US"], 0)
        h, h12 = _last_and_lag(known["HPIPONM226S"], 12)
        rows.append(
            {
                "period": month,
                "unemployment": u,
                "unemployment_change_12m": u - u12,
                "mortgage_rate": r,
                "hpi_growth_12m": float(np.log(h / h12)) if h12 and h12 > 0 else np.nan,
                "revisions_ignored": mode == "latest",
            }
        )
    return pl.DataFrame(rows).with_columns(pl.col(FEATURES).fill_nan(None))


def load_observations(raw_dir: str | Path) -> dict[str, pl.DataFrame]:
    """Read saved observation tables (Parquet, one per series) from `raw_dir`."""
    raw_dir = Path(raw_dir)
    return {sid: pl.read_parquet(raw_dir / f"{sid}.parquet") for sid in SERIES}


def download_all(client: FredClient, out_dir: str | Path, realtime: bool = True) -> list[Path]:
    """Fetch every series and save it as Parquet. Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sid in SERIES:
        path = out_dir / f"{sid}.parquet"
        fetch_observations(client, sid, realtime=realtime).write_parquet(path)
        paths.append(path)
    return paths
