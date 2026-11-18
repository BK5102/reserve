"""Write `cassette.json`: FAKE FRED API responses in the documented JSON format.

No FRED API key exists yet, so these are not recordings of real data. Every value is
invented. The structure (top-level `count`, `offset`, `limit`, `observations`, and each
observation's `realtime_start`, `realtime_end`, `date`, `value` as text) follows
https://fred.stlouisfed.org/docs/api/fred/series_observations.html. Once a key exists,
replace this with a real recording (`reserve prepare macro --record`).

Run from the repo root: `uv run python tests/fixtures/fred/make_cassette.py`.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from reserve import months
from reserve.data.macro import ALL_REALTIME, PAGE, request_key

FOREVER = "9999-12-31"
START = months.parse("2018-01")
N_MONTHS = 27  # 2018-01 to 2020-03


def obs(date, value, start, end=FOREVER):
    return {"realtime_start": start, "realtime_end": end, "date": date, "value": value}


def monthly(base: float, step: float, release_lag: int, revise: dict[str, str] | None = None):
    """One row per month, first released on the 7th of the month `release_lag` months later.

    `revise` maps an observation month to (new value): its first release ends the day
    before the next month's release, and the revision starts then.
    """
    rows = []
    for i in range(N_MONTHS):
        d = months.add(START, i)
        released = months.add(d, release_lag).replace(day=7).isoformat()
        value = f"{base + step * i:.1f}"
        key = d.isoformat()
        if revise and key in revise:
            revised_on = months.add(d, release_lag + 1).replace(day=7)
            end = date.fromordinal(revised_on.toordinal() - 1).isoformat()
            rows.append(obs(key, value, released, end))
            rows.append(obs(key, revise[key], revised_on.isoformat()))
        else:
            rows.append(obs(key, value, released))
    return rows


def weekly():
    """Thursdays from 2018-01-04, released the same day."""
    rows = []
    d = date(2018, 1, 4)
    i = 0
    while d <= date(2020, 3, 26):
        rows.append(obs(d.isoformat(), f"{4.00 + 0.01 * (i % 20):.2f}", d.isoformat()))
        d += timedelta(days=7)
        i += 1
    return rows


def response(observations, count=None, offset=0):
    return {
        **ALL_REALTIME,
        "observation_start": "1776-07-04",
        "observation_end": FOREVER,
        "units": "lin",
        "output_type": 1,
        "file_type": "json",
        "order_by": "observation_date",
        "sort_order": "asc",
        "count": len(observations) if count is None else count,
        "offset": offset,
        "limit": PAGE,
        "observations": observations,
    }


def main() -> None:
    cassette = {}
    unrate = monthly(4.0, 0.1, 1, revise={"2020-01-01": "6.9"})
    unrate.append(obs("2020-04-01", ".", "2020-05-07"))  # a missing value, FRED style
    hpi = monthly(250.0, 1.0, 2)
    series = {"UNRATE": [unrate], "MORTGAGE30US": [weekly()], "HPIPONM226S": [hpi[:10], hpi[10:]]}
    for sid, pages in series.items():
        total = sum(len(p) for p in pages)
        offset = 0
        for page in pages:
            params = {"series_id": sid, "limit": PAGE, "offset": offset, **ALL_REALTIME}
            cassette[request_key("series/observations", params)] = response(page, total, offset)
            offset += len(page)
    # Latest-mode request (no real-time range) for UNRATE: today's values only.
    newest = {r["date"]: r["value"] for r in unrate}  # revisions come after first releases
    latest = [obs(d, v, "2026-11-17", "2026-11-17") for d, v in newest.items()]
    params = {"series_id": "UNRATE", "limit": PAGE, "offset": 0}
    cassette[request_key("series/observations", params)] = response(latest)
    out = Path(__file__).with_name("cassette.json")
    out.write_text(json.dumps(cassette, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(cassette)} responses")


if __name__ == "__main__":
    main()
