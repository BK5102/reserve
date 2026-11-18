"""Forecast origins as Cleanroom splits (CLAUDE.md rule 2; methodology M22).

A forecast origin is a calendar month t: everything up to and including month t is
known, and the forecast covers months t+1 to t+horizon. The splits come from the shared
`cleanroom.split.rolling_origins`, which steps by a `timedelta`. Calendar months have
different lengths, so this module puts the cells on a "month axis" where one calendar
month is exactly one day after a fixed epoch. On that axis, a step of 3 days is exactly
3 calendar months, and the split's train window [start, t+1) holds exactly the months
up to t. `Split.check` then re-verifies the order on the actual rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl
from cleanroom.split import Split, rolling_origins

from reserve import months

EPOCH = datetime(2000, 1, 1)
_EPOCH_INDEX = months.index(date(2000, 1, 1))
TIME_COL = "month_time"


def month_time(d: date) -> datetime:
    """Calendar month -> its point on the month axis (one month = one day)."""
    return EPOCH + timedelta(days=months.index(d) - _EPOCH_INDEX)


def from_month_time(t: datetime) -> date:
    return months.from_index(_EPOCH_INDEX + (t - EPOCH).days)


def with_month_time(df: pl.DataFrame, col: str = "period") -> pl.DataFrame:
    idx = months.expr_index(col) - _EPOCH_INDEX
    return df.with_columns(
        (pl.lit(EPOCH) + pl.duration(days=idx)).cast(pl.Datetime("us")).alias(TIME_COL)
    )


@dataclass(frozen=True)
class Origin:
    month: date  # the last month whose data is known
    split: Split

    def apply(self, cells: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """(history, future) cells for this origin, checked for time order on the rows."""
        parts = self.split.apply(cells, TIME_COL)
        self.split.check(parts, TIME_COL)
        return parts["train"].drop(TIME_COL), parts["test"].drop(TIME_COL)


def monthly_origins(first: date, last: date, step_months: int, horizon_months: int) -> list[Origin]:
    """Origins first, first + step, ... up to last, each with a `horizon_months` test window."""
    splits = rolling_origins(
        start=month_time(months.add(first, 1)),
        end=month_time(months.add(last, 1 + horizon_months)),
        step=timedelta(days=step_months),
        horizon=timedelta(days=horizon_months),
    )
    return [Origin(months.add(from_month_time(s.train.end), -1), s) for s in splits]
