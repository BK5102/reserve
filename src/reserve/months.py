"""Calendar months as integers, so month arithmetic is exact and cheap.

A month index is year * 12 + (month - 1). Dates always stand for the first day of their month.
"""

from __future__ import annotations

from datetime import date

import polars as pl


def index(d: date) -> int:
    return d.year * 12 + d.month - 1


def from_index(m: int) -> date:
    return date(m // 12, m % 12 + 1, 1)


def add(d: date, months: int) -> date:
    return from_index(index(d) + months)


def parse(s: str) -> date:
    """The first of the month for "2005-03", "2005-03-01", or a date."""
    if isinstance(s, date):
        return date(s.year, s.month, 1)
    parts = str(s).split("-")
    return date(int(parts[0]), int(parts[1]), 1)


def expr_index(col: str | pl.Expr) -> pl.Expr:
    e = pl.col(col) if isinstance(col, str) else col
    return e.dt.year().cast(pl.Int64) * 12 + e.dt.month().cast(pl.Int64) - 1


def expr_from_index(e: pl.Expr) -> pl.Expr:
    return pl.date(e // 12, e % 12 + 1, 1)


def quarter_end(vintage: str) -> date:
    """The last month of a quarter: vintage "2005Q1" gives 2005-03-01."""
    year, q = vintage.split("Q")
    return date(int(year), int(q) * 3, 1)


def expr_quarter_end(col: str | pl.Expr) -> pl.Expr:
    e = pl.col(col) if isinstance(col, str) else col
    year = e.str.slice(0, 4).cast(pl.Int64)
    q = e.str.slice(5, 1).cast(pl.Int64)
    return pl.date(year, q * 3, 1)


def vintage_of(d: date) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"
