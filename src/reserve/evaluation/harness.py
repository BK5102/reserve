"""The forecasting harness: every contestant, every origin, the same inputs (methodology M22).

For each origin, the Cleanroom split cuts the vintage cells into history (months up to the
origin) and future (the next `horizon` months). Contestants see only the history. The
future is used only to look up the outcomes after every contestant has forecast.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl

from reserve import months
from reserve.evaluation.origins import Origin, with_month_time
from reserve.models.base import Contestant


def qcol(q: float) -> str:
    return f"q{q:g}"


def eligible_vintages(
    history: pl.DataFrame, origin: date, min_history: int, max_age: int | None = None
) -> list[str]:
    """Vintages at the origin with at least `min_history` and at most `max_age` months of curve."""
    cond = (pl.col("period") == origin) & (pl.col("age") >= min_history)
    if max_age is not None:
        cond = cond & (pl.col("age") <= max_age)
    return sorted(history.filter(cond)["vintage"].to_list())


def thin_history(history: pl.DataFrame, origin: date, months: int) -> pl.DataFrame:
    """History of a portfolio that started `months` months before the origin (methodology M34).

    Keeps only vintages whose curve began within the last `months` months, that is, whose age
    at the origin is at most `months`. All their cells lie inside that window, and every
    older vintage disappears, as it would for a new loan product.
    """
    young = history.filter((pl.col("period") == origin) & (pl.col("age") <= months))["vintage"]
    return history.filter(pl.col("vintage").is_in(young.to_list()))


def run(
    curves: pl.DataFrame,
    contestants: Sequence[Contestant],
    origins: Sequence[Origin],
    horizon: int,
    quantiles: Sequence[float],
    min_history: int = 6,
    history_months: int | None = None,
    max_target_age: int | None = None,
) -> pl.DataFrame:
    """Long table: one row per (origin, vintage, k, contestant) with outcome and quantiles.

    `y` is the observed cumulative rate at age a0 + k, null when that month is past the
    data end. `c0` is the rate known at the origin. `history_months` cuts every
    contestant's history to a portfolio that is that many months old (`thin_history`);
    `max_target_age` forecasts only vintages at most that old. Both act on the history
    after the Cleanroom split, so they can only remove information, never add it.
    """
    cells = with_month_time(curves)
    rows: list[pl.DataFrame] = []
    for origin in origins:
        history, future = origin.apply(cells)
        if history_months is not None:
            history = thin_history(history, origin.month, history_months)
        vintages = eligible_vintages(history, origin.month, min_history, max_target_age)
        if not vintages:
            continue
        state = history.filter(
            (pl.col("period") == origin.month) & pl.col("vintage").is_in(vintages)
        ).select("vintage", pl.col("age").alias("a0"), pl.col("cum_rate").alias("c0"), "n_orig")
        grid = state.join(pl.DataFrame({"k": list(range(1, horizon + 1))}), how="cross")
        grid = grid.with_columns((pl.col("a0") + pl.col("k")).alias("age"))
        truth = grid.join(
            future.select("vintage", "age", "period", pl.col("cum_rate").alias("y")),
            on=["vintage", "age"],
            how="left",
        ).with_columns(
            pl.lit(origin.month).alias("origin"),
            months.expr_from_index(pl.col("k") + months.index(origin.month)).alias("target_period"),
        )
        for c in contestants:
            preds = c.forecast(history, origin.month, vintages, horizon, quantiles)
            missing = set(vintages) - set(preds)
            if missing:
                raise ValueError(f"{c.name} gave no forecast for {sorted(missing)[:3]}")
            arr = np.stack([preds[v] for v in state["vintage"]])  # (vintages, horizon, Q)
            flat = arr.reshape(-1, len(quantiles))
            qframe = pl.DataFrame({qcol(q): flat[:, j] for j, q in enumerate(quantiles)})
            block = (
                state.select("vintage")
                .join(pl.DataFrame({"k": list(range(1, horizon + 1))}), how="cross")
                .hstack(qframe)
                .with_columns(pl.lit(c.name).alias("contestant"))
            )
            rows.append(truth.join(block, on=["vintage", "k"], how="left"))
    if not rows:
        raise ValueError("no origin had any eligible vintage")
    out = pl.concat(rows, how="vertical_relaxed")
    return out.select(
        "origin",
        "vintage",
        "k",
        "a0",
        "age",
        "target_period",
        "n_orig",
        "c0",
        "y",
        "contestant",
        *[qcol(q) for q in quantiles],
    ).sort("contestant", "origin", "vintage", "k")
