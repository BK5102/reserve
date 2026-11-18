"""What every contestant looks like, and the shared clean-up of quantile forecasts.

A contestant receives only the vintage cells known at the origin (`history`, every row with
period <= origin month; the harness cuts it with a Cleanroom split) and returns, for each
requested vintage, an array of shape (horizon, n_quantiles): forecast cumulative rates for
ages a0 + 1 ... a0 + horizon, where a0 is the vintage's age at the origin.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol

import numpy as np
import polars as pl

QUANTILES: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


class Contestant(Protocol):
    name: str

    def forecast(
        self,
        history: pl.DataFrame,
        origin: date,
        vintages: Sequence[str],
        horizon: int,
        quantiles: Sequence[float],
    ) -> dict[str, np.ndarray]: ...

    def describe(self) -> dict: ...


def tidy(paths: np.ndarray, c0: float) -> np.ndarray:
    """Make a (horizon, n_quantiles) forecast of a cumulative share coherent.

    A cumulative share cannot fall below its last known value or rise above 1, cannot
    decrease with horizon, and its quantiles cannot cross (methodology M23).
    """
    out = np.clip(np.asarray(paths, dtype=float), c0, 1.0)
    out = np.maximum.accumulate(out, axis=0)
    return np.sort(out, axis=1)


def vintage_state(history: pl.DataFrame, origin: date, vintages: Sequence[str]) -> pl.DataFrame:
    """Each vintage's cell at the origin month: age a0, cum_rate c0, n_orig, at-risk counts."""
    return history.filter(
        (pl.col("period") == origin) & pl.col("vintage").is_in(list(vintages))
    ).sort("vintage")
