"""Classical contestants A and B (methodology M24, M25). Contestant C is in `hazard.py`.

A. Last increment persists: the vintage keeps going bad at its own recent monthly pace.
B. Average age curve ("chain-ladder style"): the vintage follows the average monthly
   increments that recent older vintages showed at the same ages.

Both use only the history cells they are given.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl
from scipy import stats as sps

from reserve.models.base import tidy, vintage_state


class LastIncrement:
    """A: c(a0 + k) = c0 + k * d, d = mean monthly increment over the vintage's last `window` ages.

    Quantiles treat the future event count over k months as Poisson with mean k * d * n_orig,
    so they reflect sampling noise only (M24).
    """

    name = "A_last_increment"

    def __init__(self, window: int = 3):
        self.window = window

    def describe(self) -> dict:
        return {"name": self.name, "window": self.window}

    def forecast(
        self,
        history: pl.DataFrame,
        origin: date,
        vintages: Sequence[str],
        horizon: int,
        quantiles: Sequence[float],
    ) -> dict[str, np.ndarray]:
        state = vintage_state(history, origin, vintages)
        recent = (
            history.filter(pl.col("vintage").is_in(list(vintages)))
            .sort("vintage", "age")
            .group_by("vintage", maintain_order=True)
            .agg(pl.col("increment").tail(self.window).mean().alias("d"))
        )
        d = dict(zip(recent["vintage"], recent["d"], strict=True))
        k = np.arange(1, horizon + 1)
        out = {}
        for row in state.iter_rows(named=True):
            lam = max(float(d[row["vintage"]]), 0.0) * row["n_orig"] * k
            counts = np.stack(
                [sps.poisson.ppf(q, lam) if lam.any() else 0 * k for q in quantiles], 1
            )
            out[row["vintage"]] = tidy(row["cum_rate"] + counts / row["n_orig"], row["cum_rate"])
        return out


class AgeCurve:
    """B: c(a0 + k) = c0 + sum of the mean increments of the `donors` newest vintages at each age.

    Interval: normal, with variance the sum over ages of the donors' increment variance
    (assumes increments independent across ages, M25). Ages no donor has reached yet reuse
    the donors at the oldest age available (flat extrapolation).
    """

    name = "B_age_curve"

    def __init__(self, donors: int = 8):
        self.donors = donors

    def describe(self) -> dict:
        return {"name": self.name, "donors": self.donors}

    def _age_stats(self, history: pl.DataFrame) -> pl.DataFrame:
        return (
            history.sort("vintage", descending=True)
            .group_by("age")
            .agg(
                pl.col("increment").head(self.donors).mean().alias("mean"),
                pl.col("increment").head(self.donors).var(ddof=1).fill_null(0.0).alias("var"),
                pl.col("increment").head(self.donors).len().alias("n_donors"),
            )
            .sort("age")
        )

    def forecast(
        self,
        history: pl.DataFrame,
        origin: date,
        vintages: Sequence[str],
        horizon: int,
        quantiles: Sequence[float],
    ) -> dict[str, np.ndarray]:
        state = vintage_state(history, origin, vintages)
        # Donors for an age are vintages other than the one forecast; at its future ages
        # only older vintages have data, so excluding it changes nothing there.
        by_age = self._age_stats(history)
        ages = by_age["age"].to_numpy()
        mean = by_age["mean"].to_numpy()
        var = by_age["var"].to_numpy()
        z = sps.norm.ppf(np.asarray(quantiles))
        out = {}
        for row in state.iter_rows(named=True):
            target = row["age"] + np.arange(1, horizon + 1)
            pos = np.searchsorted(ages, np.minimum(target, ages.max()), side="left")
            pos = np.minimum(pos, len(ages) - 1)
            m = np.cumsum(mean[pos])
            s = np.sqrt(np.cumsum(var[pos]))
            out[row["vintage"]] = tidy(
                row["cum_rate"] + m[:, None] + s[:, None] * z, row["cum_rate"]
            )
        return out
