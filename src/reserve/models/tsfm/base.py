"""The TSFM interface: any pretrained forecaster that returns quantiles plugs in here (M27).

A `QuantileForecaster` takes plain number sequences (no dates) and returns quantile
forecasts. `TSFMContestant` turns it into a contestant: each vintage's context is its own
cumulative curve from age 1 to the origin, indexed by age only, so the model never sees a
calendar date (the "date-masked" input of PLAN.md FR7 is the default here).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol

import numpy as np
import polars as pl

from reserve.models.base import tidy, vintage_state


class QuantileForecaster(Protocol):
    name: str
    version: str  # model id plus revision or library version, for provenance

    def forecast(
        self, contexts: list[np.ndarray], horizon: int, quantiles: Sequence[float]
    ) -> np.ndarray:
        """Return an array of shape (len(contexts), horizon, len(quantiles))."""
        ...


class TSFMContestant:
    def __init__(self, model: QuantileForecaster, max_context: int | None = None):
        self.model = model
        self.max_context = max_context
        self.name = f"D_tsfm_{model.name}"

    def describe(self) -> dict:
        return {"name": self.name, "model": self.model.name, "version": self.model.version}

    def forecast(
        self,
        history: pl.DataFrame,
        origin: date,
        vintages: Sequence[str],
        horizon: int,
        quantiles: Sequence[float],
    ) -> dict[str, np.ndarray]:
        state = vintage_state(history, origin, vintages)
        names = state["vintage"].to_list()
        curves = (
            history.filter(pl.col("vintage").is_in(names))
            .sort("vintage", "age")
            .group_by("vintage", maintain_order=True)
            .agg(pl.col("cum_rate"))
        )
        by_name = dict(zip(curves["vintage"], curves["cum_rate"], strict=True))
        contexts = []
        for v in names:
            c = np.asarray(by_name[v].to_list(), dtype=float)
            contexts.append(c[-self.max_context :] if self.max_context else c)
        if not contexts:
            return {}
        raw = np.asarray(self.model.forecast(contexts, horizon, quantiles), dtype=float)
        expected = (len(contexts), horizon, len(quantiles))
        if raw.shape != expected:
            raise ValueError(f"{self.model.name} returned shape {raw.shape}, expected {expected}")
        c0 = state["cum_rate"].to_numpy()
        return {v: tidy(raw[i], c0[i]) for i, v in enumerate(names)}
