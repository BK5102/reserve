"""The TSFM interface: any pretrained forecaster that returns quantiles plugs in here (M27).

A `QuantileForecaster` takes plain number sequences (no dates) and returns quantile
forecasts. `TSFMContestant` turns it into a contestant: each vintage's context is its own
cumulative curve from age 1 to the origin, indexed by age only, so the model never sees a
calendar date (the "date-masked" input of PLAN.md FR7 is the default here).
"""

from __future__ import annotations

import time
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


def map_quantiles(
    native_levels: Sequence[float], values: np.ndarray, targets: Sequence[float]
) -> np.ndarray:
    """Map forecasts at a model's own quantile levels to Reserve's levels (methodology M32).

    `values` has the native levels on its last axis. A target inside the native range is
    linearly interpolated between the two nearest native levels; a target outside it gets
    the nearest native level (flat). No tail is invented: if a model was trained on 0.1 to
    0.9, Reserve's 0.05 and 0.95 are its 0.1 and 0.9, so its "90%" interval is really its
    own 80% interval, and its 90% coverage is expected to be low.
    """
    lv = np.asarray(native_levels, dtype=float)
    v = np.asarray(values, dtype=float)
    if lv.ndim != 1 or v.shape[-1] != lv.size:
        raise ValueError(f"values' last axis ({v.shape[-1]}) must match {lv.size} native levels")
    if np.any(np.diff(lv) <= 0):
        raise ValueError("native levels must be strictly increasing")
    out = np.empty((*v.shape[:-1], len(targets)))
    for j, t in enumerate(targets):
        if t <= lv[0]:
            out[..., j] = v[..., 0]
        elif t >= lv[-1]:
            out[..., j] = v[..., -1]
        else:
            i = int(np.searchsorted(lv, t))  # lv[i - 1] < t <= lv[i]
            w = (t - lv[i - 1]) / (lv[i] - lv[i - 1])
            out[..., j] = (1 - w) * v[..., i - 1] + w * v[..., i]
    return out


def describe_mapping(native_levels: Sequence[float], targets: Sequence[float]) -> dict[str, str]:
    """Plain description of `map_quantiles` for each target level, stored with every run."""
    lv = [float(x) for x in native_levels]
    out = {}
    for t in targets:
        if any(abs(t - x) < 1e-9 for x in lv):
            out[f"{t:g}"] = "native"
        elif t < lv[0]:
            out[f"{t:g}"] = f"native {lv[0]:g} (below the trained range, held flat)"
        elif t > lv[-1]:
            out[f"{t:g}"] = f"native {lv[-1]:g} (above the trained range, held flat)"
        else:
            i = int(np.searchsorted(lv, t))
            out[f"{t:g}"] = f"linear between native {lv[i - 1]:g} and {lv[i]:g}"
    return out


class TSFMContestant:
    def __init__(self, model: QuantileForecaster, max_context: int | None = None):
        self.model = model
        self.max_context = max_context
        self.name = f"D_tsfm_{model.name}"
        # Cost of the model calls only (not context building), for docs/tsfm-notes.md.
        self.cost = {"calls": 0, "series": 0, "wall_seconds": 0.0, "cpu_seconds": 0.0}

    def describe(self) -> dict:
        out = {"name": self.name, "model": self.model.name, "version": self.model.version}
        extra = getattr(self.model, "describe", None)
        if callable(extra):
            out.update(extra())
        c = self.cost
        out["cost"] = {
            **c,
            "wall_ms_per_series": round(1000 * c["wall_seconds"] / c["series"], 3)
            if c["series"]
            else None,
            "cpu_ms_per_series": round(1000 * c["cpu_seconds"] / c["series"], 3)
            if c["series"]
            else None,
        }
        return out

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
        t_wall, t_cpu = time.perf_counter(), time.process_time()
        raw = np.asarray(self.model.forecast(contexts, horizon, quantiles), dtype=float)
        self.cost["calls"] += 1
        self.cost["series"] += len(contexts)
        self.cost["wall_seconds"] += time.perf_counter() - t_wall
        self.cost["cpu_seconds"] += time.process_time() - t_cpu
        expected = (len(contexts), horizon, len(quantiles))
        if raw.shape != expected:
            raise ValueError(f"{self.model.name} returned shape {raw.shape}, expected {expected}")
        c0 = state["cum_rate"].to_numpy()
        return {v: tidy(raw[i], c0[i]) for i, v in enumerate(names)}
