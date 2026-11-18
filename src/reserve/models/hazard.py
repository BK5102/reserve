"""Contestant C: a monthly hazard regression on loan age plus past-only macro (methodology M26).

The hazard is the chance that a loan still at risk goes 90+ days late this month. C fits a
binomial logistic regression on every history cell (new events out of loans at risk) with

    logit(h) = b0 + age spline + b . macro(period)

where macro(period) is what was known at the end of that cell's month (`macro_features`).
The forecast for a vintage holds the macro features at their origin-month values (future
macro values are unknown at the origin), projects the loans at risk forward with a pooled
recent exit rate, and draws parameter and binomial noise by Monte Carlo for the quantiles.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl

from reserve.data.macro import FEATURES
from reserve.models.base import tidy, vintage_state

KNOTS = (6, 12, 24, 36, 60)


def age_basis(age: np.ndarray) -> np.ndarray:
    """Linear spline in age (in years) with knots at 6, 12, 24, 36, 60 months, plus age <= 3."""
    a = np.asarray(age, dtype=float)
    cols = [a / 12] + [np.maximum(a - k, 0) / 12 for k in KNOTS] + [(a <= 3).astype(float)]
    return np.column_stack(cols)


def fit_logistic(
    x: np.ndarray, events: np.ndarray, trials: np.ndarray, ridge: float = 1e-3, iters: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Binomial logistic regression by Newton's method. Returns (coefficients, covariance).

    The first column is the intercept and is not penalized; the ridge keeps the fit finite
    when a column is nearly constant (for example age <= 3, where no event can happen).
    """
    n_feat = x.shape[1]
    beta = np.zeros(n_feat)
    rate = events.sum() / max(trials.sum(), 1)
    beta[0] = np.log(max(rate, 1e-9) / (1 - max(rate, 1e-9)))
    pen = np.full(n_feat, ridge * max(trials.sum(), 1.0) * 1e-3)
    pen[0] = 0.0
    hess = np.eye(n_feat)
    for _ in range(iters):
        eta = np.clip(x @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        grad = x.T @ (events - trials * p) - pen * beta
        w = trials * p * (1 - p)
        hess = (x * w[:, None]).T @ x + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta, np.linalg.inv(hess)


class HazardRegression:
    name = "C_hazard_macro"

    def __init__(
        self,
        macro: pl.DataFrame | None,
        features: Sequence[str] = FEATURES,
        exit_window: int = 24,
        draws: int = 200,
        seed: int = 0,
    ):
        self.macro = macro
        self.features = tuple(features) if macro is not None else ()
        self.exit_window = exit_window
        self.draws = draws
        self.seed = seed
        self.last_fit: dict = {}

    def describe(self) -> dict:
        revisions = None
        if self.macro is not None and "revisions_ignored" in self.macro.columns:
            revisions = bool(self.macro["revisions_ignored"].any())
        return {
            "name": self.name,
            "features": list(self.features),
            "knots": list(KNOTS),
            "exit_window": self.exit_window,
            "draws": self.draws,
            "seed": self.seed,
            "macro_revisions_ignored": revisions,
        }

    def _design(self, cells: pl.DataFrame) -> tuple[np.ndarray, pl.DataFrame]:
        if self.features:
            cells = cells.join(self.macro.select("period", *self.features), on="period", how="left")
            cells = cells.drop_nulls(list(self.features))
        x = age_basis(cells["age"].to_numpy())
        if self.features:
            x = np.column_stack([x, cells.select(self.features).to_numpy()])
        return np.column_stack([np.ones(len(x)), x]), cells

    def fit(self, history: pl.DataFrame) -> None:
        cells = history.filter(pl.col("at_risk") > 0)
        x, cells = self._design(cells)
        # Standardize non-intercept columns on the training cells only.
        mu = x[:, 1:].mean(axis=0)
        sd = x[:, 1:].std(axis=0)
        sd[sd == 0] = 1.0
        xs = np.column_stack([x[:, 0], (x[:, 1:] - mu) / sd])
        beta, cov = fit_logistic(xs, cells["new_events"].to_numpy(), cells["at_risk"].to_numpy())
        # Pooled monthly exit rate (prepayment and other exits) over the last `exit_window` months.
        last = history["period"].max()
        window = history.filter(pl.col("period").dt.offset_by(f"{self.exit_window}mo") > last)
        exits = window.select(
            (pl.col("exits_prepaid") + pl.col("exits_other") + pl.col("exits_unexplained")).sum()
        ).item()
        at_risk = window["at_risk"].sum()
        self.last_fit = {
            "beta": beta,
            "cov": cov,
            "mu": mu,
            "sd": sd,
            "exit_rate": float(exits / at_risk) if at_risk else 0.0,
            "n_cells": int(len(xs)),
        }

    def _x_future(self, ages: np.ndarray, origin: date) -> np.ndarray:
        x = age_basis(ages)
        if self.features:
            row = self.macro.filter(pl.col("period") == origin).select(self.features)
            if row.height == 0 or any(v is None for v in row.row(0)):
                raise ValueError(f"no macro features known at origin {origin}")
            x = np.column_stack(
                [x, np.repeat(np.array(row.row(0), dtype=float)[None, :], len(ages), 0)]
            )
        f = self.last_fit
        return np.column_stack([np.ones(len(ages)), (x - f["mu"]) / f["sd"]])

    def forecast(
        self,
        history: pl.DataFrame,
        origin: date,
        vintages: Sequence[str],
        horizon: int,
        quantiles: Sequence[float],
    ) -> dict[str, np.ndarray]:
        self.fit(history)
        f = self.last_fit
        state = vintage_state(history, origin, vintages)
        rng = np.random.default_rng([self.seed, zlib.crc32(origin.isoformat().encode())])
        chol = np.linalg.cholesky(f["cov"] + 1e-12 * np.eye(len(f["beta"])))
        betas = f["beta"] + rng.standard_normal((self.draws, len(f["beta"]))) @ chol.T
        out = {}
        for row in state.iter_rows(named=True):
            ages = row["age"] + np.arange(1, horizon + 1)
            xf = self._x_future(ages, origin)
            h = 1 / (1 + np.exp(-np.clip(betas @ xf.T, -30, 30)))  # (draws, horizon)
            leaving = row["new_events"] + row["exits_prepaid"] + row["exits_other"]
            alive = np.full(self.draws, max(row["at_risk"] - leaving - row["exits_unexplained"], 0))
            cum = np.zeros((self.draws, horizon))
            total = np.zeros(self.draws)
            q_exit = min(f["exit_rate"], 0.5)
            for k in range(horizon):
                ev = rng.binomial(alive, h[:, k])
                ex = rng.binomial(alive - ev, q_exit)
                alive = alive - ev - ex
                total = total + ev
                cum[:, k] = total
            paths = row["cum_rate"] + np.quantile(cum, quantiles, axis=0).T / row["n_orig"]
            out[row["vintage"]] = tidy(paths, row["cum_rate"])
        return out
