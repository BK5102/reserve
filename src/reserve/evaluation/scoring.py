"""Scores, from the harness's long table, using the shared Cleanroom forecast tools (M29).

All losses are computed on the change since the origin (y - c0 against q - c0), so the part
of the curve already known does not dilute them. Pinball loss is shift invariant, so for it
this equals scoring the level.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl
from cleanroom import forecast as cf
from cleanroom.stats import diebold_mariano, holm

from reserve.evaluation.harness import qcol

CLASSICAL = ("A_last_increment", "B_age_curve", "C_hazard_macro")


def scoreable(
    preds: pl.DataFrame, exclude: Sequence[tuple[date, date]] = (), horizon: int | None = None
) -> pl.DataFrame:
    """Rows with an observed outcome, outside every excluded target window, at one horizon."""
    out = preds.filter(pl.col("y").is_not_null())
    for lo, hi in exclude:
        out = out.filter(~pl.col("target_period").is_between(lo, hi))
    if horizon is not None:
        out = out.filter(pl.col("k") == horizon)
    return out


def mean_quantile_loss(df: pl.DataFrame, quantiles: Sequence[float]) -> float:
    """Pinball loss averaged over items and quantiles."""
    y = df["y"].to_numpy()
    return float(np.mean([cf.pinball_loss(y, df[qcol(q)].to_numpy(), q) for q in quantiles]))


def _coverage(df: pl.DataFrame, lo_q: float, hi_q: float) -> dict:
    cov = cf.interval_coverage(
        df["y"].to_numpy(), df[qcol(lo_q)].to_numpy(), df[qcol(hi_q)].to_numpy()
    )
    return {"nominal": round(hi_q - lo_q, 4), **cov.to_dict()}


def contestant_scores(df: pl.DataFrame, quantiles: Sequence[float]) -> dict:
    inc_y = (df["y"] - df["c0"]).to_numpy()
    inc_q = np.column_stack([(df[qcol(q)] - df["c0"]).to_numpy() for q in quantiles])
    wql = cf.weighted_quantile_loss(inc_y, inc_q, quantiles) if np.abs(inc_y).sum() > 0 else None
    out = {
        "n": df.height,
        "mean_quantile_loss": mean_quantile_loss(df, quantiles),
        "wql_on_change": wql,
        "median_abs_error": float((df["y"] - df[qcol(0.5)]).abs().mean())
        if 0.5 in quantiles
        else None,
    }
    if {0.25, 0.75} <= set(quantiles):
        out["coverage_50"] = _coverage(df, 0.25, 0.75)
    if {0.05, 0.95} <= set(quantiles):
        out["coverage_90"] = _coverage(df, 0.05, 0.95)
        out["interval_score_90"] = cf.interval_score(
            df["y"].to_numpy(), df[qcol(0.05)].to_numpy(), df[qcol(0.95)].to_numpy(), alpha=0.1
        )
    return out


def per_origin_losses(df: pl.DataFrame, quantiles: Sequence[float]) -> pl.DataFrame:
    """Mean quantile loss per (contestant, origin), in origin order."""
    rows = []
    for (name, origin), g in df.group_by("contestant", "origin"):
        rows.append(
            {"contestant": name, "origin": origin, "loss": mean_quantile_loss(g, quantiles)}
        )
    return pl.DataFrame(rows).sort("contestant", "origin")


def compare(df: pl.DataFrame, quantiles: Sequence[float], horizon: int, step_months: int) -> dict:
    """Each contestant against the best classical baseline in this table (M29).

    The reference is the classical contestant with the lowest overall mean quantile loss.
    Relative error is the geometric mean over origins of loss ratios; the Diebold-Mariano
    test uses per-origin losses with lags up to ceil(horizon / step) - 1, because forecasts
    from origins closer than the horizon share outcome months. Holm adjusts the p-values
    for the number of comparisons, which is reported.
    """
    names = sorted(df["contestant"].unique().to_list())
    overall = {
        n: mean_quantile_loss(df.filter(pl.col("contestant") == n), quantiles) for n in names
    }
    classical = [n for n in names if n in CLASSICAL]
    if not classical:
        return {"reference": None, "comparisons": []}
    ref = min(classical, key=lambda n: overall[n])
    po = per_origin_losses(df, quantiles)
    ref_loss = po.filter(pl.col("contestant") == ref).select("origin", pl.col("loss").alias("ref"))
    lag_h = max(1, math.ceil(horizon / step_months))
    comps = []
    for n in names:
        if n == ref:
            continue
        pair = po.filter(pl.col("contestant") == n).join(ref_loss, on="origin").sort("origin")
        entry = {"contestant": n, "n_origins": pair.height}
        if (pair["ref"] > 0).all() and pair.height:
            entry["relative_error"] = cf.relative_error(
                pair["loss"].to_numpy(), pair["ref"].to_numpy()
            )
        else:
            entry["relative_error"] = None
        try:
            dm = diebold_mariano(pair["loss"].to_numpy(), pair["ref"].to_numpy(), horizon=lag_h)
            entry["dm"] = dm.to_dict()
        except ValueError as err:
            entry["dm"] = None
            entry["dm_error"] = str(err)
        comps.append(entry)
    tested = [c for c in comps if c["dm"] is not None]
    if tested:
        adj = holm([c["dm"]["p_value"] for c in tested])
        for c, p in zip(tested, adj, strict=True):
            c["holm_p"] = float(p)
    return {
        "reference": ref,
        "overall_mean_quantile_loss": overall,
        "dm_lag_horizon": lag_h,
        "n_comparisons": len(comps),
        "comparisons": comps,
    }


def score(
    preds: pl.DataFrame,
    quantiles: Sequence[float],
    headline_horizon: int,
    step_months: int,
    exclude: Sequence[tuple[date, date]] = (),
) -> dict:
    """Headline scores at one horizon, plus losses by horizon for every contestant."""
    head = scoreable(preds, exclude, headline_horizon)
    allh = scoreable(preds, exclude)
    if head.height == 0:
        raise ValueError("no scoreable rows at the headline horizon")
    by_contestant = {
        n: contestant_scores(head.filter(pl.col("contestant") == n), quantiles)
        for n in sorted(head["contestant"].unique().to_list())
    }
    by_horizon = {}
    for (n, k), g in allh.group_by("contestant", "k"):
        by_horizon.setdefault(n, {})[int(k)] = mean_quantile_loss(g, quantiles)
    return {
        "headline_horizon": headline_horizon,
        "n_rows_headline": head.height,
        "excluded_target_windows": [[lo.isoformat(), hi.isoformat()] for lo, hi in exclude],
        "contestants": by_contestant,
        "loss_by_horizon": {n: dict(sorted(v.items())) for n, v in by_horizon.items()},
        "comparison": compare(head, quantiles, headline_horizon, step_months),
    }
