"""Harness options run on the development (validation) period only: thin history and
HYPOTHETICAL macro scenarios (PLAN.md FR8 and FR12; methodology M34 and M35).

Both reuse `harness.run` with the same Cleanroom origins as an ordinary run. Neither can
touch the sealed test period: `pipeline.run` refuses an experiment unless the config's
period is `dev`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import polars as pl
from cleanroom.stats import holm

from reserve.evaluation import harness, scoring
from reserve.evaluation.origins import Origin
from reserve.stress import scenarios as sc

KINDS = ("thin_history", "scenarios")
DEFAULT_THIN_MONTHS = (6, 12, 24)
DEFAULT_SCENARIOS = ("baseline", "adverse", "severe")
FULL = "full"


def run_thin_history(
    curves: pl.DataFrame,
    make_contestants: Callable[[str | None], list],
    origins: Sequence[Origin],
    horizon: int,
    quantiles: Sequence[float],
    step_months: int,
    min_history: int,
    months: Sequence[int] = DEFAULT_THIN_MONTHS,
    exclude=(),
) -> tuple[pl.DataFrame, dict]:
    """Score every contestant with the portfolio's history cut to each of `months`, and full.

    The forecast targets are the same at every level: vintages between `min_history` and
    min(`months`) months old at the origin. Only what else the contestants can see changes,
    so the levels form a learning curve. Holm's correction is applied across every
    comparison at every level together, and their number is reported (CLAUDE.md rule 8).
    """
    months = sorted(int(m) for m in months)
    if months[0] < min_history:
        raise ValueError(f"thin-history levels {months} must be at least min_history {min_history}")
    target_age = months[0]
    levels: list[int | None] = [*months, None]
    by_level, frames = {}, []
    for m in levels:
        key = FULL if m is None else str(m)
        preds = harness.run(
            curves,
            make_contestants(key),
            origins,
            horizon,
            quantiles,
            min_history,
            history_months=m,
            max_target_age=target_age,
        )
        by_level[key] = scoring.score(preds, quantiles, horizon, step_months, exclude)
        frames.append(preds.with_columns(pl.lit(key).alias("level")))
    joint = []
    for key, s in by_level.items():
        for c in s["comparison"]["comparisons"]:
            if c.get("dm") is not None:
                joint.append({"level": key, "contestant": c["contestant"], "p": c["dm"]["p_value"]})
    if joint:
        for row, p in zip(joint, holm([r["p"] for r in joint]), strict=True):
            row["holm_p_all_levels"] = float(p)
    curve = {}
    for key, s in by_level.items():
        for name, v in s["contestants"].items():
            curve.setdefault(name, {})[key] = v["mean_quantile_loss"]
    payload = {
        "levels": [FULL if m is None else str(m) for m in levels],
        "target_vintage_age": [min_history, target_age],
        "definition": (
            "history cut to vintages originated within the last N months before each origin "
            "(a portfolio N months old); targets are the same young vintages at every level"
        ),
        "by_level": by_level,
        "learning_curve": curve,
        "n_comparisons_all_levels": len(joint),
        "n_comparisons_tried_all_levels": sum(
            s["comparison"]["n_comparisons"] for s in by_level.values()
        ),
        "holm_all_levels": joint,
    }
    return pl.concat(frames, how="vertical_relaxed"), payload


def _increase_summary(preds: pl.DataFrame, quantiles: Sequence[float], horizon: int) -> dict:
    """Mean forecast increase over the horizon, per contestant, at the median and 90% ends."""
    at = preds.filter(pl.col("k") == horizon)
    out = {}
    for name in sorted(at["contestant"].unique().to_list()):
        g = at.filter(pl.col("contestant") == name)
        row = {"n_forecasts": g.height}
        for q in (min(quantiles), 0.5, max(quantiles)):
            if q in quantiles:
                inc = g[harness.qcol(q)] - g["c0"]
                row[f"mean_increase_{harness.qcol(q)}"] = float(inc.mean())
        out[name] = row
    return out


def run_scenarios(
    curves: pl.DataFrame,
    make_contestants: Callable[[str | None], list],
    origins: Sequence[Origin],
    horizon: int,
    quantiles: Sequence[float],
    min_history: int,
    names: Sequence[str] = DEFAULT_SCENARIOS,
) -> tuple[pl.DataFrame, dict]:
    """Conditional forecasts under each HYPOTHETICAL scenario; never scored against outcomes.

    Outcomes did not follow any scenario, so a loss against them would mean nothing. The
    result reports how far each contestant's forecasts move from the baseline scenario,
    and which contestants ignore the economy entirely (their forecasts do not move).
    """
    if "baseline" not in names:
        names = ("baseline", *names)
    frames, summary = [], {}
    by_scenario: dict[str, pl.DataFrame] = {}
    for name in names:
        sc.get(name)
        preds = harness.run(
            curves, make_contestants(name), origins, horizon, quantiles, min_history
        ).with_columns(pl.lit(name).alias("scenario"), pl.lit(sc.LABEL).alias("label"))
        by_scenario[name] = preds
        summary[name] = _increase_summary(preds, quantiles, horizon)
        frames.append(preds)
    qcols = [harness.qcol(q) for q in quantiles]
    base = by_scenario["baseline"]
    responds = {}
    for c in sorted(base["contestant"].unique().to_list()):
        b = base.filter(pl.col("contestant") == c).select(qcols).to_numpy()
        moved = [
            not np.allclose(by_scenario[n].filter(pl.col("contestant") == c).select(qcols), b)
            for n in names
            if n != "baseline"
        ]
        responds[c] = bool(any(moved))
    payload = {
        "label": sc.LABEL,
        "hypothetical": True,
        "disclaimer": sc.DISCLAIMER,
        "scenarios": {n: sc.get(n).describe() for n in names},
        "not_scored": "outcomes did not follow any scenario, so no loss is computed",
        "summary": summary,
        "responds_to_scenario": responds,
    }
    return pl.concat(frames, how="vertical_relaxed"), payload
