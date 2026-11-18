"""A simple drift monitor with fixed thresholds (PLAN.md FR13; methodology M36).

Three checks, each green, amber, or red:

1. Coverage: the share of recent outcomes inside the 90% interval. Red when even the top of
   its Wilson interval is below 90% (too few outcomes inside to be bad luck, if forecasts
   were independent); amber when the share itself is more than 5 points below 90%.
2. Loss ratio: recent mean quantile loss divided by the loss over a reference (backtest)
   window. Amber above 1.5, red above 2.
3. Macro range: each macro feature at a month against its values in the training months.
   Amber outside the central 90% of training values, red outside their minimum and maximum,
   where contestant C has to extrapolate.

"Recent" is the last `window` forecast origins whose outcomes have arrived. Every check
reads only rows that would be known at the time; outcomes are matched by `known_at`, the
month the outcome arrives (origin plus horizon).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import polars as pl
from cleanroom import forecast as cf

from reserve import months
from reserve.evaluation.harness import qcol
from reserve.evaluation.scoring import mean_quantile_loss

GREEN, AMBER, RED = "green", "amber", "red"
ORDER = {GREEN: 0, AMBER: 1, RED: 2}

COVERAGE_NOMINAL = 0.9
COVERAGE_AMBER_MARGIN = 0.05
LOSS_AMBER, LOSS_RED = 1.5, 2.0
MACRO_AMBER_QUANTILES = (0.05, 0.95)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    value: float | None
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def worst(statuses: Sequence[str]) -> str:
    return max(statuses, key=ORDER.__getitem__) if statuses else GREEN


def coverage_check(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal: float = COVERAGE_NOMINAL,
    amber_margin: float = COVERAGE_AMBER_MARGIN,
) -> Check:
    cov = cf.interval_coverage(y, lower, upper)
    if cov.ci_high < nominal:
        status = RED
    elif cov.coverage < nominal - amber_margin:
        status = AMBER
    else:
        status = GREEN
    detail = (
        f"{cov.n_inside} of {cov.n} inside; Wilson interval [{cov.ci_low:.3f}, {cov.ci_high:.3f}] "
        f"against nominal {nominal:g}"
    )
    return Check("coverage_90", status, cov.coverage, detail)


def loss_ratio_check(
    recent_loss: float, reference_loss: float, amber: float = LOSS_AMBER, red: float = LOSS_RED
) -> Check:
    if reference_loss <= 0:
        return Check("loss_ratio", AMBER, None, "reference loss is zero; ratio undefined")
    ratio = recent_loss / reference_loss
    status = RED if ratio > red else AMBER if ratio > amber else GREEN
    return Check("loss_ratio", status, ratio, f"amber above {amber:g}, red above {red:g}")


def macro_range_check(
    features: pl.DataFrame, train_until: date, at: date, names: Sequence[str]
) -> list[Check]:
    """Each feature at month `at` against its values in months up to `train_until`."""
    train = features.filter(pl.col("period") <= train_until)
    row = features.filter(pl.col("period") == at)
    if row.height == 0:
        raise ValueError(f"no macro features for {at}")
    out = []
    lo_q, hi_q = MACRO_AMBER_QUANTILES
    for f in names:
        vals = train[f].drop_nulls().to_numpy()
        v = row[f].item()
        if v is None or len(vals) == 0:
            out.append(Check(f"macro_{f}", AMBER, None, "no value or no training values"))
            continue
        lo, hi = float(vals.min()), float(vals.max())
        plo, phi = (float(x) for x in np.quantile(vals, [lo_q, hi_q]))
        status = RED if not lo <= v <= hi else AMBER if not plo <= v <= phi else GREEN
        detail = f"training range [{lo:.4g}, {hi:.4g}], central 90% [{plo:.4g}, {phi:.4g}]"
        out.append(Check(f"macro_{f}", status, float(v), detail))
    return out


def monitor_predictions(
    preds: pl.DataFrame,
    contestant: str,
    reference_end: date,
    quantiles: Sequence[float],
    horizon: int,
    window: int = 4,
) -> pl.DataFrame:
    """Coverage and loss-ratio status for every origin after `reference_end`.

    The reference loss comes from origins up to `reference_end`. Each later origin is
    judged on itself and the `window - 1` origins before it (all after the reference).
    `known_at` is the month the newest of those outcomes arrives.
    """
    lo_q, hi_q = min(quantiles), max(quantiles)
    rows = preds.filter(
        (pl.col("contestant") == contestant) & (pl.col("k") == horizon) & pl.col("y").is_not_null()
    )
    ref = rows.filter(pl.col("origin") <= reference_end)
    if ref.height == 0:
        raise ValueError(f"no reference rows for {contestant} up to {reference_end}")
    ref_loss = mean_quantile_loss(ref, quantiles)
    later = sorted(o for o in rows["origin"].unique().to_list() if o > reference_end)
    out = []
    for i, o in enumerate(later):
        recent_origins = later[max(0, i - window + 1) : i + 1]
        g = rows.filter(pl.col("origin").is_in(recent_origins))
        cov = coverage_check(g["y"].to_numpy(), g[qcol(lo_q)].to_numpy(), g[qcol(hi_q)].to_numpy())
        loss = loss_ratio_check(mean_quantile_loss(g, quantiles), ref_loss)
        out.append(
            {
                "origin": o,
                "known_at": months.add(o, horizon),
                "n": g.height,
                "coverage_90": cov.value,
                "coverage_status": cov.status,
                "loss_ratio": loss.value,
                "loss_status": loss.status,
                "status": worst([cov.status, loss.status]),
            }
        )
    return pl.DataFrame(out)
