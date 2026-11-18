"""A synthetic loan portfolio with known answers, for tests and pipeline checks only.

Never used for a headline result (methodology M17). It writes loans in the exact Freddie
Mac file layout, so the whole pipeline (loader, vintage builder, data-fit check, models)
runs end to end without real data.

The true monthly chance that an event-free, still-active loan goes 90+ days late is

    h = exp(base + age(a) + season(p) + cohort(v) + beta * (unemployment(p) - 5))

for vintage v, vintage age a, and calendar month p, capped at 0.5. `age` is a hump that is
zero at `age_peak` and minus infinity below age 3 (a loan cannot be 90 days late sooner).
Loans also prepay with a fixed monthly chance. The expected cumulative curve under this
model is computed exactly (`truth`), so tests can check that the pipeline recovers it.

Macro series are written as FRED-style observation tables with publication dates and one
revision per unemployment value, so the as-of machinery is exercised too.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from reserve import months
from reserve.data import loans as loans_io

FOREVER = date(9999, 12, 31)


@dataclass(frozen=True)
class SimConfig:
    first_vintage: str = "2001Q1"
    n_vintages: int = 48
    loans_per_vintage: int = 300
    data_end: str = "2014-12"
    seed: int = 0
    base_log_hazard: float = -7.4
    age_peak: float = 30.0
    age_shape: float = 2.0
    cohort_sd: float = 0.25
    season_amplitude: float = 0.1
    macro_beta: float = 0.35
    prepay_monthly: float = 0.015
    cure_share: float = 0.3
    months_to_disposition: int = 6
    # Unemployment: 5% plus a recession bump that rises to +rise points, then decays.
    recession_start: str = "2008-01"
    recession_peak: str = "2009-10"
    recession_end: str = "2012-06"
    recession_rise: float = 5.0
    # Optional forbearance window: extra events flagged with Borrower Assistance Plan "F".
    forbearance_start: str | None = None
    forbearance_end: str | None = None
    forbearance_log_hazard_boost: float = 1.5
    extra: dict = field(default_factory=dict)

    def vintages(self) -> list[str]:
        first = months.quarter_end(self.first_vintage)
        return [months.vintage_of(months.add(first, 3 * i)) for i in range(self.n_vintages)]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Simulation:
    config: SimConfig
    orig: pl.DataFrame
    perf: pl.DataFrame
    truth: pl.DataFrame  # (vintage, age, period, hazard, prepay, expected_cum_rate)
    loan_events: pl.DataFrame  # (loan_id, vintage, event_age or null)
    macro: dict[str, pl.DataFrame]  # FRED-style observation tables


def unemployment(cfg: SimConfig, month: date) -> float:
    m = months.index(month)
    s, p, e = (
        months.index(months.parse(x))
        for x in (cfg.recession_start, cfg.recession_peak, cfg.recession_end)
    )
    if s <= m <= p:
        bump = cfg.recession_rise * (m - s) / max(p - s, 1)
    elif p < m <= e:
        bump = cfg.recession_rise * (e - m) / max(e - p, 1)
    else:
        bump = 0.0
    return 5.0 + bump


def _age_effect(cfg: SimConfig, age: np.ndarray) -> np.ndarray:
    r = age / cfg.age_peak
    with np.errstate(divide="ignore"):
        out = cfg.age_shape * (np.log(r) - r + 1.0)
    return np.where(age < 3, -np.inf, out)


def _in_forbearance(cfg: SimConfig, month_idx: np.ndarray) -> np.ndarray:
    if cfg.forbearance_start is None or cfg.forbearance_end is None:
        return np.zeros_like(month_idx, dtype=bool)
    lo = months.index(months.parse(cfg.forbearance_start))
    hi = months.index(months.parse(cfg.forbearance_end))
    return (month_idx >= lo) & (month_idx <= hi)


def true_hazard(cfg: SimConfig, cohort: float, ages: np.ndarray, month_idx: np.ndarray):
    """True monthly event chance and its forbearance part, for one vintage."""
    season = cfg.season_amplitude * np.cos(2 * np.pi * (month_idx % 12) / 12)
    u = np.array([unemployment(cfg, months.from_index(int(m))) for m in month_idx])
    log_h = cfg.base_log_hazard + _age_effect(cfg, ages) + season + cohort
    log_h = log_h + cfg.macro_beta * (u - 5.0)
    forb = _in_forbearance(cfg, month_idx)
    base = np.minimum(np.exp(log_h), 0.5)
    total = np.minimum(np.exp(log_h + cfg.forbearance_log_hazard_boost * forb), 0.5)
    return total, total - base


def simulate(cfg: SimConfig) -> Simulation:
    rng = np.random.default_rng(cfg.seed)
    end_idx = months.index(months.parse(cfg.data_end))
    orig_rows, perf_parts, truth_parts, event_rows = [], [], [], []
    for vi, vintage in enumerate(cfg.vintages()):
        qend = months.index(months.quarter_end(vintage))
        max_age = end_idx - qend
        if max_age < 1:
            continue
        cohort = rng.normal(0.0, cfg.cohort_sd)
        ages = np.arange(1, max_age + 1)
        month_idx = qend + ages
        hazard, forb_part = true_hazard(cfg, cohort, ages, month_idx)
        prepay = np.full_like(hazard, cfg.prepay_monthly)
        survive = np.cumprod(1.0 - hazard - prepay)
        at_risk_before = np.concatenate([[1.0], survive[:-1]])
        expected = np.cumsum(at_risk_before * hazard)
        truth_parts.append(
            pl.DataFrame(
                {
                    "vintage": vintage,
                    "age": ages,
                    "period": [months.from_index(int(m)) for m in month_idx],
                    "hazard": hazard,
                    "prepay": prepay,
                    "expected_cum_rate": expected,
                }
            )
        )

        n = cfg.loans_per_vintage
        yy = int(vintage[2:4])
        q = vintage[-1]
        ids = [f"F{yy:02d}Q{q}{vi:03d}{i:04d}" for i in range(n)]
        # Monthly draws: event first, then prepayment, for loans still active and event-free.
        u = rng.random((n, max_age))
        event_age = np.full(n, -1)
        exit_age = np.full(n, -1)
        active = np.ones(n, dtype=bool)
        for k in range(max_age):
            ev = active & (u[:, k] < hazard[k])
            pp = active & ~ev & (u[:, k] < hazard[k] + prepay[k])
            event_age[ev] = k + 1
            exit_age[pp] = k + 1
            active &= ~(ev | pp)
        cures = rng.random(n) < cfg.cure_share
        later_prepay = rng.geometric(cfg.prepay_monthly, n)
        forb_flag = np.zeros(n, dtype=bool)
        has_event = event_age > 0
        if has_event.any():
            share = forb_part[event_age[has_event] - 1] / hazard[event_age[has_event] - 1]
            forb_flag[has_event] = rng.random(has_event.sum()) < share

        for i, lid in enumerate(ids):
            e, x = int(event_age[i]), int(exit_age[i])
            orig_rows.append(_orig_row(lid, vintage, qend, rng))
            event_rows.append(
                {"loan_id": lid, "vintage": vintage, "event_age": e if e > 0 else None}
            )
            perf_parts.append(
                _perf_rows(
                    lid,
                    qend,
                    max_age,
                    e,
                    x,
                    bool(cures[i]),
                    int(later_prepay[i]),
                    bool(forb_flag[i]),
                    cfg,
                )
            )

    orig = pl.DataFrame(orig_rows)
    perf = pl.concat(perf_parts)
    truth = pl.concat(truth_parts)
    loan_events = pl.DataFrame(event_rows, schema_overrides={"event_age": pl.Int64})
    return Simulation(cfg, orig, perf, truth, loan_events, synthetic_macro(cfg, rng))


def _orig_row(lid: str, vintage: str, qend: int, rng: np.random.Generator) -> dict:
    first_pay = months.from_index(qend + 1)
    return {
        "loan_id": lid,
        "vintage": vintage,
        "credit_score": int(rng.integers(620, 820)),
        "first_payment_date": first_pay,
        "maturity_date": months.from_index(qend + 360),
        "orig_upb": float(rng.integers(80, 400) * 1000),
        "orig_ltv": int(rng.integers(60, 96)),
        "orig_interest_rate": round(float(rng.uniform(3.0, 7.5)), 3),
        "orig_loan_term": 360,
        "amortization_type": "FRM",
        "loan_purpose": "P",
        "occupancy_status": "P",
        "property_state": "ZZ",  # not a real state: marks the record as synthetic
        "seller_name": "SYNTHETIC",
        "vantage_score_4": 9999,
    }


def _perf_rows(
    lid: str,
    qend: int,
    max_age: int,
    event: int,
    prepay_exit: int,
    cures: bool,
    later_prepay: int,
    forbearance: bool,
    cfg: SimConfig,
) -> pl.DataFrame:
    """Monthly rows from age 1 to the loan's exit or the data end."""
    dlq = []
    zbc: list[int | None] = []
    if event > 0:
        if cures:
            last = min(max_age, event + 1 + later_prepay)
        else:
            last = min(max_age, event + cfg.months_to_disposition)
    elif prepay_exit > 0:
        last = prepay_exit
    else:
        last = max_age
    for a in range(1, last + 1):
        if event > 0 and event - 2 <= a <= event:
            d = 3 - (event - a)
        elif event > 0 and a > event and not cures:
            d = 3 + (a - event)
        else:
            d = 0
        dlq.append(f"{d:02d}")
        zbc.append(None)
    if event > 0 and not cures and last == event + cfg.months_to_disposition:
        zbc[-1] = 9  # REO disposition
    elif (event > 0 and cures and last == event + 1 + later_prepay) or (
        event <= 0 and prepay_exit > 0
    ):
        zbc[-1] = 1  # prepaid
    ages = list(range(1, last + 1))
    periods = [months.from_index(qend + a) for a in ages]
    bap = ["F" if forbearance and a == event else None for a in ages]
    return pl.DataFrame(
        {
            "loan_id": lid,
            "period": periods,
            "current_upb": [0.0 if z is not None else 100000.0 for z in zbc],
            "dlq_status": dlq,
            "loan_age": ages,
            "zero_balance_code": zbc,
            "zero_balance_date": [
                p if z is not None else None for p, z in zip(periods, zbc, strict=True)
            ],
            "borrower_assistance_plan": bap,
            "servicer_name": "SYNTHETIC",
        },
        schema_overrides={
            "zero_balance_code": pl.Int64,
            "zero_balance_date": pl.Date,
            "borrower_assistance_plan": pl.String,
        },
    )


def synthetic_macro(cfg: SimConfig, rng: np.random.Generator) -> dict[str, pl.DataFrame]:
    """FRED-style tables. Unemployment is first published with noise, then revised a month later."""
    start = months.index(months.quarter_end(cfg.first_vintage)) - 24
    end = months.index(months.parse(cfg.data_end))
    rows = {"UNRATE": [], "MORTGAGE30US": [], "HPIPONM226S": []}
    hpi = 200.0
    for m in range(start, end + 1):
        d = months.from_index(m)
        u = unemployment(cfg, d)
        first_release = months.add(d, 1).replace(day=7)
        revised = months.add(d, 2).replace(day=7)
        noisy = round(u + float(rng.normal(0, 0.1)), 1)
        rows["UNRATE"].append((d, noisy, first_release, revised - timedelta(days=1)))
        rows["UNRATE"].append((d, round(u, 1), revised, FOREVER))
        in_rec = months.parse(cfg.recession_start) <= d <= months.parse(cfg.recession_peak)
        hpi *= 1.0 - 0.006 if in_rec else 1.003
        rows["HPIPONM226S"].append((d, round(hpi, 2), months.add(d, 2).replace(day=25), FOREVER))
        rate = 6.5 - 3.0 * (m - start) / max(end - start, 1)
        thursday = d + timedelta(days=(3 - d.weekday()) % 7)
        while thursday.month == d.month:
            rows["MORTGAGE30US"].append((thursday, round(rate, 2), thursday, FOREVER))
            thursday += timedelta(days=7)
    out = {}
    for sid, r in rows.items():
        out[sid] = pl.DataFrame(
            r, schema=["date", "value", "realtime_start", "realtime_end"], orient="row"
        ).select(pl.lit(sid).alias("series_id"), pl.all())
    return out


def write(sim: Simulation, out_dir: str | Path) -> dict[str, list[Path]]:
    """Write loans as sample_orig_YYYY.txt / sample_perf_YYYY.txt, plus macro and truth Parquet."""
    out_dir = Path(out_dir)
    loan_dir = out_dir / "freddie"
    macro_dir = out_dir / "fred"
    loan_dir.mkdir(parents=True, exist_ok=True)
    macro_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {"loans": [], "macro": [], "truth": []}
    for year in sorted(sim.orig["vintage"].str.slice(0, 4).unique().to_list()):
        o = sim.orig.filter(pl.col("vintage").str.starts_with(year))
        p = sim.perf.filter(pl.col("loan_id").is_in(o["loan_id"].to_list()))
        po, pp = loan_dir / f"sample_orig_{year}.txt", loan_dir / f"sample_perf_{year}.txt"
        loans_io.write_file(o, po, "orig")
        loans_io.write_file(p, pp, "perf")
        written["loans"] += [po, pp]
    for sid, df in sim.macro.items():
        path = macro_dir / f"{sid}.parquet"
        df.write_parquet(path)
        written["macro"].append(path)
    truth_path = out_dir / "truth.parquet"
    sim.truth.write_parquet(truth_path)
    written["truth"].append(truth_path)
    return written
