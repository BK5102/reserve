"""Data-fit checks: can this data answer Reserve's question at all? (methodology M19 to M21)

Run on real Freddie Mac sample files once they are downloaded, or on synthetic data to
prove the checks work. Every output carries `data_source`; a synthetic run says so.

Checks: files and layouts; loans per origination quarter; events per vintage-age cell
against binomial noise, including whether a simple class-average forecast is already at
the noise floor; recession coverage; the COVID forbearance trap; count reconciliation;
power as a minimum detectable effect; the latest month available; and, if given, macro
series coverage. The gate from NEXT_PROJECTS_PLAN.md: at least 60 usable vintages and at
least 20 events per vintage by age 12.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

from reserve import months
from reserve.data import loans as loans_io
from reserve.data import vintages

# NBER US business cycle dates: peak December 2007, trough June 2009
# (https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions).
RECESSION = (date(2007, 12, 1), date(2009, 6, 1))
COVID = (date(2020, 3, 1), date(2021, 12, 1))


@dataclass(frozen=True)
class FitSettings:
    check_age: int = 12
    horizon: int = 12
    min_events: int = 20
    min_vintages: int = 60
    usable_min_age: int = 24  # 12 months of history plus a 12-month forecast
    class_window: int = 4  # vintages averaged by the class-average noise-floor check
    test_origins: tuple[str, str] = ("2013-03", "2018-12")  # NEXT_PROJECTS_PLAN.md
    origin_step_months: int = 3
    n_comparisons: int = 3
    alpha: float = 0.05
    power: float = 0.8


def load_freddie(raw_dir: str | Path) -> tuple[pl.DataFrame, pl.DataFrame, list[dict]]:
    """All sample years in raw_dir, with a per-file layout report."""
    files = loans_io.find_sample_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"no sample_orig_YYYY.txt / sample_perf_YYYY.txt pairs in {raw_dir}"
        )
    origs, perfs, report = [], [], []
    for year, o, p in files:
        orig = loans_io.read_origination(
            o, columns=("first_payment_date", *loans_io.MARKER_COLUMNS)
        )
        perf = loans_io.read_performance(p)
        report.append(
            {
                "year": year,
                "orig_file": o.name,
                "orig_layout": loans_io.detect_layout(o, "orig")[0],
                "orig_rows": orig.height,
                "perf_file": p.name,
                "perf_layout": loans_io.detect_layout(p, "perf")[0],
                "perf_rows": perf.height,
            }
        )
        origs.append(orig)
        perfs.append(perf)
    return pl.concat(origs, how="diagonal"), pl.concat(perfs, how="diagonal"), report


def _summary(values: pl.Series) -> dict:
    if values.len() == 0:
        return {"n": 0}
    return {
        "n": int(values.len()),
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def noise_floor(curves: pl.DataFrame, s: FitSettings) -> dict:
    """Is a class-average forecast of the next 12 months already at the binomial noise floor?

    For each vintage with ages check_age and check_age + horizon observed, the target is the
    increment c(a + h) - c(a). The forecast is the mean increment of the previous
    `class_window` vintages. The noise is the binomial standard deviation of the increment.
    A ratio of forecast error to noise near 1 means no model can do much better.
    """
    a, h = s.check_age, s.horizon
    if not {a, a + h} <= set(curves["age"].unique().to_list()):
        return {"n_vintages": 0, "ratio": None}
    wide = (
        curves.filter(pl.col("age").is_in([a, a + h]))
        .pivot(on="age", index=["vintage", "n_orig"], values="cum_rate")
        .rename({str(a): "c0", str(a + h): "c1"})
        .drop_nulls()
        .sort("vintage")
        .with_columns((pl.col("c1") - pl.col("c0")).alias("inc"))
    )
    if wide.height <= s.class_window:
        return {"n_vintages": wide.height, "ratio": None}
    inc = wide["inc"].to_numpy()
    n = wide["n_orig"].to_numpy()
    c0 = wide["c0"].to_numpy()
    errors, noise_var = [], []
    for i in range(s.class_window, len(inc)):
        forecast = inc[i - s.class_window : i].mean()
        errors.append(inc[i] - forecast)
        # Binomial variance of the increment, given the curve level at the start.
        p = inc[i]
        noise_var.append(max(p * (1 - p - c0[i]), 0.0) / n[i])
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    noise = float(np.sqrt(np.mean(noise_var)))
    return {
        "n_vintages": int(len(errors)),
        "class_average_rmse": rmse,
        "binomial_noise_sd": noise,
        "ratio": rmse / noise if noise > 0 else None,
    }


def recession_coverage(data: vintages.VintageData) -> dict:
    lo, hi = RECESSION
    curves = data.curves
    in_window = curves.filter((pl.col("period") >= lo) & (pl.col("period") <= hi))
    pre = curves.filter(pl.col("period") < lo).group_by("vintage").agg(pl.col("age").max())
    seasoned = pre.filter(pl.col("age") >= 12).height
    ev = data.loans.filter((pl.col("event_period") >= lo) & (pl.col("event_period") <= hi)).height
    return {
        "window": [lo.isoformat(), hi.isoformat()],
        "vintages_observed_in_window": int(in_window["vintage"].n_unique()),
        "vintages_with_12_months_before_window": int(seasoned),
        "events_in_window": int(ev),
    }


def covid_check(data: vintages.VintageData) -> dict:
    """Events in 2020-03 to 2021-12, the share flagged as forbearance, and the spike size."""
    lo, hi = COVID
    loans = data.loans.filter(pl.col("event_period").is_not_null())
    window = loans.filter((pl.col("event_period") >= lo) & (pl.col("event_period") <= hi))
    by_month = loans.group_by("event_period").len()
    base = by_month.filter(pl.col("event_period").dt.year() == 2019)["len"]
    peak = by_month.filter((pl.col("event_period") >= lo) & (pl.col("event_period") <= hi))["len"]
    return {
        "window": [lo.isoformat(), hi.isoformat()],
        "events_in_window": int(window.height),
        "share_flagged_forbearance": (
            float(window["event_in_forbearance"].mean()) if window.height else None
        ),
        "monthly_events_2019_mean": float(base.mean()) if base.len() else None,
        "monthly_events_window_max": int(peak.max()) if peak.len() else None,
    }


def power(data: vintages.VintageData, s: FitSettings) -> dict:
    """Minimum detectable effect for the headline comparison, in standard-deviation units.

    MDE = (z(1 - alpha / (2 m)) + z(power)) / sqrt(n), with m comparisons (Bonferroni, a
    conservative stand-in for Holm). n is reported two ways: every test origin, and only
    non-overlapping 12-month windows (forecasts from origins less than a horizon apart share
    outcome months, so they are not independent).
    """
    lo, hi = (months.parse(x) for x in s.test_origins)
    origins = [
        months.from_index(m)
        for m in range(months.index(lo), months.index(hi) + 1, s.origin_step_months)
    ]
    cutoff = data.cutoff
    curves = data.curves
    pairs = 0
    vint: set[str] = set()
    for t in origins:
        target = months.add(t, s.horizon)
        if target > cutoff:
            continue
        at_t = curves.filter(pl.col("period") == t)
        at_target = curves.filter(pl.col("period") == target).select("vintage")
        ok = at_t.join(at_target, on="vintage")["vintage"].to_list()
        pairs += len(ok)
        vint.update(ok)
    usable_origins = sum(1 for t in origins if months.add(t, s.horizon) <= cutoff)
    span = months.index(hi) - months.index(lo) + s.horizon
    n_independent = max(span // s.horizon, 0)
    z = NormalDist().inv_cdf(1 - s.alpha / (2 * s.n_comparisons)) + NormalDist().inv_cdf(s.power)

    def mde(n: int):
        return z / np.sqrt(n) if n > 0 else None

    return {
        "test_origins": [lo.isoformat(), hi.isoformat()],
        "origins_with_targets": usable_origins,
        "vintage_origin_pairs": pairs,
        "distinct_vintages": len(vint),
        "independent_windows": int(n_independent),
        "mde_sd_units_all_origins": mde(usable_origins),
        "mde_sd_units_independent_windows": mde(n_independent),
        "comparisons_assumed": s.n_comparisons,
    }


def macro_coverage(observations: dict[str, pl.DataFrame]) -> dict:
    out = {}
    for sid, obs in observations.items():
        out[sid] = {
            "first_observation": obs["date"].min().isoformat(),
            "last_observation": obs["date"].max().isoformat(),
            "n_vintages": int(obs["realtime_start"].n_unique()),
            "first_vintage": obs["realtime_start"].min().isoformat(),
            "missing_values": int(obs["value"].null_count()),
        }
    return out


def check(
    orig: pl.DataFrame,
    perf: pl.DataFrame,
    data_source: str,
    settings: FitSettings | None = None,
    file_report: list[dict] | None = None,
    macro_observations: dict[str, pl.DataFrame] | None = None,
) -> dict:
    s = settings or FitSettings()
    marked = [c for c in loans_io.MARKER_COLUMNS if c in orig.columns]
    n_marked = loans_io.synthetic_records(orig) if marked else 0
    data = vintages.build(orig, perf)
    curves = data.curves
    counts = vintages.reconcile(
        data.loans, curves, orig, perf.filter(pl.col("period") <= data.cutoff)
    )
    counts["loans_first_seen_after_age_1"] = int(
        data.loans.filter(pl.col("first_obs_age") > 1).height
    )
    per_q = data.loans.group_by("vintage").len().sort("vintage")
    max_age = curves.group_by("vintage").agg(pl.col("age").max().alias("max_age"))
    usable = max_age.filter(pl.col("max_age") >= s.usable_min_age)
    at_age = curves.filter(
        (pl.col("age") == s.check_age) & pl.col("vintage").is_in(usable["vintage"].to_list())
    )
    cum = at_age["cum_events"]
    rate = at_age["cum_rate"].to_numpy()
    n = at_age["n_orig"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_se = np.where(rate > 0, np.sqrt(rate * (1 - rate) / n) / rate, np.nan)
    median_events = float(cum.median()) if cum.len() else 0.0
    gate = {
        "usable_vintages": int(usable.height),
        "usable_vintages_needed": s.min_vintages,
        "median_events_by_age": median_events,
        "events_needed": s.min_events,
        "share_vintages_meeting_events": float((cum >= s.min_events).mean()) if cum.len() else 0.0,
    }
    gate["passes"] = bool(
        gate["usable_vintages"] >= s.min_vintages and median_events >= s.min_events
    )
    report = {
        "data_source": data_source,
        # Synthetic if it came from the simulator, or if any record carries its markers,
        # even when read from a "freddie" folder (M40).
        "synthetic": data_source.startswith("synthetic") or n_marked > 0,
        "synthetic_records": n_marked,
        "settings": {k: v for k, v in s.__dict__.items()},
        "files": file_report or [],
        "latest": {
            "last_performance_month": data.cutoff.isoformat(),
            "first_vintage": per_q["vintage"].min(),
            "last_vintage": per_q["vintage"].max(),
        },
        "loans_per_quarter": {
            "n_vintages": int(per_q.height),
            **{k: v for k, v in _summary(per_q["len"]).items() if k != "n"},
        },
        "events_at_check_age": {
            "age": s.check_age,
            "cumulative_events": _summary(cum),
            "new_events_in_that_month": _summary(at_age["new_events"]),
            "cumulative_rate": _summary(at_age["cum_rate"]),
            "relative_binomial_se_median": (
                float(np.nanmedian(rel_se)) if np.isfinite(rel_se).any() else None
            ),
        },
        "noise_floor": noise_floor(curves, s),
        "recession": recession_coverage(data),
        "covid": covid_check(data),
        "reconciliation": counts,
        "power": power(data, s),
        "gate": gate,
    }
    if macro_observations is not None:
        report["macro"] = macro_coverage(macro_observations)
    return report


def _f(x) -> str:
    """Three significant digits, or 'n/a'."""
    if x is None:
        return "n/a"
    return f"{x:.3g}" if isinstance(x, float) else str(x)


def to_markdown(report: dict) -> str:
    """A short human-readable summary of a report."""
    lines = []
    if report["synthetic"]:
        lines += ["SYNTHETIC DATA: these numbers test the checks, not the real dataset.", ""]
    g, lq, ea = report["gate"], report["loans_per_quarter"], report["events_at_check_age"]
    nf, r, c, p = report["noise_floor"], report["recession"], report["covid"], report["power"]
    lines += [
        f"Data source: {report['data_source']}",
        f"Last performance month: {report['latest']['last_performance_month']}",
        f"Vintages: {lq['n_vintages']} ({report['latest']['first_vintage']} to "
        f"{report['latest']['last_vintage']}); loans per quarter median {_f(lq.get('median'))}",
        f"Events by age {ea['age']} (usable vintages): median "
        f"{_f(ea['cumulative_events'].get('median'))}; relative binomial SE median "
        f"{_f(ea['relative_binomial_se_median'])}",
        f"Class-average 12-month error / binomial noise: {_f(nf.get('ratio'))}",
        f"Recession {r['window'][0]} to {r['window'][1]}: {r['vintages_observed_in_window']} "
        f"vintages observed, {r['events_in_window']} events",
        f"COVID window: {c['events_in_window']} events, share flagged forbearance "
        f"{_f(c['share_flagged_forbearance'])}",
        f"Power: {p['origins_with_targets']} test origins with targets, "
        f"{p['independent_windows']} independent windows, MDE {_f(p['mde_sd_units_all_origins'])}"
        f" to {_f(p['mde_sd_units_independent_windows'])} standard deviations",
        f"Gate: {g['usable_vintages']} usable vintages (need {g['usable_vintages_needed']}), "
        f"median events {_f(g['median_events_by_age'])} (need {g['events_needed']}): "
        f"{'PASS' if g['passes'] else 'FAIL'}",
    ]
    return "\n".join(lines)
