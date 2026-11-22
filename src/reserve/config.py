"""Experiment configs: YAML files under configs/, validated before anything runs (methodology M30).

The whole parsed YAML is what gets hashed and sealed, so any change to the plan (origins,
horizon, contestants, exclusions) produces a new config hash and needs a new lock.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

from reserve import months

PERIODS = ("dev", "test", "recession")
SEALED_PERIODS = ("test",)  # scoring these needs a committed lock (CLAUDE.md rule 3)


class ConfigError(ValueError):
    """The config is invalid."""


def load(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate(cfg)
    return cfg


def origin_window(cfg: dict[str, Any], period: str) -> tuple[date, date]:
    o = cfg["evaluation"]["origins"][period]
    return months.parse(o["first"]), months.parse(o["last"])


def exclusions(cfg: dict[str, Any]) -> list[tuple[date, date]]:
    return [
        (months.parse(lo), months.parse(hi))
        for lo, hi in cfg["evaluation"].get("exclude_target_windows", [])
    ]


def validate(cfg: dict[str, Any]) -> None:
    for key in ("name", "data", "evaluation", "contestants"):
        if key not in cfg:
            raise ConfigError(f"config is missing {key!r}")
    data = cfg["data"]
    if data.get("source") not in ("synthetic", "freddie"):
        raise ConfigError("data.source must be 'synthetic' or 'freddie'")
    if data["source"] == "synthetic" and "synthetic" not in data:
        raise ConfigError("a synthetic config needs data.synthetic settings")
    ev = cfg["evaluation"]
    if ev.get("period") not in PERIODS:
        raise ConfigError(f"evaluation.period must be one of {PERIODS}")
    if ev["period"] not in ev.get("origins", {}):
        raise ConfigError(f"evaluation.origins has no {ev['period']!r} window")
    horizon = int(ev["horizon"])
    if horizon < 1 or int(ev["step_months"]) < 1:
        raise ConfigError("horizon and step_months must be positive")
    quantiles = ev["quantiles"]
    if sorted(quantiles) != list(quantiles) or not all(0 < q < 1 for q in quantiles):
        raise ConfigError("quantiles must be increasing and inside (0, 1)")
    if cfg.get("macro", {}).get("mode", "asof") not in ("asof", "asof_fallback", "latest", "none"):
        raise ConfigError("macro.mode must be 'asof', 'asof_fallback', 'latest', or 'none'")
    for period in ev["origins"]:
        lo, hi = origin_window(cfg, period)
        if lo > hi:
            raise ConfigError(f"{period} origins: first is after last")
    # Dev outcomes must not reach into the test targets: the last dev forecast must end by
    # the first test origin, so tuning on dev never shows a test-period outcome (M31).
    if "dev" in ev["origins"] and "test" in ev["origins"]:
        dev_last = origin_window(cfg, "dev")[1]
        test_first = origin_window(cfg, "test")[0]
        if months.add(dev_last, horizon) > test_first:
            raise ConfigError(
                f"dev origins end {dev_last}; their {horizon}-month outcomes reach past the "
                f"first test origin {test_first}"
            )
    _validate_experiments(cfg)
    if data["source"] == "synthetic" and ev["period"] in SEALED_PERIODS:
        raise ConfigError("synthetic data is never scored as a sealed test (M17)")


def _validate_experiments(cfg: dict[str, Any]) -> None:
    """Optional `experiments:` block: thin-history levels and HYPOTHETICAL scenario names."""
    from reserve.stress.scenarios import SCENARIOS

    exp = cfg.get("experiments") or {}
    unknown = set(exp) - {"thin_history", "scenarios"}
    if unknown:
        raise ConfigError(f"unknown experiments {sorted(unknown)}")
    months = (exp.get("thin_history") or {}).get("months")
    if months is not None:
        if not months or any(int(m) < 1 for m in months) or sorted(months) != list(months):
            raise ConfigError("experiments.thin_history.months must be increasing and positive")
        if int(months[0]) < int(cfg["evaluation"].get("min_history", 6)):
            raise ConfigError("thin-history levels must be at least evaluation.min_history")
    names = (exp.get("scenarios") or {}).get("names")
    if names is not None:
        bad = set(names) - set(SCENARIOS)
        if bad:
            raise ConfigError(f"unknown scenarios {sorted(bad)}; known: {sorted(SCENARIOS)}")
