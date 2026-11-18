"""HYPOTHETICAL macro scenarios for conditional forecasts (PLAN.md FR12; methodology M35).

Every object here is labelled hypothetical (CLAUDE.md rule 6). A scenario is a what-if
path for the macro features over the forecast horizon, starting from the values known at
the origin; it is never a forecast of the economy and is never scored against outcomes.

A scenario moves each feature away from its origin value along a straight line that
reaches its full size at month 12 and stays there: unemployment rises by `unemployment_rise`
percentage points (its level and its 12-month change both rise by that amount), and the
12-month log change in house prices falls by `hpi_growth_fall`. The mortgage rate is held at
its origin value in every scenario. The sizes are round numbers chosen for a pipeline check,
not calibrated to any official stress scenario.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

LABEL = "HYPOTHETICAL"
DISCLAIMER = (
    "HYPOTHETICAL scenarios: what-if macro paths for conditional forecasts, not forecasts of "
    "the economy (CLAUDE.md rule 6)."
)
RAMP_MONTHS = 12


@dataclass(frozen=True)
class Scenario:
    name: str
    unemployment_rise: float  # percentage points by month 12
    hpi_growth_fall: float  # fall in the 12-month log change of house prices by month 12
    hypothetical: bool = True

    def deltas(self, features: Sequence[str], horizon: int) -> np.ndarray:
        """(horizon, features) amounts added to the origin values, month 1 to `horizon`."""
        ramp = np.minimum(np.arange(1, horizon + 1) / RAMP_MONTHS, 1.0)
        size = {
            "unemployment": self.unemployment_rise,
            "unemployment_change_12m": self.unemployment_rise,
            "mortgage_rate": 0.0,
            "hpi_growth_12m": -self.hpi_growth_fall,
        }
        unknown = set(features) - set(size)
        if unknown:
            raise ValueError(f"no scenario path for features {sorted(unknown)}")
        return ramp[:, None] * np.array([size[f] for f in features])[None, :]

    def describe(self) -> dict:
        return {
            "name": self.name,
            "label": LABEL,
            "hypothetical": True,
            "unemployment_rise_points": self.unemployment_rise,
            "hpi_growth_12m_fall": self.hpi_growth_fall,
            "mortgage_rate": "held at the origin value",
            "ramp_months": RAMP_MONTHS,
        }


SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario("baseline", 0.0, 0.0),
    "adverse": Scenario("adverse", 2.0, 0.05),
    "severe": Scenario("severe", 5.0, 0.15),
}


def get(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
    return SCENARIOS[name]
