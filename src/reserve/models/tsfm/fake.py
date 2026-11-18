"""A tiny deterministic stand-in for a TSFM, for tests and CI (no PyTorch, no weights).

Point forecast: the last value plus k times the mean of the last six steps. Spread: the
standard deviation of those steps times sqrt(k), on normal quantiles. It exercises every
part of the TSFM path (context building, shape checks, clean-up, scoring) for free.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats as sps


class FakeTSFM:
    name = "fake"
    version = "fake-1"

    def __init__(self, window: int = 6):
        self.window = window

    def forecast(
        self, contexts: list[np.ndarray], horizon: int, quantiles: Sequence[float]
    ) -> np.ndarray:
        z = sps.norm.ppf(np.asarray(quantiles))
        k = np.arange(1, horizon + 1)
        out = np.empty((len(contexts), horizon, len(quantiles)))
        for i, c in enumerate(contexts):
            steps = np.diff(c[-(self.window + 1) :]) if len(c) > 1 else np.zeros(1)
            mean = steps.mean()
            sd = steps.std() if len(steps) > 1 else 0.0
            out[i] = c[-1] + (k * mean)[:, None] + (np.sqrt(k) * sd)[:, None] * z
        return out
