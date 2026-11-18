"""Chronos-2 adapter (amazon/chronos-2, Apache-2.0). STUB: never run in Reserve (methodology M33).

Constructing it raises `NotImplementedError`, so it cannot enter a run by accident. The
facts below are what the adapter will rely on; each says where it was checked.

Verified, 2026-11-18:
- Model id `amazon/chronos-2`; Hugging Face commit 29ec3766d36d6f73f0696f85560a422f50e8498c is
  the current `sha` from the Hugging Face model API (last modified 2026-06-05), and is the
  revision Tells pinned on 2026-11-14. License apache-2.0 and 119,477,664 parameters from the
  same API record.
- Package: `chronos-forecasting` (the one Reserve's `tsfm` extra already installs). Version
  2.3.2 contains `chronos.Chronos2Pipeline`, whose `predict_quantiles(inputs,
  prediction_length, quantile_levels, **predict_kwargs)` returns a list with one tensor per
  series, shaped (n_variates, horizon, levels); `predict` takes `batch_size` (default 256),
  `context_length`, `cross_learning`, and `limit_prediction_length` (read from
  `chronos/chronos2/pipeline.py`).
- Trained quantile levels (21): 0.01, 0.05, 0.1, ..., 0.95, 0.99; context length 8192; output
  patches of 16 steps, at most 64 of them. Read from `config.json` at the pinned revision in
  the local Hugging Face cache (downloaded earlier by Tells; nothing was downloaded here).
  All seven Reserve levels are native, so no mapping is needed.

Unverified:
- How well it handles very short inputs (6 points) and very small values; never run here.
- Covariate support (past-only and known-future covariates) is from Tells' notes on the
  model card, not rechecked for Reserve, and not used: Reserve's TSFM input is univariate.
- Pretraining cutoff: undisclosed (Tells checked the paper, Ansari et al. 2025).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MODEL_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
PACKAGE = "chronos-forecasting"
NATIVE_LEVELS = (
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
)  # fmt: skip
CONTEXT_LENGTH = 8192
PRETRAINING_CUTOFF = "undisclosed"
UNVERIFIED = (
    "behaviour on 6-point inputs and tiny values (never run)",
    "covariate support (from Tells' notes, not rechecked; unused)",
)


class Chronos2:
    name = "chronos2"
    version = f"{MODEL_ID}@{REVISION[:8]} (stub)"

    def __init__(self, **_kwargs):
        raise NotImplementedError(
            "Chronos-2 is a stub in Reserve: its weights have not been loaded or run here. "
            "See src/reserve/models/tsfm/chronos2.py and docs/tsfm-notes.md."
        )

    def forecast(
        self, contexts: list[np.ndarray], horizon: int, quantiles: Sequence[float]
    ) -> np.ndarray:  # pragma: no cover - unreachable while the constructor refuses
        raise NotImplementedError
