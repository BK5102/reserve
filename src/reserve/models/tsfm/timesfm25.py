"""TimesFM 2.5 adapter (google/timesfm-2.5-200m-pytorch, Apache-2.0). STUB (methodology M33).

Constructing it raises `NotImplementedError`. Nothing is installed for it: the package is
not in any Reserve extra yet.

Verified, 2026-11-18:
- Model id `google/timesfm-2.5-200m-pytorch`, license apache-2.0, 231,289,280 parameters
  (Safetensors total; the card says 0.2B), current commit 1d952420fba87f3c6dee4f240de0f1a0fbc790e3
  (last modified 2025-10-02), all from the Hugging Face model API and model card.
- Model card code: `timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)`, then
  `model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256,
  normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True,
  infer_is_positive=True, fix_quantile_crossing=True))`, then
  `point, quantiles = model.forecast(horizon=12, inputs=[...])` with `quantiles` shaped
  (series, horizon, 10): "mean, then 10th to 90th quantiles". So the native levels are
  0.1 to 0.9 at positions 1 to 9 (`native_from_output`), and Reserve maps them exactly as
  for Chronos-Bolt (M32): its "90%" interval would be its own 80% interval.
- Pretraining data on the card: GiftEvalPretrain, Wikimedia pageviews (cutoff November 2023),
  Google Trends top queries (cutoff end of 2022), and synthetic and augmented data.
- The GitHub README (google-research/timesfm) says weights up to 2.5 are Apache-2.0, 2.5
  supports up to 16k context, and covariates through XReg were added back on 2025-10-29.

Unverified:
- Which install gives the 2.5 class: the card says PyPI is "coming soon" and to install
  from source; the README says `pip install timesfm[torch]`; PyPI's latest `timesfm` is
  3.0.2 (requires Python 3.10 or newer). Whether 3.0.2 still ships `TimesFM_2p5_200M_torch`
  was not checked.
- Which revision to pin: the commit above is today's head, not yet tested.
- Overall pretraining cutoff: only two component cutoffs are stated; GiftEvalPretrain's
  is not. Treat as undisclosed.
- Context: the card's example uses 1024 and the README says up to 16k; which limit applies
  with a given `ForecastConfig` was not checked.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
REVISION_CANDIDATE = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"  # head on 2026-11-18, unpinned
PACKAGE = "timesfm"
NATIVE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PRETRAINING_CUTOFF = "undisclosed (components: Wikimedia to 2023-11, Google Trends to 2022-12)"
UNVERIFIED = (
    "which timesfm release on PyPI ships TimesFM_2p5_200M_torch",
    "revision to pin",
    "overall pretraining cutoff",
    "context limit under the chosen ForecastConfig",
)


def native_from_output(quantile_forecast: np.ndarray) -> np.ndarray:
    """(series, horizon, 10) model output -> (series, horizon, 9) at NATIVE_LEVELS.

    Position 0 is the mean, per the model card, and is dropped.
    """
    q = np.asarray(quantile_forecast, dtype=float)
    if q.shape[-1] != 1 + len(NATIVE_LEVELS):
        raise ValueError(f"expected {1 + len(NATIVE_LEVELS)} outputs per step, got {q.shape[-1]}")
    return q[..., 1:]


class TimesFM25:
    name = "timesfm_2p5"
    version = f"{MODEL_ID} (stub, unpinned)"

    def __init__(self, **_kwargs):
        raise NotImplementedError(
            "TimesFM 2.5 is a stub in Reserve: the package is not installed and the install "
            "route and revision are unverified. See src/reserve/models/tsfm/timesfm25.py."
        )

    def forecast(
        self, contexts: list[np.ndarray], horizon: int, quantiles: Sequence[float]
    ) -> np.ndarray:  # pragma: no cover - unreachable while the constructor refuses
        raise NotImplementedError
