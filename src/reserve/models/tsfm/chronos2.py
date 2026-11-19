"""Chronos-2 adapter (amazon/chronos-2, Apache-2.0), methodology M33 and M43.

Needs the optional `tsfm` extra (`uv sync --extra tsfm`), the same one Chronos-Bolt uses.
Nothing here imports PyTorch, and no weights are loaded, unless the adapter is constructed.
Weights are pinned to the Hugging Face commit Tells pinned on 2026-11-14.

Verified, 2026-11-18 and 2026-11-19:
- Model id `amazon/chronos-2`; commit 29ec3766d36d6f73f0696f85560a422f50e8498c is the current
  `sha` from the Hugging Face model API (last modified 2026-06-05). License apache-2.0 and
  119,477,664 parameters from the same API record.
- Package `chronos-forecasting` 2.3.2 (`uv.lock`): `chronos.Chronos2Pipeline.predict_quantiles(
  inputs, prediction_length, quantile_levels, **predict_kwargs)` returns (quantiles, mean),
  each a list with one tensor per series shaped (n_variates, horizon, levels); `predict`
  takes `batch_size` (default 256), `context_length`, `cross_learning` (default False), and
  `limit_prediction_length` (`chronos/chronos2/pipeline.py`).
- Trained quantile levels (21): 0.01, 0.05, 0.1, ..., 0.95, 0.99, from `config.json` at the
  pinned revision, and confirmed from the loaded pipeline's `quantiles` attribute, which the
  adapter checks at load time. All seven Reserve levels are native, so `map_quantiles` only
  selects them (no interpolation, no flat tails); a test shows the result equals asking the
  library for Reserve's levels directly.
- Context 8192 steps; output patches of 16 steps, at most 64 of them (`config.json`).
  Reserve's curves are at most about 330 months and its horizon is 12, so nothing is cut
  and one pass covers the horizon.
- Short inputs: 6-point curves and flat all-zero curves return finite forecasts (checked by
  a test with the real model). With 6 points the spread is wide and the lower quantiles go
  below the last known value, which Reserve's clean-up clips (M23).

Unverified:
- Covariate support (past-only and known-future covariates) is from Tells' notes on the
  model card and is not used: Reserve's TSFM input is univariate.
- Pretraining cutoff: undisclosed (Tells checked the paper, Ansari et al. 2025).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from reserve.models.tsfm.base import describe_mapping, map_quantiles

MODEL_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
PACKAGE = "chronos-forecasting"
NATIVE_LEVELS = (
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
)  # fmt: skip
CONTEXT_LENGTH = 8192
PRETRAINING_CUTOFF = "undisclosed"
UNVERIFIED = ("covariate support (from Tells' notes, not rechecked; unused)",)


class Chronos2:
    name = "chronos2"

    def __init__(self, batch_size: int = 256, threads: int | None = None):
        try:
            import torch
            from chronos import Chronos2Pipeline
        except ImportError as err:  # the optional extra is not installed
            raise ImportError("Chronos-2 needs the optional extra: uv sync --extra tsfm") from err
        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.batch_size = batch_size
        self._pipe = Chronos2Pipeline.from_pretrained(MODEL_ID, revision=REVISION, device_map="cpu")
        levels = tuple(round(float(q), 6) for q in self._pipe.quantiles)
        if levels != NATIVE_LEVELS:
            raise RuntimeError(f"{MODEL_ID} reports quantile levels {levels}, not {NATIVE_LEVELS}")

    @property
    def version(self) -> str:
        from importlib import metadata

        return f"{MODEL_ID}@{REVISION[:8]}+{PACKAGE}-{metadata.version(PACKAGE)}"

    def describe(self) -> dict:
        return {
            "native_quantile_levels": list(NATIVE_LEVELS),
            "batch_size": self.batch_size,
            "torch_threads": self._torch.get_num_threads(),
            "pretraining_cutoff": PRETRAINING_CUTOFF,
        }

    def native_forecast(self, contexts: list[np.ndarray], horizon: int) -> np.ndarray:
        """(series, horizon, 21) forecasts at the model's own levels, sorted by length."""
        torch = self._torch
        order = np.argsort([len(c) for c in contexts], kind="stable")
        series = [torch.tensor(np.asarray(contexts[i], dtype=np.float32)) for i in order]
        q, _mean = self._pipe.predict_quantiles(
            series,
            prediction_length=horizon,
            quantile_levels=list(NATIVE_LEVELS),
            batch_size=self.batch_size,
        )
        out = np.empty((len(contexts), horizon, len(NATIVE_LEVELS)))
        for pos, i in enumerate(order):
            out[i] = q[pos][0].float().cpu().numpy()  # one variate per series
        return out

    def forecast(
        self, contexts: list[np.ndarray], horizon: int, quantiles: Sequence[float]
    ) -> np.ndarray:
        return map_quantiles(NATIVE_LEVELS, self.native_forecast(contexts, horizon), quantiles)

    @staticmethod
    def mapping(quantiles: Sequence[float]) -> dict[str, str]:
        return describe_mapping(NATIVE_LEVELS, quantiles)
