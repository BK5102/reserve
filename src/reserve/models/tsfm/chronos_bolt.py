"""Chronos-Bolt-tiny adapter (amazon/chronos-bolt-tiny, Apache-2.0), methodology M27 and M32.

Needs the optional `tsfm` extra (`uv sync --extra tsfm`), which installs CPU-only PyTorch
and `chronos-forecasting`. Weights are pinned to the Hugging Face commit Tells checked on
2026-11-17. Nothing here imports PyTorch, and no weights are downloaded, unless the adapter
is constructed.

Checked against chronos-forecasting 2.3.2 (`chronos/chronos_bolt.py`) and the model's
`config.json` at the pinned revision, on 2026-11-18:

- The model is trained on 9 quantile levels, 0.1, 0.2, ..., 0.9 (`chronos_config.quantiles`).
  The adapter asks for exactly those, so the library does no interpolation, and maps them
  to Reserve's seven levels itself with `map_quantiles` (linear inside 0.1 to 0.9, flat
  outside). That equals what the library would do for levels it was not trained on
  (`predict_quantiles` pads the grid with copies of its end levels and calls
  `torch.quantile`), but it no longer depends on that code path, which the library itself
  marks as a hack that assumes an equidistant grid.
- Context: `context_length` 2048 steps; longer inputs are cut to the last 2048 by the
  library. Vintage curves are at most about 300 months long.
- One forward pass predicts `prediction_length` 64 steps; Reserve's horizon is 12, so
  there is no recursive roll-out. The adapter refuses horizons above 64.
- `predict_quantiles(inputs, prediction_length, quantile_levels, **predict_kwargs)` takes a
  list of 1-D tensors of different lengths (left-padded with NaN by `left_pad_and_stack_1D`)
  and has no batch size, so the adapter batches itself, sorted by length so short series
  are not padded to a long neighbour.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from reserve.models.tsfm.base import describe_mapping, map_quantiles

MODEL_ID = "amazon/chronos-bolt-tiny"
REVISION = "a0e552de83495b5c28c14c71c374f3e33280b340"
PACKAGE = "chronos-forecasting"
# From config.json at REVISION (chronos_config), checked 2026-11-18.
NATIVE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CONTEXT_LENGTH = 2048
PREDICTION_LENGTH = 64
PRETRAINING_CUTOFF = "undisclosed"


class ChronosBoltTiny:
    name = "chronos_bolt_tiny"

    def __init__(self, batch_size: int = 64, threads: int | None = None):
        try:
            import torch
            from chronos import BaseChronosPipeline
        except ImportError as err:  # the optional extra is not installed
            raise ImportError(
                "Chronos-Bolt needs the optional extra: uv sync --extra tsfm"
            ) from err
        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.batch_size = batch_size
        self._pipe = BaseChronosPipeline.from_pretrained(
            MODEL_ID, revision=REVISION, device_map="cpu"
        )
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
        """(series, horizon, 9) forecasts at the model's own levels, batched by length."""
        if horizon > PREDICTION_LENGTH:
            raise ValueError(f"horizon {horizon} exceeds the model's {PREDICTION_LENGTH} steps")
        torch = self._torch
        order = np.argsort([len(c) for c in contexts], kind="stable")
        out = np.empty((len(contexts), horizon, len(NATIVE_LEVELS)))
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            series = [torch.tensor(np.asarray(contexts[i], dtype=np.float32)) for i in idx]
            q, _mean = self._pipe.predict_quantiles(
                series,
                prediction_length=horizon,
                quantile_levels=list(NATIVE_LEVELS),
                limit_prediction_length=True,
            )
            out[idx] = q.float().cpu().numpy()
        return out

    def forecast(
        self, contexts: list[np.ndarray], horizon: int, quantiles: Sequence[float]
    ) -> np.ndarray:
        return map_quantiles(NATIVE_LEVELS, self.native_forecast(contexts, horizon), quantiles)

    @staticmethod
    def mapping(quantiles: Sequence[float]) -> dict[str, str]:
        return describe_mapping(NATIVE_LEVELS, quantiles)
