"""Chronos-Bolt-tiny adapter (amazon/chronos-bolt-tiny, Apache-2.0). STUB: not yet run.

Needs the optional `tsfm` extra (`uv sync --extra tsfm`), which installs CPU-only PyTorch
and `chronos-forecasting`. Weights are pinned to the Hugging Face commit Tells checked on
2026-11-17. Nothing here is imported, and no weights are downloaded, unless the adapter is
constructed; tests only check that construction fails clearly without the extra.

The call follows Tells' `src/tells/models/tsfm/chronos_bolt.py`, which ran with
chronos-forecasting 2.3.2: `predict_quantiles(series, prediction_length=..., quantile_levels=...)`
returning (quantiles, mean), with quantiles shaped (series, prediction_length, levels).
Unverified for Reserve (docs/tsfm-notes.md): which quantile levels the model was trained
on, and how it treats levels outside them (0.05 and 0.95 here). Check both against the
installed library before the first real run.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MODEL_ID = "amazon/chronos-bolt-tiny"
REVISION = "a0e552de83495b5c28c14c71c374f3e33280b340"


class ChronosBoltTiny:
    name = "chronos_bolt_tiny"

    def __init__(self, threads: int | None = None):
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
        self._pipe = BaseChronosPipeline.from_pretrained(
            MODEL_ID, revision=REVISION, device_map="cpu"
        )

    @property
    def version(self) -> str:
        from importlib import metadata

        lib = metadata.version("chronos-forecasting")
        return f"{MODEL_ID}@{REVISION[:8]}+chronos-forecasting-{lib}"

    def forecast(
        self, contexts: list[np.ndarray], horizon: int, quantiles: Sequence[float]
    ) -> np.ndarray:
        series = [self._torch.tensor(np.asarray(c, dtype=np.float32)) for c in contexts]
        q, _mean = self._pipe.predict_quantiles(
            series, prediction_length=horizon, quantile_levels=list(quantiles)
        )
        return q.float().cpu().numpy()
