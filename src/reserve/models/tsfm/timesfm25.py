"""TimesFM 2.5 adapter (google/timesfm-2.5-200m-pytorch, Apache-2.0), methodology M33 and M44.

Needs the optional `timesfm` extra (`uv sync --extra timesfm`), which installs the
`timesfm` package from the official google-research/timesfm GitHub repo at a pinned commit
(SOURCE_COMMIT, the commit tagged v3.0.2) plus CPU-only PyTorch. It is not in the default
install or the `tsfm` extra. Nothing here imports PyTorch or `timesfm`, and no weights are
downloaded, unless the adapter is constructed.

Verified on 2026-11-19 by installing that commit into a separate scratch environment
(Python 3.12, torch 2.15.0+cpu) and running the model on CPU, then reading the installed
source (`timesfm/__init__.py`, `timesfm/configs.py`, `timesfm/timesfm_2p5/*.py`):
- The package exports `TimesFM_2p5_200M_torch` (when PyTorch is importable) and
  `ForecastConfig`. `from_pretrained(model_id, revision=..., torch_compile=...)` downloads
  `model.safetensors` from the Hugging Face repo; `torch_compile` defaults to True, which
  runs `torch.compile`; the adapter passes False (no compiler toolchain is assumed on a
  laptop, and the probe ran without it).
- `compile(ForecastConfig(...))` must be called before `forecast(horizon, inputs)`, which
  returns (point, quantiles) with quantiles shaped (series, horizon, 10). The quantile head
  is the mean followed by 0.1, 0.2, ..., 0.9 (`TimesFM_2p5_200M_Definition.quantiles`, and
  `q = len(quantiles) + 1` in the module); the point forecast is column 5, the median
  (`decode_index = 5`), which the probe confirmed equals quantile column 5.
- Context limit 16384 (`context_limit`); `max_context` is rounded up to a multiple of the
  32-step input patch and `max_horizon` to the 128-step output patch. Inputs longer than
  `max_context` are cut to their last `max_context` points; shorter ones are left-padded
  and masked. Reserve uses `max_context` 512 (curves are at most about 330 months) and
  `max_horizon` 128 (one output patch covers the 12-month horizon).
- `forecast` pads its input list in place to a multiple of the batch size, so the adapter
  passes a copy.
- The ForecastConfig flags are the model card's example: normalize inputs, continuous
  quantile head, flip invariance, infer non-negative series, fix quantile crossing.
- Weights pinned to Hugging Face commit 1d952420fba87f3c6dee4f240de0f1a0fbc790e3, the repo's
  current `sha` (last modified 2025-10-02) on 2026-11-18 and 2026-11-19.

Unverified:
- Overall pretraining cutoff: the card lists GiftEvalPretrain (no date), Wikimedia pageviews
  to 2023-11, Google Trends to 2022-12, and synthetic data. Treat as undisclosed; the
  stated components postdate Reserve's 2013 to 2018 test period (L25).
- Covariates (XReg) need the separate `xreg` extra (JAX) and are not used.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from reserve.models.tsfm.base import describe_mapping, map_quantiles

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"  # Hugging Face weights
SOURCE_REPO = "https://github.com/google-research/timesfm"
SOURCE_COMMIT = "8cb0628371af142e16b8c232cc9fbf667ffb12f9"  # tag v3.0.2
PACKAGE = "timesfm"
NATIVE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MAX_CONTEXT = 512
MAX_HORIZON = 128
PRETRAINING_CUTOFF = "undisclosed (components: Wikimedia to 2023-11, Google Trends to 2022-12)"
UNVERIFIED = ("overall pretraining cutoff", "covariates (XReg) unused")


def native_from_output(quantile_forecast: np.ndarray) -> np.ndarray:
    """(series, horizon, 10) model output -> (series, horizon, 9) at NATIVE_LEVELS.

    Position 0 is the mean (model card, and the module's quantile head) and is dropped.
    """
    q = np.asarray(quantile_forecast, dtype=float)
    if q.shape[-1] != 1 + len(NATIVE_LEVELS):
        raise ValueError(f"expected {1 + len(NATIVE_LEVELS)} outputs per step, got {q.shape[-1]}")
    return q[..., 1:]


class TimesFM25:
    name = "timesfm_2p5"

    # The library pads every call to a multiple of the batch size; Reserve sends one call per
    # origin of a few dozen vintages, so a small batch wastes less (measured, tsfm-notes).
    def __init__(self, batch_size: int = 16, threads: int | None = None):
        try:
            import timesfm
            import torch
        except ImportError as err:  # the optional extra is not installed
            raise ImportError(
                "TimesFM 2.5 needs the optional extra: uv sync --extra timesfm"
            ) from err
        if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            raise ImportError("this timesfm install has no TimesFM_2p5_200M_torch class")
        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.batch_size = batch_size
        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            MODEL_ID, revision=REVISION, torch_compile=False
        )
        levels = tuple(round(float(q), 6) for q in self._model.model.config.quantiles)
        if levels != NATIVE_LEVELS:
            raise RuntimeError(f"{MODEL_ID} reports quantile levels {levels}, not {NATIVE_LEVELS}")
        self._model.compile(
            timesfm.ForecastConfig(
                max_context=MAX_CONTEXT,
                max_horizon=MAX_HORIZON,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
                per_core_batch_size=batch_size,
            )
        )

    @property
    def version(self) -> str:
        from importlib import metadata

        return (
            f"{MODEL_ID}@{REVISION[:8]}+{PACKAGE}-{metadata.version(PACKAGE)}"
            f"@git-{SOURCE_COMMIT[:8]}"
        )

    def describe(self) -> dict:
        return {
            "native_quantile_levels": list(NATIVE_LEVELS),
            "batch_size": self.batch_size,
            "max_context": MAX_CONTEXT,
            "torch_threads": self._torch.get_num_threads(),
            "source_commit": SOURCE_COMMIT,
            "pretraining_cutoff": PRETRAINING_CUTOFF,
        }

    def native_forecast(self, contexts: list[np.ndarray], horizon: int) -> np.ndarray:
        """(series, horizon, 9) forecasts at the model's own levels."""
        if horizon > MAX_HORIZON:
            raise ValueError(f"horizon {horizon} exceeds the compiled maximum {MAX_HORIZON}")
        inputs = [np.asarray(c, dtype=np.float64) for c in contexts]  # forecast pads in place
        _point, quantiles = self._model.forecast(horizon=horizon, inputs=inputs)
        return native_from_output(quantiles)

    def forecast(
        self, contexts: list[np.ndarray], horizon: int, quantiles: Sequence[float]
    ) -> np.ndarray:
        return map_quantiles(NATIVE_LEVELS, self.native_forecast(contexts, horizon), quantiles)

    @staticmethod
    def mapping(quantiles: Sequence[float]) -> dict[str, str]:
        return describe_mapping(NATIVE_LEVELS, quantiles)
