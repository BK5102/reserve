"""Chronos-2: metadata everywhere, the real model only with the `tsfm` extra (methodology M43).

The skip check uses `importlib.util.find_spec`, which looks for the packages without
importing them, so PyTorch is never imported when the extra is missing (CI).
"""

from importlib.util import find_spec

import numpy as np
import pytest

from reserve.models.base import QUANTILES
from reserve.models.tsfm import MODEL_INFO, chronos2
from reserve.models.tsfm.base import describe_mapping

HAS_EXTRA = find_spec("chronos") is not None and find_spec("torch") is not None
needs_extra = pytest.mark.skipif(not HAS_EXTRA, reason="needs the tsfm extra: uv sync --extra tsfm")

# ----- no model needed ------------------------------------------------------------------


def test_every_reserve_level_is_native_so_the_mapping_is_exact():
    assert set(QUANTILES) <= set(chronos2.NATIVE_LEVELS)
    assert set(describe_mapping(chronos2.NATIVE_LEVELS, QUANTILES).values()) == {"native"}


def test_adapter_metadata_is_pinned_and_real():
    assert chronos2.MODEL_ID == "amazon/chronos-2"
    assert chronos2.REVISION == "29ec3766d36d6f73f0696f85560a422f50e8498c"
    assert MODEL_INFO["chronos2"]["status"] == "real"
    assert MODEL_INFO["chronos2"]["revision"] == chronos2.REVISION
    assert MODEL_INFO["chronos2"]["pretraining_cutoff"] == "undisclosed"


# ----- real model (tsfm extra only) -----------------------------------------------------


@pytest.fixture(scope="module")
def model():
    if not HAS_EXTRA:
        pytest.skip("needs the tsfm extra: uv sync --extra tsfm")
    return chronos2.Chronos2(batch_size=2)


def _contexts():
    rng = np.random.default_rng(1)
    # Cumulative curves of different lengths, like vintages of different ages.
    return [np.cumsum(rng.uniform(0, 0.002, size=n)) for n in (6, 9, 30, 120, 7)]


@needs_extra
def test_real_model_levels_shape_and_order(model):
    assert tuple(round(float(q), 6) for q in model._pipe.quantiles) == chronos2.NATIVE_LEVELS
    out = model.forecast(_contexts(), 12, QUANTILES)
    assert out.shape == (5, 12, len(QUANTILES))
    assert np.isfinite(out).all()
    assert (np.diff(out, axis=-1) >= -1e-7).all()  # quantiles do not cross
    assert "chronos-forecasting-" in model.version and "29ec3766" in model.version


@needs_extra
def test_real_model_mapping_equals_asking_the_library_directly(model):
    """Reserve asks for all 21 native levels and keeps its seven; same as asking for seven."""
    import torch

    ctx = _contexts()
    ours = model.forecast(ctx, 12, QUANTILES)
    q, _ = model._pipe.predict_quantiles(
        [torch.tensor(c, dtype=torch.float32) for c in ctx],
        prediction_length=12,
        quantile_levels=list(QUANTILES),
    )
    lib = np.stack([x[0].float().numpy() for x in q])
    assert np.allclose(ours, lib, atol=1e-6)


@needs_extra
def test_real_model_batching_does_not_change_forecasts(model):
    ctx = _contexts()
    a = model.forecast(ctx, 12, QUANTILES)
    b = chronos2.Chronos2(batch_size=256).forecast(ctx, 12, QUANTILES)
    assert np.allclose(a, b, atol=1e-6)


@needs_extra
def test_real_model_handles_six_points_and_flat_zero_curves(model):
    out = model.forecast([np.zeros(6), np.full(6, 1e-4)], 12, QUANTILES)
    assert np.isfinite(out).all() and out.shape == (2, 12, 7)
