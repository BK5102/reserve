"""Chronos-Bolt-tiny: the quantile mapping everywhere, the real model only with the extra.

The skip check uses `importlib.util.find_spec`, which looks for the packages without
importing them, so PyTorch is never imported when the `tsfm` extra is missing (CI).
"""

from importlib.util import find_spec

import numpy as np
import pytest

from reserve.models.base import QUANTILES
from reserve.models.tsfm import chronos_bolt
from reserve.models.tsfm.base import describe_mapping, map_quantiles

HAS_EXTRA = find_spec("chronos") is not None and find_spec("torch") is not None
needs_extra = pytest.mark.skipif(not HAS_EXTRA, reason="needs the tsfm extra: uv sync --extra tsfm")

# ----- no model needed ------------------------------------------------------------------


def test_map_quantiles_by_hand():
    native = chronos_bolt.NATIVE_LEVELS
    values = np.array([10.0 * q for q in native])  # value = 10 x level, so linear is exact
    out = map_quantiles(native, values, QUANTILES)
    # 0.05 and 0.95 are held at the 0.1 and 0.9 values; 0.25 and 0.75 are interpolated.
    assert out == pytest.approx([1.0, 1.0, 2.5, 5.0, 7.5, 9.0, 9.0])


def test_map_quantiles_keeps_leading_axes_and_checks_inputs():
    values = np.random.default_rng(0).normal(size=(3, 12, 9)).cumsum(axis=-1)
    out = map_quantiles(chronos_bolt.NATIVE_LEVELS, values, QUANTILES)
    assert out.shape == (3, 12, 7)
    assert np.allclose(out[..., 3], values[..., 4])  # 0.5 is native
    with pytest.raises(ValueError):
        map_quantiles((0.1, 0.5), values, QUANTILES)
    with pytest.raises(ValueError):
        map_quantiles((0.5, 0.1, 0.9), values[..., :3], QUANTILES)


def test_describe_mapping_names_the_flat_tails():
    m = describe_mapping(chronos_bolt.NATIVE_LEVELS, QUANTILES)
    assert m["0.5"] == "native" and m["0.1"] == "native"
    assert "held flat" in m["0.05"] and "held flat" in m["0.95"]
    assert m["0.25"] == "linear between native 0.2 and 0.3"


def test_adapter_metadata_is_pinned():
    assert chronos_bolt.MODEL_ID == "amazon/chronos-bolt-tiny"
    assert chronos_bolt.REVISION == "a0e552de83495b5c28c14c71c374f3e33280b340"
    assert chronos_bolt.PRETRAINING_CUTOFF == "undisclosed"


# ----- real model (tsfm extra only) -----------------------------------------------------


@pytest.fixture(scope="module")
def model():
    if not HAS_EXTRA:
        pytest.skip("needs the tsfm extra: uv sync --extra tsfm")
    return chronos_bolt.ChronosBoltTiny(batch_size=2)


def _contexts():
    rng = np.random.default_rng(1)
    # Cumulative curves of different lengths, like vintages of different ages.
    return [np.cumsum(rng.uniform(0, 0.002, size=n)) for n in (6, 9, 30, 120, 7)]


@needs_extra
def test_real_model_levels_shape_and_order(model):
    assert tuple(round(float(q), 6) for q in model._pipe.quantiles) == chronos_bolt.NATIVE_LEVELS
    out = model.forecast(_contexts(), 12, QUANTILES)
    assert out.shape == (5, 12, len(QUANTILES))
    assert np.isfinite(out).all()
    assert (np.diff(out, axis=-1) >= -1e-7).all()  # quantiles do not cross
    assert "chronos-forecasting-" in model.version


@needs_extra
def test_real_model_mapping_matches_the_library(model):
    """Our mapping equals the library's own handling of levels it was not trained on."""
    import torch

    ctx = _contexts()
    ours = model.forecast(ctx, 12, QUANTILES)
    order = np.argsort([len(c) for c in ctx], kind="stable")
    lib = np.empty_like(ours)
    for i in order:  # one series per call, so no padding differences
        q, _ = model._pipe.predict_quantiles(
            [torch.tensor(ctx[i], dtype=torch.float32)],
            prediction_length=12,
            quantile_levels=list(QUANTILES),
        )
        lib[i] = q[0].numpy()
    assert np.allclose(ours, lib, atol=1e-6)


@needs_extra
def test_real_model_batching_does_not_change_forecasts(model):
    ctx = _contexts()
    a = model.forecast(ctx, 12, QUANTILES)
    big = chronos_bolt.ChronosBoltTiny(batch_size=64)
    b = big.forecast(ctx, 12, QUANTILES)
    assert np.allclose(a, b, atol=1e-6)
    with pytest.raises(ValueError, match="exceeds"):
        model.forecast(ctx, 65, QUANTILES)
