"""TimesFM 2.5: metadata and output layout everywhere; the real model only with the
`timesfm` extra (methodology M44).

The skip check uses `importlib.util.find_spec`, which looks for the packages without
importing them, so neither PyTorch nor `timesfm` is imported when the extra is missing.
"""

from importlib.util import find_spec

import numpy as np
import pytest

from reserve.models.base import QUANTILES
from reserve.models.tsfm import MODEL_INFO, timesfm25

HAS_EXTRA = find_spec("timesfm") is not None and find_spec("torch") is not None
needs_extra = pytest.mark.skipif(
    not HAS_EXTRA, reason="needs the timesfm extra: uv sync --extra timesfm"
)

# ----- no model needed ------------------------------------------------------------------


def test_output_drops_the_mean_column():
    out = np.arange(2 * 12 * 10, dtype=float).reshape(2, 12, 10)
    native = timesfm25.native_from_output(out)
    assert native.shape == (2, 12, 9) and np.array_equal(native, out[..., 1:])
    with pytest.raises(ValueError):
        timesfm25.native_from_output(out[..., :9])


def test_adapter_metadata_is_pinned():
    assert timesfm25.MODEL_ID == "google/timesfm-2.5-200m-pytorch"
    assert timesfm25.REVISION == "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
    assert timesfm25.SOURCE_COMMIT == "8cb0628371af142e16b8c232cc9fbf667ffb12f9"
    info = MODEL_INFO["timesfm_2p5"]
    assert info["status"] == "real" and info["revision"] == timesfm25.REVISION
    assert info["requires"] == ("torch", "timesfm")
    assert "undisclosed" in info["pretraining_cutoff"]


def test_pyproject_pins_the_source_commit_outside_the_default_install():
    import tomllib
    from pathlib import Path

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    assert not any("timesfm" in d for d in project["project"]["dependencies"])
    assert not any("timesfm" in d for d in extras["tsfm"])
    assert any(d.startswith("timesfm") for d in extras["timesfm"])
    source = project["tool"]["uv"]["sources"]["timesfm"]
    assert source["git"] == timesfm25.SOURCE_REPO and source["rev"] == timesfm25.SOURCE_COMMIT


# ----- real model (timesfm extra only) --------------------------------------------------


@pytest.fixture(scope="module")
def model():
    if not HAS_EXTRA:
        pytest.skip("needs the timesfm extra: uv sync --extra timesfm")
    return timesfm25.TimesFM25(batch_size=4)


def _contexts():
    rng = np.random.default_rng(1)
    return [np.cumsum(rng.uniform(0, 0.002, size=n)) for n in (6, 9, 30, 120, 7)]


@needs_extra
def test_real_model_levels_shape_and_order(model):
    out = model.forecast(_contexts(), 12, QUANTILES)
    assert out.shape == (5, 12, len(QUANTILES))
    assert np.isfinite(out).all()
    assert (np.diff(out, axis=-1) >= -1e-7).all()  # quantiles do not cross
    assert "timesfm-" in model.version and "8cb06283" in model.version


@needs_extra
def test_real_model_median_is_the_point_forecast(model):
    ctx = _contexts()
    point, q = model._model.forecast(horizon=12, inputs=list(ctx))
    assert np.allclose(point, q[..., 5])  # column 5 is the 0.5 level, column 0 the mean
    assert np.allclose(model.native_forecast(ctx, 12)[..., 4], point)


@needs_extra
def test_real_model_does_not_change_its_input_and_refuses_long_horizons(model):
    ctx = _contexts()
    model.forecast(ctx, 12, QUANTILES)
    assert len(ctx) == 5  # the library pads its list in place; the adapter passes a copy
    with pytest.raises(ValueError, match="exceeds"):
        model.forecast(ctx, timesfm25.MAX_HORIZON + 1, QUANTILES)
