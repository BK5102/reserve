"""The package and its shared Cleanroom dependency import."""

import cleanroom

import reserve


def test_reserve_imports():
    assert reserve.__version__


def test_cleanroom_is_the_tagged_release():
    # Pinned to the cleanroom-v0.2 tag in pyproject.toml.
    assert cleanroom.__version__ == "0.2.0"


def test_cleanroom_v02_modules_are_available():
    from cleanroom import asof, forecast, guard, split, stats

    assert callable(split.rolling_origins) and callable(asof.join_as_known)
    assert callable(forecast.weighted_quantile_loss) and callable(stats.diebold_mariano)
    assert callable(guard.sealed_run)
