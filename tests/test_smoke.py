"""The package and its shared Cleanroom dependency import."""

import cleanroom

import reserve


def test_reserve_imports():
    assert reserve.__version__


def test_cleanroom_is_the_tagged_release():
    # Pinned to cleanroom-v0.1 in pyproject.toml. TODO(cleanroom-v0.2): update with the bump.
    assert cleanroom.__version__ == "0.1.0"
