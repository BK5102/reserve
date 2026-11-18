"""TSFM adapters. Real models need the optional `tsfm` extra; `fake` never does."""

from __future__ import annotations

from reserve.models.tsfm.base import QuantileForecaster, TSFMContestant

KNOWN_MODELS = ("fake", "chronos_bolt_tiny")


def make_forecaster(name: str, **kwargs) -> QuantileForecaster:
    if name == "fake":
        from reserve.models.tsfm.fake import FakeTSFM

        return FakeTSFM(**kwargs)
    if name == "chronos_bolt_tiny":
        from reserve.models.tsfm.chronos_bolt import ChronosBoltTiny

        return ChronosBoltTiny(**kwargs)
    raise ValueError(f"unknown TSFM {name!r}; known: {KNOWN_MODELS}")


__all__ = ["KNOWN_MODELS", "QuantileForecaster", "TSFMContestant", "make_forecaster"]
