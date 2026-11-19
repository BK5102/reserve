"""TSFM adapters. Real models need the optional `tsfm` extra; `fake` never does.

`MODEL_INFO` holds each model's id, pinned revision, status, documented pretraining
cutoff (CLAUDE.md rule 9), and the importable packages it needs (checked before sealing).
The agent's Critic checks that every TSFM in a run has an entry.
Importing this module never imports PyTorch.
"""

from __future__ import annotations

from reserve.models.tsfm import chronos2, chronos_bolt, timesfm25
from reserve.models.tsfm.base import QuantileForecaster, TSFMContestant

KNOWN_MODELS = ("fake", "chronos_bolt_tiny", "chronos2", "timesfm_2p5")

MODEL_INFO: dict[str, dict] = {
    "fake": {
        "model_id": None,
        "revision": None,
        "status": "test stand-in, no pretraining",
        "pretraining_cutoff": "not applicable",
        "requires": (),
    },
    "chronos_bolt_tiny": {
        "model_id": chronos_bolt.MODEL_ID,
        "revision": chronos_bolt.REVISION,
        "status": "real",
        "pretraining_cutoff": chronos_bolt.PRETRAINING_CUTOFF,
        "requires": ("torch", "chronos"),  # the `tsfm` extra
    },
    "chronos2": {
        "model_id": chronos2.MODEL_ID,
        "revision": chronos2.REVISION,
        "status": "stub",
        "pretraining_cutoff": chronos2.PRETRAINING_CUTOFF,
        "requires": ("torch", "chronos"),  # the `tsfm` extra
    },
    "timesfm_2p5": {
        "model_id": timesfm25.MODEL_ID,
        "revision": None,
        "status": "stub",
        "pretraining_cutoff": timesfm25.PRETRAINING_CUTOFF,
        "requires": ("torch", "timesfm"),
    },
}


def missing_packages(name: str) -> list[str]:
    """Packages a TSFM needs that are not installed, found without importing them."""
    from importlib.util import find_spec

    return [m for m in MODEL_INFO[name]["requires"] if find_spec(m) is None]


def make_forecaster(name: str, **kwargs) -> QuantileForecaster:
    if name == "fake":
        from reserve.models.tsfm.fake import FakeTSFM

        return FakeTSFM(**kwargs)
    if name == "chronos_bolt_tiny":
        return chronos_bolt.ChronosBoltTiny(**kwargs)
    if name == "chronos2":
        return chronos2.Chronos2(**kwargs)
    if name == "timesfm_2p5":
        return timesfm25.TimesFM25(**kwargs)
    raise ValueError(f"unknown TSFM {name!r}; known: {KNOWN_MODELS}")


__all__ = [
    "KNOWN_MODELS",
    "MODEL_INFO",
    "QuantileForecaster",
    "TSFMContestant",
    "make_forecaster",
    "missing_packages",
]
