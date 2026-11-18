"""Prepare data and run one experiment from a config (methodology M30, M31).

`prepare` turns raw files into Parquet tables under the config's `data.interim_dir`:
`curves.parquet` (the vintage table), `loans.parquet` (one row per loan), and, when macro
data exists, `macro_features.parquet`. For synthetic configs it first writes the fake
portfolio in the Freddie Mac layout and then reads it back with the real loader, so the
synthetic path exercises the same code as real data.

`run` scores the config's evaluation period. The test period is scored only inside a
`cleanroom.guard.sealed_run`, which needs a committed lock for exactly this config and
data checksum and a clean git tree, and records every attempt.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from cleanroom import guard, lock, provenance

from reserve import config as cfgmod
from reserve.data import loans, macro, synthetic, vintages
from reserve.evaluation import harness, origins, scoring
from reserve.models import baselines, hazard
from reserve.models.tsfm import TSFMContestant, make_forecaster

INTERIM_FILES = ("curves.parquet", "loans.parquet")


def _read_freddie(raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    files = loans.find_sample_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"no sample_orig_YYYY.txt / sample_perf_YYYY.txt pairs in {raw_dir}; see README, "
            "'Manual steps'"
        )
    orig = pl.concat(
        [loans.read_origination(o, columns=("first_payment_date",)) for _, o, _ in files]
    )
    perf = pl.concat([loans.read_performance(p) for _, _, p in files], how="diagonal")
    return orig, perf


def prepare(cfg: dict[str, Any]) -> dict[str, str]:
    """Build the interim tables for a config. Returns {file name: sha256}."""
    data = cfg["data"]
    interim = Path(data["interim_dir"])
    interim.mkdir(parents=True, exist_ok=True)
    observations = None
    if data["source"] == "synthetic":
        sim = synthetic.simulate(synthetic.SimConfig(**data["synthetic"]))
        raw = Path(data.get("raw_dir", f"data/synthetic/{cfg['name']}"))
        synthetic.write(sim, raw)
        orig, perf = _read_freddie(raw / "freddie")
        observations = macro.load_observations(raw / "fred")
    else:
        orig, perf = _read_freddie(Path(data["raw_dir"]))
        fred_dir = data.get("fred_dir")
        if fred_dir and all((Path(fred_dir) / f"{s}.parquet").exists() for s in macro.SERIES):
            observations = macro.load_observations(fred_dir)
    built = vintages.build(orig, perf)
    built.curves.write_parquet(interim / "curves.parquet")
    built.loans.write_parquet(interim / "loans.parquet")
    mode = cfg.get("macro", {}).get("mode", "asof")
    if observations is not None and mode != "none":
        periods = sorted(built.curves["period"].unique().to_list())
        feats = macro.macro_features(observations, periods, mode=mode)
        feats.write_parquet(interim / "macro_features.parquet")
    written = sorted(p.name for p in interim.glob("*.parquet"))
    return {name: provenance.file_sha256(interim / name) for name in written}


def load_prepared(cfg: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame | None, dict[str, Path]]:
    interim = Path(cfg["data"]["interim_dir"])
    missing = [f for f in INTERIM_FILES if not (interim / f).exists()]
    if missing:
        raise FileNotFoundError(f"{missing} not in {interim}; run `reserve prepare` first")
    files = {"curves": interim / "curves.parquet"}
    feats = None
    if cfg.get("macro", {}).get("mode", "asof") != "none":
        path = interim / "macro_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; prepare macro data or set macro.mode: none")
        feats = pl.read_parquet(path)
        files["macro_features"] = path
    return pl.read_parquet(files["curves"]), feats, files


def build_contestants(cfg: dict[str, Any], feats: pl.DataFrame | None) -> list:
    spec = cfg["contestants"]
    out = []
    if "A_last_increment" in spec:
        out.append(baselines.LastIncrement(**(spec["A_last_increment"] or {})))
    if "B_age_curve" in spec:
        out.append(baselines.AgeCurve(**(spec["B_age_curve"] or {})))
    if "C_hazard_macro" in spec:
        out.append(
            hazard.HazardRegression(
                feats, seed=cfg.get("seed", 0), **(spec["C_hazard_macro"] or {})
            )
        )
    for name in spec.get("tsfm", []) or []:
        out.append(TSFMContestant(make_forecaster(name)))
    if not out:
        raise cfgmod.ConfigError("no contestants configured")
    return out


def data_checksum(files: dict[str, Path]) -> str:
    """One checksum for all inputs of a run, to pin in a lock."""
    joined = "|".join(f"{k}={provenance.file_sha256(p)}" for k, p in sorted(files.items()))
    return provenance.config_hash({"inputs": joined})


def seal(cfg: dict[str, Any], note: str = "") -> lock.Lock:
    """Seal a config and its prepared data before any test score exists (CLAUDE.md rule 3)."""
    _, _, files = load_prepared(cfg)
    return lock.seal(cfg, cfg.get("lock_dir", "locks"), note=note, data_sha256=data_checksum(files))


def run(cfg: dict[str, Any], _allow_untracked_lock_for_tests: bool = False) -> Path:
    """Score the config's evaluation period; returns the run folder."""
    cfgmod.validate(cfg)
    ev = cfg["evaluation"]
    period = ev["period"]
    curves, feats, files = load_prepared(cfg)
    first, last = cfgmod.origin_window(cfg, period)
    os_ = origins.monthly_origins(first, last, int(ev["step_months"]), int(ev["horizon"]))
    contestants = build_contestants(cfg, feats)
    chash = provenance.config_hash(cfg)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{cfg['name']}_{period}_{chash[:8]}"
    out_dir = Path(cfg.get("artifacts_dir", "artifacts/runs")) / run_id

    def execute(sealed: dict | None) -> Path:
        t0 = time.perf_counter()
        quantiles = list(ev["quantiles"])
        preds = harness.run(
            curves, contestants, os_, int(ev["horizon"]), quantiles, int(ev.get("min_history", 6))
        )
        scores = scoring.score(
            preds, quantiles, int(ev["horizon"]), int(ev["step_months"]), cfgmod.exclusions(cfg)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        preds.write_parquet(out_dir / "predictions.parquet")
        revisions = None if feats is None else bool(feats["revisions_ignored"].any())
        result = {
            "run_id": run_id,
            "config_name": cfg["name"],
            "config_hash": chash,
            "period": period,
            "role": {"dev": "development", "test": "sealed test", "recession": "descriptive"}[
                period
            ],
            "synthetic": cfg["data"]["source"] == "synthetic",
            "macro_mode": cfg.get("macro", {}).get("mode", "asof"),
            "macro_revisions_ignored": revisions,
            "sealed": sealed,
            "n_origins": len(os_),
            "origins": [o.month.isoformat() for o in os_],
            "contestants": [c.describe() for c in contestants],
            "n_contestants": len(contestants),
            "scores": scores,
            "seconds": round(time.perf_counter() - t0, 2),
            "provenance": provenance.stamp(
                cfg, files, packages=("reserve", "cleanroom", "polars", "numpy", "scipy")
            ),
            "config": cfg,
        }
        (out_dir / "results.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        return out_dir

    if period in cfgmod.SEALED_PERIODS:
        with guard.sealed_run(
            cfg,
            cfg.get("lock_dir", "locks"),
            run_id,
            data_sha256=data_checksum(files),
            allow_untracked_lock_for_tests=_allow_untracked_lock_for_tests,
        ) as sealed_run:
            out = execute(
                {"run_number": sealed_run.run_number, "git_commit": sealed_run.git_commit}
            )
            sealed_run.finish(results=str(out / "results.json"))
            return out
    return execute(None)
