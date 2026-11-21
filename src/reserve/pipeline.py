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
from reserve.evaluation import experiments, harness, origins, scoring
from reserve.models import baselines, hazard
from reserve.models.tsfm import TSFMContestant, make_forecaster
from reserve.stress import scenarios

INTERIM_FILES = ("curves.parquet", "loans.parquet")
# Written by `prepare` next to the tables: what was read and whether it was synthetic (M40).
MANIFEST = "prepare.json"


def _read_freddie(raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, list[Path]]:
    files = loans.find_sample_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"no sample_orig_YYYY.txt / sample_perf_YYYY.txt pairs in {raw_dir}; see "
            "docs/runbook.md, step 1"
        )
    cols = ("first_payment_date", *loans.MARKER_COLUMNS)
    orig = pl.concat([loans.read_origination(o, columns=cols) for _, o, _ in files])
    # One year of monthly rows in memory at a time: each is cut to one row per loan (M47).
    perf = pl.concat(
        [vintages.summarize_performance(loans.read_performance(p)) for _, _, p in files],
        how="diagonal",
    )
    return orig, perf, [p for _, o, pf in files for p in (o, pf)]


def read_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    path = Path(cfg["data"]["interim_dir"]) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `reserve prepare` again (M40)")
    return json.loads(path.read_text(encoding="utf-8"))


def is_synthetic(cfg: dict[str, Any]) -> bool:
    """True for a synthetic config, or for any data whose records carry synthetic markers."""
    if cfg["data"]["source"] == "synthetic":
        return True
    return bool(read_manifest(cfg)["synthetic"])


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
        orig, perf, raw_files = _read_freddie(raw / "freddie")
        observations = macro.load_observations(raw / "fred")
    else:
        orig, perf, raw_files = _read_freddie(Path(data["raw_dir"]))
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
    n_marked = loans.synthetic_records(orig)
    manifest = {
        "source": data["source"],
        "synthetic": data["source"] == "synthetic" or n_marked > 0,
        "synthetic_records": n_marked,
        "n_loans": orig.height,
        "raw_files": [{"name": p.name, "bytes": p.stat().st_size} for p in raw_files],
        "macro_tables": observations is not None,
        "prepared_utc": datetime.now(UTC).isoformat(),
    }
    (interim / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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


def build_contestants(
    cfg: dict[str, Any], feats: pl.DataFrame | None, scenario: scenarios.Scenario | None = None
) -> list:
    """Contestants from the config; `scenario` gives C a HYPOTHETICAL macro path (M35)."""
    spec = cfg["contestants"]
    out = []
    if "A_last_increment" in spec:
        out.append(baselines.LastIncrement(**(spec["A_last_increment"] or {})))
    if "B_age_curve" in spec:
        out.append(baselines.AgeCurve(**(spec["B_age_curve"] or {})))
    if "C_hazard_macro" in spec:
        out.append(
            hazard.HazardRegression(
                feats,
                seed=cfg.get("seed", 0),
                scenario=scenario,
                **(spec["C_hazard_macro"] or {}),
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


class SealRefused(ValueError):
    """The config or its data is not ready to be sealed (methodology M42)."""


REHEARSAL_NOTE = "REHEARSAL ON SYNTHETIC DATA, not a real test plan (M41). "


def seal_problems(cfg: dict[str, Any], rehearsal: bool = False) -> list[str]:
    """Everything that must be true before a plan is sealed; [] when it is ready (M42).

    `rehearsal` is for scripts/rehearse_real_path.py only. It does not loosen any check
    for real data: it is refused unless the data carries synthetic markers, and it lets
    the `fake` stand-in TSFM through so the rehearsal runs without the `tsfm` extra.
    """
    from reserve import checksums
    from reserve.models.tsfm import MODEL_INFO, missing_packages

    found: list[str] = []
    if cfg["data"]["source"] != "freddie":
        found.append("only a Freddie Mac config is sealed; synthetic configs are never tested")
        return found
    period = cfg["evaluation"]["period"]
    if period not in cfgmod.SEALED_PERIODS:
        found.append(
            f"evaluation.period is {period!r}; seal the test config itself (configs/"
            "freddie_test.yaml). The period is part of the hashed plan, so a lock made for "
            "another period can never open the test"
        )
    try:
        load_prepared(cfg)
        synthetic_data = is_synthetic(cfg)
    except FileNotFoundError as err:
        found.append(f"{err}")
        return found
    if synthetic_data and not rehearsal:
        found.append(
            "the prepared tables carry synthetic markers (M40): only Freddie Mac's own files "
            "can be sealed"
        )
    if rehearsal and not synthetic_data:
        found.append("--rehearsal seals synthetic rehearsal data only; seal real data without it")
    found += checksums.problems(cfg)
    raw = {(f.name, f.stat().st_size) for f in checksums.raw_files(cfg)["freddie"]}
    prepared = {(f["name"], f["bytes"]) for f in read_manifest(cfg)["raw_files"]}
    if raw != prepared:
        found.append("the interim tables were prepared from other raw files; run `reserve prepare`")
    spec = cfg["contestants"]
    for name in ("A_last_increment", "B_age_curve", "C_hazard_macro"):
        if name not in spec:
            found.append(f"the sealed plan needs every classical baseline; {name} is missing")
    tsfms = spec.get("tsfm") or []
    if not tsfms:
        found.append("the sealed plan has no TSFM contestant")
    for name in tsfms:
        if name == "fake" and rehearsal:
            continue
        info = MODEL_INFO.get(name)
        if info is None or info["status"] != "real":
            status = "unknown" if info is None else info["status"]
            found.append(f"TSFM {name!r} is {status}, not a real model; remove it before sealing")
        elif missing := missing_packages(name):
            found.append(f"TSFM {name!r} needs {missing}; install its extra before sealing")
    return found


def seal(cfg: dict[str, Any], note: str = "", rehearsal: bool = False) -> lock.Lock:
    """Seal a config and its prepared data before any test score exists (CLAUDE.md rule 3).

    Refuses (`SealRefused`) unless `seal_problems` finds nothing, so a lock always names
    real, checksummed data and a plan that can run.
    """
    found = seal_problems(cfg, rehearsal=rehearsal)
    if found:
        raise SealRefused("cannot seal yet:\n  - " + "\n  - ".join(found))
    _, _, files = load_prepared(cfg)
    note = (REHEARSAL_NOTE if rehearsal else "") + note
    return lock.seal(cfg, cfg.get("lock_dir", "locks"), note=note, data_sha256=data_checksum(files))


def run(
    cfg: dict[str, Any],
    _allow_untracked_lock_for_tests: bool = False,
    experiment: str | None = None,
) -> Path:
    """Score the config's evaluation period; returns the run folder.

    `experiment` ("thin_history" or "scenarios") runs a harness option instead of the
    ordinary scoring; it is refused unless the period is dev (methodology M34, M35).
    """
    cfgmod.validate(cfg)
    ev = cfg["evaluation"]
    period = ev["period"]
    if experiment is not None:
        if experiment not in experiments.KINDS:
            raise cfgmod.ConfigError(f"experiment must be one of {experiments.KINDS}")
        if period != "dev":
            raise cfgmod.ConfigError(
                f"the {experiment} experiment runs on the dev (validation) period only, not "
                f"{period!r}"
            )
    curves, feats, files = load_prepared(cfg)
    synthetic_data = is_synthetic(cfg)
    first, last = cfgmod.origin_window(cfg, period)
    os_ = origins.monthly_origins(first, last, int(ev["step_months"]), int(ev["horizon"]))
    contestants = build_contestants(cfg, feats)
    chash = provenance.config_hash(cfg)
    kind = f"_{experiment}" if experiment else ""
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{cfg['name']}_{period}{kind}_{chash[:8]}"
    out_dir = Path(cfg.get("artifacts_dir", "artifacts/runs")) / run_id
    exp_cfg = (cfg.get("experiments") or {}).get(experiment or "", {}) or {}

    def with_scenario(name: str) -> list:
        """Same contestant objects, except C, which gets the scenario's macro path."""
        c_spec = cfg["contestants"].get("C_hazard_macro") or {}
        return [
            hazard.HazardRegression(
                feats, seed=cfg.get("seed", 0), scenario=scenarios.get(name), **c_spec
            )
            if c.name == "C_hazard_macro"
            else c
            for c in contestants
        ]

    def compute(quantiles: list[float]) -> tuple[pl.DataFrame, dict]:
        horizon, step = int(ev["horizon"]), int(ev["step_months"])
        min_history = int(ev.get("min_history", 6))
        if experiment == "thin_history":
            preds, payload = experiments.run_thin_history(
                curves,
                lambda _key: contestants,
                os_,
                horizon,
                quantiles,
                step,
                min_history,
                months=exp_cfg.get("months", experiments.DEFAULT_THIN_MONTHS),
                exclude=cfgmod.exclusions(cfg),
            )
            return preds, {"thin_history": payload}
        if experiment == "scenarios":
            preds, payload = experiments.run_scenarios(
                curves,
                with_scenario,
                os_,
                horizon,
                quantiles,
                min_history,
                names=exp_cfg.get("names", experiments.DEFAULT_SCENARIOS),
            )
            return preds, {"scenarios": payload}
        preds = harness.run(curves, contestants, os_, horizon, quantiles, min_history)
        return preds, {
            "scores": scoring.score(preds, quantiles, horizon, step, cfgmod.exclusions(cfg))
        }

    def execute(sealed: dict | None) -> Path:
        t0 = time.perf_counter()
        quantiles = list(ev["quantiles"])
        preds, payload = compute(quantiles)
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
            "synthetic": synthetic_data,
            "macro_mode": cfg.get("macro", {}).get("mode", "asof"),
            "macro_revisions_ignored": revisions,
            "sealed": sealed,
            "n_origins": len(os_),
            "origins": [o.month.isoformat() for o in os_],
            "contestants": [c.describe() for c in contestants],
            "n_contestants": len(contestants),
            "experiment": experiment,
            **payload,
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
