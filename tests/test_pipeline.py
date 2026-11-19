"""Configs, the prepare and run pipeline, the sealed test guard, and the CLI."""

import copy
import json

import polars as pl
import pytest
from cleanroom.guard import SealedRunError

from reserve import cli, pipeline
from reserve import config as cfgmod
from reserve.data import synthetic


def small_cfg(tmp_path, **overrides):
    cfg = {
        "name": "t_small",
        "seed": 0,
        "data": {
            "source": "synthetic",
            "raw_dir": str(tmp_path / "raw"),
            "interim_dir": str(tmp_path / "interim"),
            "synthetic": {
                "first_vintage": "2004Q1",
                "n_vintages": 20,
                "loans_per_vintage": 150,
                "data_end": "2010-12",
                "seed": 1,
            },
        },
        "macro": {"mode": "asof"},
        "evaluation": {
            "period": "dev",
            "origins": {"dev": {"first": "2007-03", "last": "2009-06"}},
            "step_months": 3,
            "horizon": 12,
            "min_history": 6,
            "quantiles": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            "exclude_target_windows": [],
        },
        "contestants": {
            "A_last_increment": {"window": 3},
            "B_age_curve": {"donors": 8},
            "C_hazard_macro": {"draws": 50},
            "tsfm": ["fake"],
        },
        "lock_dir": str(tmp_path / "locks"),
        "artifacts_dir": str(tmp_path / "runs"),
    }
    cfg.update(overrides)
    return cfg


def test_validation_rules(tmp_path):
    cfgmod.validate(small_cfg(tmp_path))
    bad = small_cfg(tmp_path)
    bad["evaluation"]["quantiles"] = [0.9, 0.1]
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.validate(bad)
    sealed_synth = small_cfg(tmp_path)
    sealed_synth["evaluation"]["period"] = "test"
    sealed_synth["evaluation"]["origins"]["test"] = {"first": "2010-06", "last": "2010-09"}
    with pytest.raises(cfgmod.ConfigError, match="never scored as a sealed test"):
        cfgmod.validate(sealed_synth)
    overlap = small_cfg(tmp_path)
    overlap["evaluation"]["origins"]["test"] = {"first": "2009-09", "last": "2010-06"}
    with pytest.raises(cfgmod.ConfigError, match="reach past the first test origin"):
        cfgmod.validate(overlap)


def test_shipped_configs_are_valid():
    for name in ("synthetic_small", "synthetic_tsfm", "freddie_sample"):
        cfgmod.load(f"configs/{name}.yaml")


def test_prepare_and_run_synthetic(tmp_path):
    cfg = small_cfg(tmp_path)
    sums = pipeline.prepare(cfg)
    assert set(sums) == {"curves.parquet", "loans.parquet", "macro_features.parquet"}
    out = pipeline.run(cfg)
    res = json.loads((out / "results.json").read_text())
    assert res["synthetic"] is True and res["sealed"] is None and res["role"] == "development"
    assert res["macro_revisions_ignored"] is False
    prov = res["provenance"]
    assert prov["config_hash"] == res["config_hash"]
    assert {"curves", "macro_features"} <= set(prov["data"])
    assert prov["packages"]["cleanroom"] == "0.2.0"
    assert res["scores"]["comparison"]["n_comparisons"] == 3
    preds = pl.read_parquet(out / "predictions.parquet")
    again = pl.read_parquet(pipeline.run(cfg) / "predictions.parquet")
    assert preds.equals(again)  # deterministic


@pytest.fixture
def freddie_style(tmp_path):
    """Synthetic loans written in the Freddie Mac layout, read through the Freddie path."""
    sim = synthetic.simulate(
        synthetic.SimConfig(
            first_vintage="2004Q1", n_vintages=20, loans_per_vintage=120, data_end="2011-12", seed=2
        )
    )
    synthetic.write(sim, tmp_path / "raw")
    cfg = small_cfg(tmp_path)
    cfg["data"] = {
        "source": "freddie",
        "raw_dir": str(tmp_path / "raw" / "freddie"),
        "fred_dir": str(tmp_path / "raw" / "fred"),
        "interim_dir": str(tmp_path / "interim_f"),
    }
    cfg["evaluation"]["origins"] = {
        "dev": {"first": "2007-03", "last": "2009-06"},
        "test": {"first": "2010-06", "last": "2010-12"},
    }
    pipeline.prepare(cfg)
    return cfg


def test_simulator_files_read_as_freddie_are_labelled_synthetic(freddie_style, capsys):
    """Fake files in the real layout must never produce an unlabelled result (M40)."""
    manifest = pipeline.read_manifest(freddie_style)
    assert manifest["source"] == "freddie" and manifest["synthetic"] is True
    assert manifest["synthetic_records"] == manifest["n_loans"] == 20 * 120
    assert len(manifest["raw_files"]) == 2 * 5  # 20 vintages from 2004Q1: 2004 to 2008
    res = json.loads((pipeline.run(freddie_style) / "results.json").read_text())
    assert res["synthetic"] is True


def test_run_refuses_tables_without_a_manifest(freddie_style):
    from pathlib import Path

    (Path(freddie_style["data"]["interim_dir"]) / pipeline.MANIFEST).unlink()
    with pytest.raises(FileNotFoundError, match="prepare"):
        pipeline.run(freddie_style)


def test_test_period_needs_a_lock(freddie_style):
    cfg = copy.deepcopy(freddie_style)
    cfg["evaluation"]["period"] = "test"
    with pytest.raises(SealedRunError):
        pipeline.run(cfg, _allow_untracked_lock_for_tests=True)


def test_sealed_run_counts_attempts_and_rejects_changed_plans(freddie_style):
    cfg = copy.deepcopy(freddie_style)
    cfg["evaluation"]["period"] = "test"
    lk = pipeline.seal(cfg, note="unit test")
    out = pipeline.run(cfg, _allow_untracked_lock_for_tests=True)
    res = json.loads((out / "results.json").read_text())
    assert res["sealed"]["run_number"] == 1 and res["role"] == "sealed test"
    statuses = [u["status"] for u in lk.uses()]
    assert statuses == ["started", "finished"]
    changed = copy.deepcopy(cfg)
    changed["evaluation"]["horizon"] = 6
    with pytest.raises(SealedRunError):
        pipeline.run(changed, _allow_untracked_lock_for_tests=True)


def test_real_git_checks_refuse_an_uncommitted_lock(freddie_style):
    cfg = copy.deepcopy(freddie_style)
    cfg["evaluation"]["period"] = "test"
    pipeline.seal(cfg)
    with pytest.raises(SealedRunError):
        pipeline.run(cfg)  # the lock in a temp folder is not tracked by git


def test_cli(tmp_path, capsys):
    path = tmp_path / "c.yaml"
    import yaml

    path.write_text(yaml.safe_dump(small_cfg(tmp_path)))
    assert cli.main(["run", str(path)]) == 1  # not prepared yet
    assert "reserve prepare" in capsys.readouterr().err
    assert cli.main(["prepare", "synthetic", "--config", str(path)]) == 0
    assert cli.main(["run", str(path)]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC DATA" in out and "reference:" in out
    assert cli.main(["prepare", "freddie", "--config", str(path)]) == 2  # wrong source


def test_prepare_macro_needs_a_key(monkeypatch, capsys):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert cli.main(["prepare", "macro"]) == 1
    assert "FRED_API_KEY" in capsys.readouterr().err


def test_experiments_run_on_dev_only_and_are_labelled(tmp_path, capsys):
    cfg = small_cfg(tmp_path)
    cfg["experiments"] = {"thin_history": {"months": [6, 12]}, "scenarios": {"names": ["severe"]}}
    cfgmod.validate(cfg)
    pipeline.prepare(cfg)
    thin = json.loads((pipeline.run(cfg, experiment="thin_history") / "results.json").read_text())
    assert thin["experiment"] == "thin_history" and "_dev_thin_history_" in thin["run_id"]
    assert thin["thin_history"]["levels"] == ["6", "12", "full"]
    assert thin["provenance"]["config_hash"] == thin["config_hash"]
    scen = json.loads((pipeline.run(cfg, experiment="scenarios") / "results.json").read_text())
    assert scen["scenarios"]["hypothetical"] is True
    assert list(scen["scenarios"]["scenarios"]) == ["baseline", "severe"]
    c = next(x for x in scen["contestants"] if x["name"] == "C_hazard_macro")
    assert c["scenario"] is None  # the run's own contestant list is the unshocked one
    recession = copy.deepcopy(cfg)
    recession["evaluation"]["period"] = "recession"
    recession["evaluation"]["origins"]["recession"] = {"first": "2007-03", "last": "2008-03"}
    for kind in ("thin_history", "scenarios"):
        with pytest.raises(cfgmod.ConfigError, match=r"dev \(validation\) period only"):
            pipeline.run(recession, experiment=kind)
    with pytest.raises(cfgmod.ConfigError, match="experiment must be one of"):
        pipeline.run(cfg, experiment="peek_at_test")
    import yaml

    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert cli.main(["run", str(path), "--experiment", "scenarios"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC DATA" in out and "HYPOTHETICAL scenario severe" in out


def test_experiment_settings_are_validated(tmp_path):
    for bad in (
        {"thin_history": {"months": [12, 6]}},
        {"thin_history": {"months": [3]}},  # below min_history
        {"scenarios": {"names": ["boom"]}},
        {"peek": {}},
    ):
        cfg = small_cfg(tmp_path)
        cfg["experiments"] = bad
        with pytest.raises(cfgmod.ConfigError):
            cfgmod.validate(cfg)
