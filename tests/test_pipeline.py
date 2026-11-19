"""Configs, the prepare and run pipeline, the sealed test guard, and the CLI."""

import copy
import json
from pathlib import Path

import polars as pl
import pytest
from cleanroom.guard import SealedRunError

from reserve import checksums, cli, pipeline
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
        "checksums": str(tmp_path / "checksums.sha256"),
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
    (Path(freddie_style["data"]["interim_dir"]) / pipeline.MANIFEST).unlink()
    with pytest.raises(FileNotFoundError, match="prepare"):
        pipeline.run(freddie_style)


def test_test_period_needs_a_lock(freddie_style):
    cfg = copy.deepcopy(freddie_style)
    cfg["evaluation"]["period"] = "test"
    with pytest.raises(SealedRunError):
        pipeline.run(cfg, _allow_untracked_lock_for_tests=True)


def test_shipped_dev_and_test_configs_are_the_same_plan():
    """What is tuned on dev is what gets sealed: only the name and the period differ."""
    dev = cfgmod.load("configs/freddie_sample.yaml")
    test = cfgmod.load("configs/freddie_test.yaml")
    assert dev["evaluation"]["period"] == "dev" and test["evaluation"]["period"] == "test"
    assert test["evaluation"]["origins"]["test"] == {"first": "2013-03", "last": "2018-12"}
    assert test["evaluation"]["horizon"] == 12
    for c in (dev, test):
        c.pop("name")
        c["evaluation"].pop("period")
    assert dev == test


def test_seal_refuses_until_the_data_and_plan_are_ready(freddie_style):
    cfg = copy.deepcopy(freddie_style)
    found = " ".join(pipeline.seal_problems(cfg))
    assert "period is 'dev'" in found and "does not exist" in found  # no checksums yet
    assert "synthetic markers" in found
    cfg["evaluation"]["period"] = "test"
    checksums.write(cfg)
    found = pipeline.seal_problems(cfg)
    assert len(found) == 2 and "synthetic markers" in found[0] and "'fake'" in found[1]
    with pytest.raises(pipeline.SealRefused, match="cannot seal yet"):
        pipeline.seal(cfg)
    assert pipeline.seal_problems(cfg, rehearsal=True) == []  # rehearsal: synthetic only
    stub = copy.deepcopy(cfg)
    stub["contestants"]["tsfm"] = ["timesfm_2p5"]
    assert any("stub" in p or "needs" in p for p in pipeline.seal_problems(stub, rehearsal=True))
    # A raw file that changes after its checksum was recorded blocks the seal.
    raw = sorted(Path(cfg["data"]["raw_dir"]).glob("sample_orig_*.txt"))[0]
    raw.write_text(raw.read_text() + "\n")
    found = pipeline.seal_problems(cfg, rehearsal=True)
    assert any("changed since" in p for p in found)
    assert any("prepared from other raw files" in p for p in found)


def test_sealed_run_counts_attempts_and_rejects_changed_plans(freddie_style):
    cfg = copy.deepcopy(freddie_style)
    cfg["evaluation"]["period"] = "test"
    checksums.write(cfg)
    lk = pipeline.seal(cfg, note="unit test", rehearsal=True)
    assert lk.path.read_text().count("REHEARSAL") == 1
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
    checksums.write(cfg)
    pipeline.seal(cfg, rehearsal=True)
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


def test_cli_seal_path_errors_are_messages_not_tracebacks(freddie_style, tmp_path, capsys):
    import yaml

    cfg = copy.deepcopy(freddie_style)
    cfg["evaluation"]["period"] = "test"
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert cli.main(["run", str(path)]) == 1  # no lock: SealedRunError becomes a message
    assert "seal" in capsys.readouterr().err
    assert cli.main(["seal", str(path)]) == 1
    assert "cannot seal yet" in capsys.readouterr().err
    assert cli.main(["checksums", "--config", str(path)]) == 0
    assert cli.main(["seal", str(path), "--rehearsal"]) == 0
    assert "REHEARSAL ON SYNTHETIC DATA" in capsys.readouterr().out
    dev = tmp_path / "d.yaml"
    dev.write_text(yaml.safe_dump(freddie_style))
    with pytest.raises(SystemExit):  # the test period cannot be picked with --period
        cli.main(["run", str(dev), "--period", "test"])


def test_checksum_file_round_trips_and_rejects_bad_lines(freddie_style, tmp_path):
    path, sums = checksums.write(freddie_style)
    assert checksums.read(path) == sums and len(sums) == 2 * 5 + 3  # 5 years + 3 FRED tables
    assert path.read_text().startswith("# SHA-256")
    assert checksums.problems(freddie_style) == []
    path.write_text(path.read_text() + "not a checksum line\n")
    assert "is not" in checksums.problems(freddie_style)[0]


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
