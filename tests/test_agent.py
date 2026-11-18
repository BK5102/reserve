"""Agent tools, runner, critic, adversarial pass, and reporter (methodology M37 to M39).

The cassettes are written by hand in this file to mimic the shape of Claude's responses.
They are not recordings of the real API, and no test calls it. Recorded cassettes can be
added once a key is available (`reserve agent ... --record path.json`).
"""

import copy
import json
from importlib.util import find_spec
from pathlib import Path

import polars as pl
import pytest
import yaml

from reserve import cli, pipeline
from reserve.agent import critic, llm, reporter, runner
from reserve.agent.planner import SYSTEM
from reserve.agent.tools import Toolbox, ToolDenied
from reserve.data import synthetic


def _cfg(tmp: Path, name: str = "agent_small") -> dict:
    return {
        "name": name,
        "seed": 0,
        "data": {
            "source": "synthetic",
            "raw_dir": str(tmp / "raw"),
            "interim_dir": str(tmp / "interim"),
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
            "origins": {
                "dev": {"first": "2007-03", "last": "2009-06"},
                "recession": {"first": "2006-03", "last": "2006-09"},
            },
            "step_months": 3,
            "horizon": 12,
            "min_history": 6,
            "quantiles": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            "exclude_target_windows": [],
        },
        "experiments": {"thin_history": {"months": [6, 12]}},
        "contestants": {
            "A_last_increment": {"window": 3},
            "B_age_curve": {"donors": 8},
            "C_hazard_macro": {"draws": 50},
            "tsfm": ["fake"],
        },
        "lock_dir": str(tmp / "locks"),
        "artifacts_dir": str(tmp / "runs"),
    }


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("agent")
    configs = tmp / "configs"
    configs.mkdir()
    cfg = _cfg(tmp)
    (configs / "agent_small.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    pipeline.prepare(cfg)
    box = Toolbox(configs_dir=configs, runs_dir=tmp / "runs")
    std = box.call(
        "run_forecast_experiment",
        {"config": "agent_small", "period": "dev", "experiment": "standard"},
    )
    thin = box.call(
        "run_forecast_experiment",
        {"config": "agent_small", "period": "dev", "experiment": "thin_history"},
    )
    scen = box.call(
        "run_forecast_experiment",
        {"config": "agent_small", "period": "dev", "experiment": "scenarios"},
    )
    return {"tmp": tmp, "cfg": cfg, "box": box, "std": std, "thin": thin, "scen": scen}


def _claim(lab, contestant: str = "D_tsfm_fake") -> dict:
    comp = lab["std"]["scores"]["comparison"]
    entry = next(c for c in comp["comparisons"] if c["contestant"] == contestant)
    return {
        "contestant": contestant,
        "metric": "mean_quantile_loss",
        "run_id": lab["std"]["run_id"],
        "direction": critic.direction(entry),
    }


def _final(lab, prose: str | None = None, claims=None, runs=None) -> str:
    prose = prose or (
        f"On the synthetic portfolio (a pipeline check, not a result), D_tsfm_fake was compared "
        f"with the reference in run {lab['std']['run_id']}; see the backtest table for "
        "mean_quantile_loss and interval coverage. The thin-history run "
        f"{lab['thin']['run_id']} shows the learning_curve. The hypothetical scenario run "
        f"{lab['scen']['run_id']} shows that only C_hazard_macro responds."
    )
    runs = runs if runs is not None else [lab[k]["run_id"] for k in ("std", "thin", "scen")]
    claims = claims if claims is not None else [_claim(lab)]
    return prose + "\n" + json.dumps({"runs": runs, "claims": claims})


def _cassette(path: Path, responses: list[dict]) -> Path:
    path.write_text(
        json.dumps([{"request_digest": "any", "response": r} for r in responses]), encoding="utf-8"
    )
    return path


def _loop_responses(lab, final: str) -> list[dict]:
    rid = lab["std"]["run_id"]
    return [
        {"stop_reason": "tool_use", "content": [
            {"type": "text", "text": "I will read the existing run first."},
            {"type": "tool_use", "id": "t1", "name": "load_results", "input": {"run_id": rid}},
        ]},
        {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "data/raw"}},
            {"type": "tool_use", "id": "t3", "name": "run_forecast_experiment",
             "input": {"config": "agent_small", "period": "test", "experiment": "standard"}},
        ]},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": final}]},
    ]  # fmt: skip


def _objections(lab, n: int = 3, path: str | None = None, text: str | None = None) -> dict:
    rid = lab["std"]["run_id"]
    paths = [
        "scores.contestants.D_tsfm_fake.coverage_90.coverage",
        "scores.comparison.comparisons.0.holm_p",
        "n_origins",
    ]
    objs = [
        {
            "claim": 0,
            "objection": text or "The evidence comes from a synthetic development run only.",
            "evidence": {"run_id": rid, "path": path or paths[i % 3]},
        }
        for i in range(n)
    ]
    return {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": json.dumps({"objections": objs})}],
    }


# ----- tools ----------------------------------------------------------------------------


def test_only_four_tools_exist(lab):
    box = lab["box"]
    assert [s["name"] for s in box.specs()] == [
        "run_forecast_experiment",
        "load_results",
        "plot",
        "compute_stats",
    ]
    for name in ("shell", "read_file", "fetch_url", "write_file", "run_experiment"):
        with pytest.raises(ToolDenied, match="allow-list"):
            box.call(name, {})


def test_bad_ids_configs_and_arguments_are_denied(lab):
    box = lab["box"]
    for bad in ("../../data/raw", lab["std"]["run_id"] + "/../..", "predictions"):
        with pytest.raises(ToolDenied):
            box.call("load_results", {"run_id": bad})
    with pytest.raises(ToolDenied, match="unknown config"):
        box.call(
            "run_forecast_experiment", {"config": "../x", "period": "dev", "experiment": "standard"}
        )
    with pytest.raises(ToolDenied, match="unexpected arguments"):
        box.call("load_results", {"run_id": lab["std"]["run_id"], "path": "x"})
    with pytest.raises(ToolDenied, match="dev period only"):
        box.call(
            "run_forecast_experiment",
            {"config": "agent_small", "period": "recession", "experiment": "thin_history"},
        )


def test_test_period_needs_the_flag_and_then_a_lock(lab, tmp_path):
    box = lab["box"]
    spec = next(s for s in box.specs() if s["name"] == "run_forecast_experiment")
    assert "test" not in spec["input_schema"]["properties"]["period"]["enum"]
    with pytest.raises(ToolDenied, match="--allow-test"):
        box.call(
            "run_forecast_experiment",
            {"config": "agent_small", "period": "test", "experiment": "standard"},
        )
    # With the flag, a synthetic config is still refused (it can never be a sealed test) ...
    flagged = Toolbox(configs_dir=box.configs_dir, runs_dir=box.runs_dir, allow_test=True)
    with pytest.raises(ToolDenied, match="config refused"):
        flagged.call(
            "run_forecast_experiment",
            {"config": "agent_small", "period": "test", "experiment": "standard"},
        )
    # ... and real-layout data without a committed lock is refused by the sealed-run guard.
    sim = synthetic.simulate(
        synthetic.SimConfig(
            first_vintage="2004Q1", n_vintages=16, loans_per_vintage=80, data_end="2011-12", seed=2
        )
    )
    synthetic.write(sim, tmp_path / "raw")
    cfg = _cfg(tmp_path, "real_layout")
    cfg["data"] = {
        "source": "freddie",
        "raw_dir": str(tmp_path / "raw" / "freddie"),
        "fred_dir": str(tmp_path / "raw" / "fred"),
        "interim_dir": str(tmp_path / "interim_f"),
    }
    cfg["evaluation"]["origins"]["test"] = {"first": "2010-06", "last": "2010-12"}
    pipeline.prepare(cfg)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "real_layout.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    real = Toolbox(configs_dir=configs, runs_dir=tmp_path / "runs", allow_test=True)
    with pytest.raises(ToolDenied, match="test scoring refused"):
        real.call(
            "run_forecast_experiment",
            {"config": "real_layout", "period": "test", "experiment": "standard"},
        )
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())


def test_results_never_include_rows_and_unsealed_test_runs_are_unreadable(lab):
    box = lab["box"]
    view = box.call("load_results", {"run_id": lab["std"]["run_id"]})
    assert "config" not in view and "provenance" not in view and view["synthetic"] is True
    assert not {"y", "predictions"} & set(view)
    fake = copy.deepcopy(
        json.loads((box.runs_dir / lab["std"]["run_id"] / "results.json").read_text())
    )
    fake_id = lab["std"]["run_id"].replace("_dev_", "_test_")
    fake.update(run_id=fake_id, period="test", sealed=None)
    (box.runs_dir / fake_id).mkdir()
    (box.runs_dir / fake_id / "results.json").write_text(json.dumps(fake), encoding="utf-8")
    for tool, args in (
        ("load_results", {}),
        ("compute_stats", {"horizon": 6}),
        ("plot", {"kind": "coverage"}),
    ):
        with pytest.raises(ToolDenied, match="without a lock"):
            box.call(tool, {"run_id": fake_id, **args})


def test_compute_stats_and_plots(lab):
    box = lab["box"]
    rid = lab["std"]["run_id"]
    six = box.call("compute_stats", {"run_id": rid, "horizon": 6})
    assert six["headline_horizon"] == 6 and six["comparison"]["n_comparisons"] == 3
    for bad in (0, 13):
        with pytest.raises(ToolDenied, match="horizon"):
            box.call("compute_stats", {"run_id": rid, "horizon": bad})
    with pytest.raises(ToolDenied, match="standard runs"):
        box.call("compute_stats", {"run_id": lab["thin"]["run_id"], "horizon": 6})
    loss = box.call("plot", {"run_id": rid, "kind": "loss_by_horizon"})
    assert loss["x"] == list(range(1, 13)) and loss["title"].startswith("SYNTHETIC")
    curve = box.call("plot", {"run_id": lab["thin"]["run_id"], "kind": "learning_curve"})
    assert curve["x"] == ["6", "12", "full"]
    scen = box.call("plot", {"run_id": lab["scen"]["run_id"], "kind": "scenarios"})
    assert scen["label"] == "HYPOTHETICAL" and "HYPOTHETICAL" in scen["title"]
    with pytest.raises(ToolDenied):
        box.call("plot", {"run_id": rid, "kind": "learning_curve"})


# ----- runner ---------------------------------------------------------------------------


def test_loop_executes_allowed_tools_and_denies_others(lab, tmp_path):
    final = _final(lab)
    path = _cassette(tmp_path / "c.json", _loop_responses(lab, final))
    result = runner.run_agent("Does the TSFM help?", lab["box"], llm.CassetteLLM(path))
    assert [c["tool"] for c in result.tool_calls] == [
        "load_results",
        "read_file",
        "run_forecast_experiment",
    ]
    reasons = " ".join(d["reason"] for d in result.denied)
    assert "allow-list" in reasons and "--allow-test" in reasons
    assert result.final_text == final


def test_tool_budget_stops_the_loop(lab, tmp_path):
    path = _cassette(tmp_path / "c.json", _loop_responses(lab, _final(lab)))
    result = runner.run_agent("q", lab["box"], llm.CassetteLLM(path), max_tool_calls=1)
    assert result.stopped_by_budget and len(result.tool_calls) == 1


def test_replay_rejects_a_changed_prompt(tmp_path):
    recorded = llm.request_digest(SYSTEM, [{"role": "user", "content": "original"}], [])
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            [{"request_digest": recorded, "response": {"stop_reason": "end_turn", "content": []}}]
        )
    )
    with pytest.raises(RuntimeError, match="differs from the recording"):
        llm.CassetteLLM(path).create(
            system=SYSTEM, messages=[{"role": "user", "content": "changed"}], tools=[]
        )


def test_live_backend_needs_the_extra():
    assert llm.MODEL == "claude-opus-5-5"
    if find_spec("anthropic") is not None:
        pytest.skip("anthropic is installed; constructing would need a credential")
    with pytest.raises(ImportError, match="--extra agent"):
        llm.AnthropicLLM()


# ----- critic ---------------------------------------------------------------------------


def test_critic_passes_a_good_answer(lab):
    report = critic.review(_final(lab), lab["box"])
    assert report.passed, [c for c in report.checks if not c["ok"]]
    names = " ".join(c["check"] for c in report.checks)
    for expected in (
        "split order",
        "multiple-comparison correction",
        "contamination entry",
        "synthetic run labelled",
    ):
        assert expected in names


@pytest.mark.parametrize(
    "phrase", ["it was 35% better", "about two thirds", "roughly Half the loss", "it doubled"]
)
def test_critic_rejects_typed_numbers(lab, phrase):
    text = _final(lab).replace("see the backtest table", phrase)
    report = critic.review(text, lab["box"])
    assert any(c["check"] == "no typed numbers" and not c["ok"] for c in report.checks)


def test_critic_requires_labels(lab):
    text = _final(lab).replace("synthetic portfolio (a pipeline check, not a result)", "portfolio")
    report = critic.review(text, lab["box"])
    assert any(
        c["check"].startswith("synthetic run labelled") and not c["ok"] for c in report.checks
    )
    text = _final(lab).replace("hypothetical scenario", "scenario")
    assert not critic.review(text, lab["box"]).passed


def test_critic_checks_claim_direction_and_scope(lab):
    claim = _claim(lab)
    wrong = {**claim, "direction": "better" if claim["direction"] != "better" else "worse"}
    assert not critic.review(_final(lab, claims=[wrong]), lab["box"]).passed
    ref = lab["std"]["scores"]["comparison"]["reference"]
    assert not critic.review(_final(lab, claims=[{**claim, "contestant": ref}]), lab["box"]).passed
    on_scen = {**claim, "run_id": lab["scen"]["run_id"]}
    assert not critic.review(_final(lab, claims=[on_scen]), lab["box"]).passed
    assert not critic.review(
        _final(lab, claims=[{**claim, "metric": "coverage_90"}]), lab["box"]
    ).passed


def test_critic_flags_a_coverage_gap_left_unmentioned(lab):
    gaps = critic._coverage_gaps(lab["box"].summary(lab["std"]["run_id"]))
    text = _final(lab).replace(" and interval coverage", "")
    report = critic.review(text, lab["box"])
    failed = [c["check"] for c in report.checks if not c["ok"]]
    if gaps:
        assert any(f.startswith("coverage gap flagged") for f in failed)
    else:
        assert report.passed


def test_critic_catches_a_broken_split(lab, tmp_path):
    box = lab["box"]
    rid = lab["std"]["run_id"]
    path = box.runs_dir / rid / "predictions.parquet"
    original = pl.read_parquet(path)
    try:
        original.with_columns(pl.col("origin").alias("target_period")).write_parquet(path)
        report = critic.review(_final(lab), box)
        assert any(c["check"] == f"split order: {rid}" and not c["ok"] for c in report.checks)
    finally:
        original.write_parquet(path)


# ----- adversarial pass and reporter ----------------------------------------------------


def test_challenge_runs_only_after_the_deterministic_checks(lab, tmp_path):
    bad = critic.review(
        _final(lab).replace("see the backtest table", "it was 35% better"), lab["box"]
    )
    path = _cassette(tmp_path / "c.json", [_objections(lab)])
    out = critic.challenge(_final(lab), lab["box"], llm.CassetteLLM(path), bad)
    assert not out.ran and not out.passed


def test_challenge_validates_and_resolves_every_objection(lab, tmp_path):
    good = critic.review(_final(lab), lab["box"])
    path = _cassette(tmp_path / "c.json", [_objections(lab)])
    out = critic.challenge(_final(lab), lab["box"], llm.CassetteLLM(path), good)
    assert out.ran and out.passed, [c for c in out.checks if not c["ok"]]
    s = lab["box"].summary(lab["std"]["run_id"])
    assert (
        out.objections[0]["value"]
        == s["scores"]["contestants"]["D_tsfm_fake"]["coverage_90"]["coverage"]
    )
    assert out.objections[2]["value"] == s["n_origins"]
    for responses in (
        [_objections(lab, n=2)],
        [_objections(lab, path="scores.no_such_key")],
        [_objections(lab, text="Coverage was only 40% so the claim fails.")],
    ):
        p = _cassette(tmp_path / "bad.json", responses)
        assert not critic.challenge(_final(lab), lab["box"], llm.CassetteLLM(p), good).passed


def test_pack_has_every_section_numbers_from_files_and_labels(lab, tmp_path):
    final = _final(lab)
    path = _cassette(tmp_path / "c.json", [*_loop_responses(lab, final), _objections(lab)])
    model = llm.CassetteLLM(path)
    result = runner.run_agent("Does the TSFM help?", lab["box"], model)
    review = critic.review(result.final_text, lab["box"])
    ch = critic.challenge(result.final_text, lab["box"], model, review)
    pack = reporter.build_pack(result, review, ch, lab["box"])
    for section in (
        "Purpose and scope",
        "Data",
        "Method",
        "Backtest",
        "Sensitivity",
        "Limitations",
        "Monitoring plan",
        "Open issues",
    ):
        assert f". {section}" in pack
    assert pack.count(reporter.NOT_COMPLIANCE) == 2
    assert "SYNTHETIC DATA" in pack and "HYPOTHETICAL" in pack
    s = lab["box"].summary(lab["std"]["run_id"])
    ref = s["scores"]["comparison"]["reference"]
    assert f"{s['scores']['contestants'][ref]['mean_quantile_loss']:.6f}" in pack
    assert "denied `read_file`" in pack and "Evidence `" in pack
    assert s["provenance"]["data"]["curves"]["sha256"][:12] in pack  # checksum, by code
    assert "pretraining cutoff not applicable" in pack  # the fake model's MODEL_INFO entry


def test_cli_agent_with_a_cassette_writes_a_pack(lab, tmp_path, monkeypatch, capsys):
    final = _final(lab)
    path = _cassette(tmp_path / "c.json", [*_loop_responses(lab, final), _objections(lab)])
    box = lab["box"]
    # The command uses the default folders (configs/, artifacts/runs/) under the working
    # directory, so run it from the lab folder with the runs copied into place.
    monkeypatch.chdir(box.configs_dir.parent)
    (box.configs_dir.parent / "artifacts").mkdir(exist_ok=True)
    link = box.configs_dir.parent / "artifacts" / "runs"
    if not link.exists():
        link.mkdir()
        for run in box.runs_dir.iterdir():
            dest = link / run.name
            dest.mkdir()
            for f in run.iterdir():
                (dest / f.name).write_bytes(f.read_bytes())
    out = tmp_path / "pack.md"
    code = cli.main(["agent", "Does the TSFM help?", "--cassette", str(path), "--out", str(out)])
    assert code == 0, capsys.readouterr()
    assert reporter.NOT_COMPLIANCE in out.read_text(encoding="utf-8")
