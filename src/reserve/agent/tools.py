"""The agent's tool layer: four tools, nothing else (CLAUDE.md agent safety rules; M37).

The agent never gets a shell, file paths, or network access. It may only:

- `run_forecast_experiment(config, period, experiment)`: run a config from `configs/` by
  name. The sealed test period is not even offered unless the Toolbox was built with
  `allow_test=True` (`reserve agent --allow-test`, a person's decision), and even then the
  pipeline refuses to score it without a committed lock for that exact config and data.
  The thin-history and scenario experiments run on the dev period only.
- `load_results(run_id)`: a finished run's summary numbers. Predictions and outcomes are
  never returned.
- `plot(run_id, kind)`: chart data built from a run's summary.
- `compute_stats(run_id, horizon)`: scores and Holm-corrected comparisons at another
  horizon, computed by code from the run's saved predictions. Refused for a test-period run
  without a lock record, so test actuals cannot be read before the lock exists.

Every argument is checked against an allow-list or a strict pattern before anything runs.
Anything else raises `ToolDenied`.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RUN_ID = re.compile(
    r"^[0-9]{8}T[0-9]{6}Z_[a-z0-9_]+_(dev|test|recession)(_thin_history|_scenarios)?_[0-9a-f]{8}$"
)
CONFIG_NAME = re.compile(r"^[a-z0-9_]+$")
EXPERIMENTS = ("standard", "thin_history", "scenarios")
PLOTS = ("loss_by_horizon", "coverage", "learning_curve", "scenarios")
VIEW_KEYS = (
    "run_id", "config_name", "config_hash", "period", "role", "synthetic", "experiment",
    "macro_mode", "macro_revisions_ignored", "sealed", "n_origins", "contestants",
    "n_contestants", "scores", "thin_history", "scenarios",
)  # fmt: skip


class ToolDenied(PermissionError):
    """The agent asked for something outside its allow-list."""


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


@dataclass
class Toolbox:
    configs_dir: Path = Path("configs")
    runs_dir: Path = Path("artifacts/runs")
    runner: Callable[..., Path] | None = None  # injected in tests; default pipeline.run
    # The sealed test period is spent by a person's decision, not the agent's.
    allow_test: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)  # audit trail of every call

    def periods(self) -> list[str]:
        return ["dev", "recession", "test"] if self.allow_test else ["dev", "recession"]

    def config_names(self) -> list[str]:
        """Experiment configs only: files with an `evaluation` block. Settings files such as
        configs/agent.yaml (the spend cap) and configs/synthetic_fit.yaml are never offered."""
        import yaml

        names = []
        for p in sorted(self.configs_dir.glob("*.yaml")):
            try:
                raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            except yaml.YAMLError:
                continue
            if isinstance(raw, dict) and "evaluation" in raw:
                names.append(p.stem)
        return names

    # ----- allow-list -----
    def specs(self) -> list[dict[str, Any]]:
        """Tool descriptions in the JSON-schema shape language-model APIs expect."""
        test_note = (
            " The test period is scored only if the config was sealed and the lock committed."
            if self.allow_test
            else ""
        )
        return [
            {
                "name": "run_forecast_experiment",
                "description": (
                    "Run an experiment config by name. 'standard' scores every contestant; "
                    "'thin_history' and 'scenarios' (HYPOTHETICAL, never scored) run on the dev "
                    "period only." + test_note
                ),
                "input_schema": _schema(
                    {
                        "config": {"type": "string", "enum": self.config_names()},
                        "period": {"type": "string", "enum": self.periods()},
                        "experiment": {"type": "string", "enum": list(EXPERIMENTS)},
                    },
                    ["config", "period", "experiment"],
                ),
            },
            {
                "name": "load_results",
                "description": "Read the summary numbers of a finished run (no raw rows).",
                "input_schema": _schema({"run_id": {"type": "string"}}, ["run_id"]),
            },
            {
                "name": "plot",
                "description": "Chart data from a finished run's summary.",
                "input_schema": _schema(
                    {"run_id": {"type": "string"}, "kind": {"type": "string", "enum": list(PLOTS)}},
                    ["run_id", "kind"],
                ),
            },
            {
                "name": "compute_stats",
                "description": (
                    "Scores and Holm-corrected comparisons of a standard run at another "
                    "forecast horizon (months ahead)."
                ),
                "input_schema": _schema(
                    {"run_id": {"type": "string"}, "horizon": {"type": "integer"}},
                    ["run_id", "horizon"],
                ),
            },
        ]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The only way the agent touches anything."""
        self.calls.append({"tool": name, "args": dict(args)})
        handlers = {
            "run_forecast_experiment": self._run_forecast_experiment,
            "load_results": self._load_results,
            "plot": self._plot,
            "compute_stats": self._compute_stats,
        }
        if name not in handlers:
            raise ToolDenied(f"tool {name!r} is not on the allow-list {sorted(handlers)}")
        spec = next(s for s in self.specs() if s["name"] == name)
        allowed = set(spec["input_schema"]["properties"])
        extra = set(args) - allowed
        if extra:
            raise ToolDenied(f"unexpected arguments {sorted(extra)} for {name}")
        missing = set(spec["input_schema"]["required"]) - set(args)
        if missing:
            raise ToolDenied(f"missing arguments {sorted(missing)} for {name}")
        return handlers[name](**args)

    # ----- tools -----
    def _run_forecast_experiment(self, config: str, period: str, experiment: str) -> dict:
        if not CONFIG_NAME.match(str(config)) or config not in self.config_names():
            raise ToolDenied(f"unknown config {config!r}; choose from {self.config_names()}")
        if period == "test" and not self.allow_test:
            raise ToolDenied(
                "test scoring refused: the agent may score the test period only when a person "
                "starts it with --allow-test, and only under a committed lock"
            )
        if period not in self.periods():
            raise ToolDenied(f"period must be one of {self.periods()}")
        if experiment not in EXPERIMENTS:
            raise ToolDenied(f"experiment must be one of {list(EXPERIMENTS)}")
        if experiment != "standard" and period != "dev":
            raise ToolDenied(f"the {experiment} experiment runs on the dev period only")
        from cleanroom.guard import SealedRunError

        from reserve import config as cfgmod
        from reserve import pipeline

        cfg = copy.deepcopy(cfgmod.load(self.configs_dir / f"{config}.yaml"))
        cfg["evaluation"]["period"] = period
        try:
            cfgmod.validate(cfg)
        except cfgmod.ConfigError as err:
            raise ToolDenied(f"config refused for period {period!r}: {err}") from err
        runner = self.runner or pipeline.run
        try:
            out = runner(cfg, experiment=None if experiment == "standard" else experiment)
        except SealedRunError as err:
            raise ToolDenied(f"test scoring refused: {err}") from err
        return self._view(json.loads((Path(out) / "results.json").read_text(encoding="utf-8")))

    def summary(self, run_id: str) -> dict[str, Any]:
        """A run's results.json after checking the id; shared with the Critic and Reporter."""
        if not RUN_ID.match(str(run_id)):
            raise ToolDenied(f"run_id {run_id!r} is not a valid run id")
        path = self.runs_dir / run_id / "results.json"
        if not path.exists():
            raise ToolDenied(f"no finished run {run_id!r}")
        s = json.loads(path.read_text(encoding="utf-8"))
        if s.get("period") == "test" and not s.get("sealed"):
            raise ToolDenied(f"run {run_id!r} is a test-period run without a lock record")
        return s

    def _view(self, s: dict[str, Any]) -> dict[str, Any]:
        out = {k: s.get(k) for k in VIEW_KEYS if k in s}
        out["git_commit"] = (s.get("provenance") or {}).get("git", {}).get("commit")
        return out

    def _load_results(self, run_id: str) -> dict[str, Any]:
        return self._view(self.summary(run_id))

    def _plot(self, run_id: str, kind: str) -> dict[str, Any]:
        if kind not in PLOTS:
            raise ToolDenied(f"kind must be one of {list(PLOTS)}")
        s = self.summary(run_id)
        banner = "SYNTHETIC: pipeline check, not a result. " if s.get("synthetic") else ""
        if kind == "loss_by_horizon":
            sc = self._scores(s)
            series = sc["loss_by_horizon"]
            return {
                "title": banner + "mean quantile loss by months ahead",
                "x": sorted({int(k) for v in series.values() for k in v}),
                "series": {n: [v[k] for k in sorted(v, key=int)] for n, v in series.items()},
            }
        if kind == "coverage":
            sc = self._scores(s)
            return {
                "title": banner + "share of outcomes inside the 50% and 90% intervals",
                "nominal": {"coverage_50": 0.5, "coverage_90": 0.9},
                "series": {
                    n: {k: c[k] for k in ("coverage_50", "coverage_90") if k in c}
                    for n, c in sc["contestants"].items()
                },
            }
        if kind == "learning_curve":
            th = s.get("thin_history")
            if not th:
                raise ToolDenied("learning_curve needs a thin_history run")
            return {
                "title": banner + "mean quantile loss by months of portfolio history",
                "x": th["levels"],
                "series": {
                    n: [v[k] for k in th["levels"]] for n, v in th["learning_curve"].items()
                },
            }
        scen = s.get("scenarios")
        if not scen:
            raise ToolDenied("scenarios plot needs a scenarios run")
        names = list(scen["summary"])
        contestants = sorted(next(iter(scen["summary"].values())))
        return {
            "title": banner + "HYPOTHETICAL scenarios: mean forecast 12-month increase (median)",
            "label": "HYPOTHETICAL",
            "x": names,
            "series": {
                c: [scen["summary"][n][c]["mean_increase_q0.5"] for n in names] for c in contestants
            },
        }

    def _scores(self, s: dict[str, Any]) -> dict[str, Any]:
        if not s.get("scores"):
            raise ToolDenied("this plot needs a standard run")
        return s["scores"]

    def _compute_stats(self, run_id: str, horizon: int) -> dict[str, Any]:
        s = self.summary(run_id)  # refuses unsealed test runs before any row is read
        if s.get("experiment"):
            raise ToolDenied("compute_stats works on standard runs only")
        ev = s["config"]["evaluation"]
        if not isinstance(horizon, int) or not 1 <= horizon <= int(ev["horizon"]):
            raise ToolDenied(f"horizon must be an integer from 1 to {ev['horizon']}")
        import polars as pl

        from reserve import config as cfgmod
        from reserve.evaluation import scoring

        preds = pl.read_parquet(self.runs_dir / run_id / "predictions.parquet")
        res = scoring.score(
            preds,
            list(ev["quantiles"]),
            horizon,
            int(ev["step_months"]),
            cfgmod.exclusions(s["config"]),
        )
        return {"run_id": run_id, "horizon": horizon, "synthetic": s.get("synthetic"), **res}
