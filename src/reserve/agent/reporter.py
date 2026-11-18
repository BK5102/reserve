"""The Reporter: an evidence pack from an agent run, for a person to review (M39).

Sections follow PLAN.md FR15: purpose and scope, data, method, backtest, sensitivity,
limitations, monitoring plan, open issues. Every number is formatted by this module from
a results file; the model's narrative is included as written, after the Critic has checked
it holds no typed numbers. Every pack says it is inspired by validation practice and is not
a compliance document (CLAUDE.md rule 7), flags synthetic runs, and labels scenario
results HYPOTHETICAL.
"""

from __future__ import annotations

from typing import Any

from reserve.agent.critic import ChallengeReport, CriticReport, split_tail
from reserve.agent.runner import AgentResult
from reserve.agent.tools import Toolbox, ToolDenied
from reserve.monitoring import drift

NOT_COMPLIANCE = (
    "Inspired by validation practice, not a compliance document. A draft for human review."
)


def _p(x: Any) -> str:
    if x is None:
        return "n/a"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _sha(v: Any) -> str:
    return str(v.get("sha256", "n/a") if isinstance(v, dict) else v)[:12]


def _f(x: Any, digits: int = 6) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _scores_table(scores: dict[str, Any]) -> list[str]:
    comp = scores["comparison"]
    by_name = {c["contestant"]: c for c in comp["comparisons"]}
    lines = [
        f"Headline horizon: {scores['headline_horizon']} months; reference (best classical): "
        f"{comp['reference']}; {comp['n_comparisons']} comparisons, Holm-adjusted.",
        "",
        "| Contestant | Mean quantile loss | Relative error | Holm p | 50% coverage | "
        "90% coverage [Wilson] | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in scores["contestants"].items():
        c = by_name.get(name, {})
        c90 = s.get("coverage_90") or {}
        c50 = s.get("coverage_50") or {}
        lines.append(
            f"| {name} | {_f(s['mean_quantile_loss'])} | {_f(c.get('relative_error'), 3)} | "
            f"{_p(c.get('holm_p'))} | {_f(c50.get('coverage'), 3)} | "
            f"{_f(c90.get('coverage'), 3)} [{_f(c90.get('ci_low'), 3)}, "
            f"{_f(c90.get('ci_high'), 3)}] | {s['n']} |"
        )
    return lines


def _thin_table(th: dict[str, Any]) -> list[str]:
    lv = th["levels"]
    lo, hi = th["target_vintage_age"]
    ages = f"{lo} months" if lo == hi else f"{lo} to {hi} months"
    lines = [
        f"Thin history (targets: vintages aged {ages}; {th['n_comparisons_all_levels']} "
        "comparisons, Holm-adjusted across all levels).",
        "",
        "| Contestant | " + " | ".join(f"{x} months" if x != "full" else "full" for x in lv) + " |",
        "|---" * (len(lv) + 1) + "|",
    ]
    for name, v in th["learning_curve"].items():
        lines.append(f"| {name} | " + " | ".join(_f(v[x]) for x in lv) + " |")
    return lines


def _scenario_table(sc: dict[str, Any]) -> list[str]:
    names = list(sc["summary"])
    lines = [
        f"**{sc['disclaimer']}** Not scored: {sc['not_scored']}.",
        "",
        "| Contestant | "
        + " | ".join(f"HYPOTHETICAL {n}: mean 12-month increase (median)" for n in names)
        + " | Responds |",
        "|---" * (len(names) + 2) + "|",
    ]
    for c in sorted(next(iter(sc["summary"].values()))):
        cells = " | ".join(_f(sc["summary"][n][c]["mean_increase_q0.5"]) for n in names)
        lines.append(f"| {c} | {cells} | {'yes' if sc['responds_to_scenario'][c] else 'no'} |")
    return lines


def build_pack(
    result: AgentResult, critic: CriticReport, challenge: ChallengeReport, toolbox: Toolbox
) -> str:
    prose, tail = split_tail(result.final_text)
    tail = tail or {}
    summaries: dict[str, dict[str, Any]] = {}
    missing = []
    for run_id in tail.get("runs", []):
        try:
            summaries[run_id] = toolbox.summary(run_id)
        except ToolDenied:
            missing.append(run_id)
    synthetic = [r for r, s in summaries.items() if s.get("synthetic")]
    lines = ["# Evidence pack (draft for human review)", "", f"**{NOT_COMPLIANCE}**", ""]
    if synthetic:
        lines += [
            "**SYNTHETIC DATA: every run below marked synthetic is a pipeline check on a "
            "simulator, not a result about real loans.**",
            "",
        ]
    status = "all checks passed" if critic.passed else "CHECKS FAILED, do not use"
    lines += [f"**Question:** {result.question}", "", f"**Critic:** {status}.", ""]

    lines += ["## 1. Purpose and scope", ""]
    lines += [
        "Does a zero-shot time series foundation model forecast the next 12 months of vintage "
        "delinquency curves better than the best classical baseline, with honest intervals? "
        "Portfolio-level curves of prime, conforming US mortgages only (methodology M1, M2).",
        "",
        "Narrative written by the model (numbers are not allowed here):",
        "",
        prose or "_The model returned no narrative._",
        "",
    ]

    lines += ["## 2. Data", ""]
    for run_id, s in summaries.items():
        prov = s.get("provenance") or {}
        data = prov.get("data") or {}
        lines.append(
            f"- `{run_id}`: config `{s['config_name']}` (hash `{s['config_hash'][:12]}`), "
            f"{'SYNTHETIC' if s.get('synthetic') else 'real'} data, macro mode "
            f"{s.get('macro_mode')}, revisions ignored: {s.get('macro_revisions_ignored')}, "
            f"git `{(prov.get('git') or {}).get('commit', 'n/a')[:8]}`, inputs "
            + ", ".join(f"{k} sha256 `{_sha(v)}`" for k, v in sorted(data.items()))
        )
    lines += [""]

    lines += ["## 3. Method", ""]
    for run_id, s in summaries.items():
        names = ", ".join(c["name"] for c in s.get("contestants") or [])
        lines.append(
            f"- `{run_id}`: {s['role']} period, {s['n_origins']} origins, "
            f"experiment {s.get('experiment') or 'standard'}, contestants {names}."
        )
    lines += [
        "",
        "Origins are Cleanroom rolling splits; every forecast uses only data known at its "
        "origin; scores are quantile losses with a Diebold-Mariano test and Holm's correction "
        "against the best classical contestant (methodology M22 to M29).",
        "",
    ]

    lines += ["## 4. Backtest", ""]
    for run_id, s in summaries.items():
        if s.get("scores"):
            lines += [f"### `{run_id}`", "", *_scores_table(s["scores"]), ""]
    lines += ["## 5. Sensitivity", ""]
    for run_id, s in summaries.items():
        if s.get("thin_history"):
            lines += [f"### `{run_id}`", "", *_thin_table(s["thin_history"]), ""]
        if s.get("scenarios"):
            lines += [f"### `{run_id}` (HYPOTHETICAL)", "", *_scenario_table(s["scenarios"]), ""]

    lines += ["## 6. Limitations", ""]
    from reserve.models.tsfm import MODEL_INFO

    seen = set()
    for s in summaries.values():
        for c in s.get("contestants") or []:
            if c.get("model") and c["name"] not in seen:
                seen.add(c["name"])
                info = MODEL_INFO.get(c["model"], {})
                lines.append(
                    f"- {c['name']}: pretraining cutoff {info.get('pretraining_cutoff', 'unknown')}"
                    f" ({info.get('status', 'unknown')}); input is date-free, memorized shapes "
                    "are not controlled."
                )
    lines += [
        "- The full register is in `docs/methodology.md` (Limitations register).",
        "- The 2007 to 2009 recession is one episode and is descriptive only.",
        "",
    ]

    lines += ["## 7. Monitoring plan", ""]
    lines += [
        f"`reserve monitor` (methodology M36): 90% coverage red when its Wilson interval is "
        f"below {drift.COVERAGE_NOMINAL}, amber when more than {drift.COVERAGE_AMBER_MARGIN} "
        f"below it; loss ratio amber above {drift.LOSS_AMBER}, red above {drift.LOSS_RED}; "
        "macro features amber outside the central 90% of training values, red outside their "
        "range. Thresholds are not tuned with a risk team.",
        "",
    ]

    lines += ["## 8. Open issues", "", "### Critic checks", "", "| Check | Result | Detail |"]
    lines += ["|---|---|---|"]
    for c in critic.checks:
        lines.append(f"| {c['check']} | {'pass' if c['ok'] else 'FAIL'} | {c['detail']} |")
    lines += ["", "### Objections (adversarial pass)", ""]
    if not challenge.ran:
        lines += [f"_Not run: {'; '.join(c['detail'] for c in challenge.checks)}._", ""]
    claims = tail.get("claims", [])
    for o in challenge.objections:
        claim = claims[o["claim"]] if 0 <= o["claim"] < len(claims) else {}
        ev = o.get("evidence") or {}
        lines.append(
            f"- Claim {o['claim']} ({claim.get('contestant')}, {claim.get('direction')}): "
            f"{o.get('objection')} Evidence `{ev.get('run_id')}:{ev.get('path')}` = "
            f"{_f(o.get('value'), 4)}."
        )
    for c in challenge.checks:
        if not c["ok"]:
            lines.append(f"- Challenge check FAILED: {c['check']} ({c['detail']})")
    for r in missing:
        lines.append(f"- Run {r} could not be loaded.")
    lines += ["", "### Tool calls", ""]
    for call in result.tool_calls:
        lines.append(f"- `{call['tool']}` {call['input']}")
    for d in result.denied:
        lines.append(f"- denied `{d['tool']}`: {d['reason']}")
    for e in result.errors:
        lines.append(f"- error in `{e['tool']}`: {e['error']}")
    if result.stopped_by_budget:
        lines.append("- stopped by the tool-call budget")
    lines += ["", f"_{NOT_COMPLIANCE}_"]
    return "\n".join(lines) + "\n"
