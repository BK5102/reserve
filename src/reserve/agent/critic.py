"""The Critic as effective challenge: deterministic checks first, then an adversarial pass.

Deterministic checks on the agent's answer and the runs it relied on (methodology M38):

1. No typed numbers: the narrative holds no digits and no number words once known
   identifiers (run ids, config, contestant, model, and metric names) are removed.
2. A structured summary line is present, and every run in it exists.
3. Split order: every saved forecast targets a month after its origin, and every origin
   lies in the run's own period window.
4. Lock: any test-period run was scored under a lock.
5. Correction applied: every standard run reports its number of comparisons and a
   Holm-adjusted p-value for each tested one; thin-history runs report Holm across levels.
6. Coverage gap flagged: if any contestant's 90% Wilson interval excludes 90%, the
   narrative mentions coverage.
7. Contamination table present: every TSFM in the runs has a documented pretraining
   cutoff (or "undisclosed") in `MODEL_INFO`.
8. Labels: a synthetic run is called synthetic, a scenario run hypothetical.
9. Claims: each names a run and contestant that exist, the tested metric, and a direction
   ("better", "worse", "no clear difference") that matches the relative error and the
   Holm-adjusted Diebold-Mariano p-value.

Only if all of these pass does `challenge` ask a model for the three strongest objections
to each claim. Each objection must point to a value in a results file; code checks that
the pointer resolves and inserts the value, and the objection text is held to the same
no-typed-numbers rule.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from reserve.agent.llm import LLM
from reserve.agent.tools import RUN_ID, Toolbox, ToolDenied

_RUN_ID_ANYWHERE = re.compile(
    r"[0-9]{8}T[0-9]{6}Z_[a-z0-9_]+_(?:dev|test|recession)(?:_thin_history|_scenarios)?_[0-9a-f]{8}"
)
_DIGITS = re.compile(r"\d")
NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "hundreds", "thousand", "thousands", "million", "millions", "billion",
    "half", "halves", "halved", "third", "thirds", "quarter", "quarters",
    "double", "doubled", "triple", "tripled", "twice", "percent", "percentage",
)  # fmt: skip
_NUMBER_WORD = re.compile(r"\b(?:" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)
ALPHA = 0.05  # Holm-adjusted level for a "better" or "worse" claim
METRIC = "mean_quantile_loss"
METRIC_NAMES = (
    "mean_quantile_loss", "wql_on_change", "median_abs_error", "coverage_50", "coverage_90",
    "interval_score_90", "relative_error", "holm_p", "loss_by_horizon", "learning_curve",
)  # fmt: skip
OBJECTIONS_PER_CLAIM = 3


@dataclass
class CriticReport:
    passed: bool = True
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        self.passed = self.passed and bool(ok)


def split_tail(text: str) -> tuple[str, dict[str, Any] | None]:
    """(narrative, summary): the summary is the last non-empty line, if it is a JSON object."""
    offset = len(text)
    for line in reversed(text.splitlines(keepends=True)):
        offset -= len(line)
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return text[:offset].rstrip(), json.loads(stripped)
            except json.JSONDecodeError:
                return text, None
        return text, None
    return text, None


def known_identifiers(summaries: dict[str, dict[str, Any]], toolbox: Toolbox) -> set[str]:
    names = set(summaries) | set(toolbox.config_names()) | set(METRIC_NAMES)
    for s in summaries.values():
        names.add(str(s.get("config_name", "")))
        for c in s.get("contestants") or []:
            names.update(str(c.get(k, "")) for k in ("name", "model"))
    return {n for n in names if n}


def typed_numbers(prose: str, identifiers: set[str]) -> list[str]:
    stripped = _RUN_ID_ANYWHERE.sub(" ", prose)
    for name in sorted(identifiers, key=len, reverse=True):
        stripped = re.sub(
            r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])",
            " ",
            stripped,
            flags=re.IGNORECASE,
        )
    return _DIGITS.findall(stripped) + [m.lower() for m in _NUMBER_WORD.findall(stripped)]


def direction(comp: dict[str, Any]) -> str:
    """The direction a comparison supports, after Holm's correction (CLAUDE.md rule 8)."""
    p, rel = comp.get("holm_p"), comp.get("relative_error")
    if p is None or rel is None or p >= ALPHA:
        return "no clear difference"
    return "better" if rel < 1 else "worse"


def _split_order(toolbox: Toolbox, run_id: str, s: dict[str, Any]) -> tuple[bool, str]:
    import polars as pl

    from reserve import config as cfgmod

    path = toolbox.runs_dir / run_id / "predictions.parquet"
    if not path.exists():
        return False, "predictions.parquet missing"
    preds = pl.read_parquet(path, columns=["origin", "target_period"])
    first, last = cfgmod.origin_window(s["config"], s["period"])
    bad_order = preds.filter(pl.col("target_period") <= pl.col("origin")).height
    outside = preds.filter(~pl.col("origin").is_between(first, last)).height
    ok = bad_order == 0 and outside == 0
    return ok, f"{bad_order} targets not after their origin, {outside} origins outside the window"


def _correction(s: dict[str, Any]) -> tuple[bool, str]:
    if s.get("experiment") == "thin_history":
        th = s["thin_history"]
        ok = all("holm_p_all_levels" in r for r in th["holm_all_levels"])
        return ok, "Holm across every level"
    if s.get("experiment") == "scenarios":
        return True, "scenario runs are not scored, so nothing is compared"
    comp = s["scores"]["comparison"]
    tested = [c for c in comp["comparisons"] if c.get("dm") is not None]
    ok = "n_comparisons" in comp and all("holm_p" in c for c in tested)
    return ok, f"{comp.get('n_comparisons')} comparisons reported"


def _coverage_gaps(s: dict[str, Any]) -> list[str]:
    if not s.get("scores"):
        return []
    gaps = []
    for name, c in s["scores"]["contestants"].items():
        cov = c.get("coverage_90")
        if cov and not cov["ci_low"] <= 0.9 <= cov["ci_high"]:
            gaps.append(name)
    return gaps


def review(final_text: str, toolbox: Toolbox) -> CriticReport:
    from reserve.models.tsfm import MODEL_INFO

    report = CriticReport()
    prose, tail = split_tail(final_text)
    runs = (tail or {}).get("runs", [])
    summaries: dict[str, dict[str, Any]] = {}
    for run_id in runs:
        try:
            summaries[run_id] = toolbox.summary(run_id)
        except ToolDenied as err:
            report.add(f"run exists: {run_id}", False, str(err))
    typed = typed_numbers(prose, known_identifiers(summaries, toolbox))
    report.add(
        "no typed numbers", not typed, f"{len(typed)} found: {sorted(set(typed))}" if typed else ""
    )
    report.add("structured summary present", tail is not None)
    if tail is None:
        return report
    lower = prose.lower()
    for run_id, s in summaries.items():
        ok, detail = _split_order(toolbox, run_id, s)
        report.add(f"split order: {run_id}", ok, detail)
        if s.get("period") == "test":
            report.add(f"test run was locked: {run_id}", bool(s.get("sealed")))
        ok, detail = _correction(s)
        report.add(f"multiple-comparison correction applied: {run_id}", ok, detail)
        gaps = _coverage_gaps(s)
        if gaps:
            report.add(
                f"coverage gap flagged: {run_id}",
                "coverage" in lower,
                f"90% coverage misses its nominal level for {', '.join(gaps)}",
            )
        for c in s.get("contestants") or []:
            model = c.get("model")
            if model is not None:
                cutoff = (MODEL_INFO.get(model) or {}).get("pretraining_cutoff")
                report.add(
                    f"contamination entry: {model}", bool(cutoff), cutoff or "no MODEL_INFO entry"
                )
        if s.get("synthetic"):
            report.add(f"synthetic run labelled: {run_id}", "synthetic" in lower)
        if s.get("experiment") == "scenarios":
            report.add(f"scenario run labelled hypothetical: {run_id}", "hypothetical" in lower)

    for claim in tail.get("claims", []):
        run_id, name = claim.get("run_id"), claim.get("contestant")
        s = summaries.get(run_id)
        if s is None or not RUN_ID.match(str(run_id)):
            report.add(f"claim run known: {name}", False, f"run {run_id!r} not in 'runs'")
            continue
        if not s.get("scores"):
            report.add(f"claim on a scored run: {name}", False, f"{run_id} is not a standard run")
            continue
        if claim.get("metric") != METRIC:
            report.add(f"claim metric: {name}", False, f"only {METRIC!r} is compared")
            continue
        comp = s["scores"]["comparison"]
        entry = next((c for c in comp["comparisons"] if c["contestant"] == name), None)
        if entry is None:
            report.add(
                f"claim contestant known: {name}", False, f"reference is {comp['reference']}"
            )
            continue
        actual = direction(entry)
        report.add(
            f"claim direction: {name} in {run_id}",
            claim.get("direction") == actual,
            f"claimed {claim.get('direction')!r}; relative error and Holm p-value say {actual!r}",
        )
    return report


# ----- adversarial pass -------------------------------------------------------------------

CHALLENGE_SYSTEM = """You are an independent model validator acting as effective challenge \
for Reserve. You receive the claims an analyst made and the summary numbers of the runs \
they rely on. For each claim, give the three strongest objections you can find, each \
pointing to a value in a run's results that supports the objection (for example a coverage \
gap, too few origins, a large Holm-adjusted p-value, a synthetic or development-only run, \
an undisclosed pretraining cutoff). Never write a number yourself, in digits or words; the \
report inserts the value your pointer names. Reply with one JSON object on its own line: \
{"objections": [{"claim": <claim index from zero>, "objection": <text>, "evidence": \
{"run_id": <run id>, "path": <dot-separated keys into results.json>}}]}"""


@dataclass
class ChallengeReport:
    passed: bool = True
    ran: bool = False
    objections: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        self.passed = self.passed and bool(ok)


def resolve(summary: dict[str, Any], path: str) -> Any:
    """Follow dot-separated keys (and list indexes) into a results dict; KeyError if absent."""
    node: Any = summary
    for part in str(path).split("."):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise KeyError(path)
    return node


def challenge(final_text: str, toolbox: Toolbox, llm: LLM, critic: CriticReport) -> ChallengeReport:
    out = ChallengeReport()
    if not critic.passed:
        out.add("deterministic checks passed first", False, "adversarial pass not run")
        return out
    _, tail = split_tail(final_text)
    claims = (tail or {}).get("claims", [])
    if not claims:
        out.add("claims to challenge", True, "no headline claims, nothing to challenge")
        return out
    summaries = {r: toolbox.summary(r) for r in tail.get("runs", [])}
    views = {r: toolbox._view(s) for r, s in summaries.items()}
    content = json.dumps({"claims": claims, "runs": views}, default=str)
    response = llm.create(
        system=CHALLENGE_SYSTEM, messages=[{"role": "user", "content": content}], tools=[]
    )
    out.ran = True
    text = "\n".join(b["text"] for b in response["content"] if b.get("type") == "text")
    _, parsed = split_tail(text)
    if parsed is None or "objections" not in parsed:
        out.add("objections returned as JSON", False)
        return out
    identifiers = known_identifiers(summaries, toolbox)
    counts = dict.fromkeys(range(len(claims)), 0)
    for i, o in enumerate(parsed["objections"]):
        k = o.get("claim")
        if k not in counts:
            out.add(f"objection {i} names a claim", False, f"claim {k!r} does not exist")
            continue
        counts[k] += 1
        ev = o.get("evidence") or {}
        run_id, path = ev.get("run_id"), ev.get("path")
        try:
            value = resolve(summaries[run_id], path)
            ok, detail = True, f"{run_id}:{path}"
        except (KeyError, IndexError, ValueError, TypeError):
            value, ok, detail = None, False, f"{run_id}:{path} does not resolve"
        out.add(f"objection {i} evidence resolves", ok, detail)
        typed = typed_numbers(str(o.get("objection", "")), identifiers)
        out.add(f"objection {i} has no typed numbers", not typed, ", ".join(typed))
        out.objections.append({**o, "value": value})
    for k, n in counts.items():
        out.add(
            f"claim {k} has {OBJECTIONS_PER_CLAIM} objections",
            n == OBJECTIONS_PER_CLAIM,
            f"{n} given",
        )
    return out
