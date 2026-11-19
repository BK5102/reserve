"""`reserve summary RUN_DIR`: the headline table of one run, as Markdown, from its results file.

Every number is read from `results.json` and printed with the run id and config hash next
to it, so `docs/findings.md` can quote it without retyping (CLAUDE.md rules 4 and 12;
methodology M46). Nothing is recomputed here.
"""

from __future__ import annotations

from typing import Any

# Reserve's 90% interval for these models is their own 80% interval (methodology M32, M44).
FLAT_TAIL_MODELS = ("D_tsfm_chronos_bolt_tiny", "D_tsfm_timesfm_2p5")


def _f(x: Any, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}g}" if isinstance(x, float) else str(x)


def _cov(c: dict | None) -> str:
    if not c:
        return "n/a"
    return f"{c['coverage']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def headline_markdown(result: dict[str, Any]) -> str:
    if "scores" not in result:
        raise ValueError("summary covers standard runs; experiments are read with load_results")
    sc = result["scores"]
    comp = sc["comparison"]
    by_name = {c["contestant"]: c for c in comp["comparisons"]}
    prov = result.get("provenance", {}).get("git", {})
    lines = []
    if result.get("synthetic"):
        lines += ["**SYNTHETIC DATA: a pipeline check, not a result.**", ""]
    lines += [
        f"Run `{result['run_id']}`, config hash `{result['config_hash'][:12]}`, "
        f"role: {result['role']}, {result['n_origins']} origins "
        f"({result['origins'][0]} to {result['origins'][-1]}), git `{str(prov.get('commit'))[:8]}`"
        f"{' (dirty tree)' if prov.get('dirty') else ''}.",
    ]
    if result.get("sealed"):
        lines.append(f"Sealed run number {result['sealed']['run_number']} under its lock.")
    lines += [
        "",
        f"Horizon {sc['headline_horizon']} months. Reference (best classical by the M29 rule): "
        f"`{comp['reference']}`. {comp['n_comparisons']} comparisons, Holm-adjusted.",
        "",
        "| Contestant | Mean quantile loss | Relative error vs reference | DM p | Holm p "
        "| 50% coverage [95% CI] | 90% coverage [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in sc["contestants"].items():
        c = by_name.get(name, {})
        ref = name == comp["reference"]
        dm = (c.get("dm") or {}).get("p_value")
        lines.append(
            f"| {name}{' (reference)' if ref else ''} | {_f(s['mean_quantile_loss'])} | "
            f"{'1' if ref else _f(c.get('relative_error'))} | {'' if ref else _f(dm)} | "
            f"{'' if ref else _f(c.get('holm_p'))} | {_cov(s.get('coverage_50'))} | "
            f"{_cov(s.get('coverage_90'))} |"
        )
    flat = [n for n in sc["contestants"] if n in FLAT_TAIL_MODELS]
    if flat:
        lines += [
            "",
            f"{', '.join(flat)}: trained only on levels 0.1 to 0.9, so the 90% interval is the "
            "model's own 80% interval and its 90% coverage is low by construction (M32).",
        ]
    lines += [
        "",
        "Coverage intervals treat forecasts as independent, which they are not (L20).",
    ]
    return "\n".join(lines)
