"""Command line for Reserve. The full real-data order is in docs/runbook.md.

reserve prepare synthetic --config configs/synthetic_small.yaml
reserve prepare freddie --config configs/freddie_sample.yaml
reserve doctor --config configs/freddie_test.yaml                    # is everything in place?
reserve prepare macro [--out data/raw/fred] [--latest] [--record cassette.json]
reserve prepare macro --cassette tests/fixtures/fred/cassette.json   # replay, no key, no network
reserve run configs/synthetic_small.yaml
reserve run configs/freddie_sample.yaml --period recession             # descriptive only
reserve run configs/synthetic_small.yaml --experiment thin_history   # dev only
reserve run configs/synthetic_small.yaml --experiment scenarios      # HYPOTHETICAL, dev only
reserve checksums --config configs/freddie_test.yaml                # -> data/checksums.sha256
reserve seal configs/freddie_test.yaml --note "plan for the 2013-2018 test"
reserve monitor artifacts/runs/<run id> --contestant B_age_curve --reference-end 2008-12
reserve agent "do TSFMs help when history is only 12 months?" [--cassette c.json] [--allow-test]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from reserve import config as cfgmod
from reserve import pipeline
from reserve.data import macro


def _prepare(args: argparse.Namespace) -> int:
    if args.what == "macro":
        if args.cassette:
            if args.record:
                print("--cassette replays and --record records; pass one", file=sys.stderr)
                return 2
            client = macro.CassetteFredClient(args.cassette, mode="replay")
            paths = macro.download_all(client, args.out, realtime=not args.latest)
            print(f"replayed from cassette {args.cassette} (no network, no key)")
            print("wrote " + ", ".join(str(p) for p in paths))
            return 0
        client = macro.HttpFredClient()
        if args.record:
            client = macro.CassetteFredClient(args.record, mode="record", backend=client)
        paths = macro.download_all(client, args.out, realtime=not args.latest)
        print("wrote " + ", ".join(str(p) for p in paths))
        if args.latest:
            print("latest mode: revisions are ignored; results will be flagged (methodology M15)")
        return 0
    if not args.config:
        print("--config is required for synthetic and freddie", file=sys.stderr)
        return 2
    cfg = cfgmod.load(args.config)
    if cfg["data"]["source"] != args.what:
        print(f"{args.config} is a {cfg['data']['source']} config", file=sys.stderr)
        return 2
    sums = pipeline.prepare(cfg)
    manifest = pipeline.read_manifest(cfg)
    if manifest["synthetic"]:
        print(
            f"SYNTHETIC DATA: {manifest['synthetic_records']} of {manifest['n_loans']} loans carry "
            "the simulator's markers; every result from these tables is labelled synthetic (M40)."
        )
    for name, digest in sums.items():
        print(f"{cfg['data']['interim_dir']}/{name}  sha256 {digest[:16]}")
    return 0


def _print_scores(scores: dict) -> None:
    comp = scores["comparison"]
    for name, s in scores["contestants"].items():
        print(
            f"  {name:22s} quantile loss {s['mean_quantile_loss']:.6f}  "
            f"90% coverage {s['coverage_90']['coverage']:.3f}"
        )
    print(f"  reference: {comp['reference']}; {comp['n_comparisons']} comparisons, Holm-adjusted")


def _run(args: argparse.Namespace) -> int:
    cfg = cfgmod.load(args.config)
    if args.period:
        # Unsealed periods only: the test is scored from its own sealed config (M31, M42).
        cfg = copy.deepcopy(cfg)
        cfg["evaluation"]["period"] = args.period
        cfgmod.validate(cfg)
    out = pipeline.run(cfg, experiment=args.experiment)
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    if result["synthetic"]:
        print("SYNTHETIC DATA: a pipeline check, not a result (methodology M17).")
    print(f"run {result['run_id']} ({result['role']}, {result['n_origins']} origins)")
    if result.get("experiment") == "thin_history":
        th = result["thin_history"]
        print(f"thin history: targets are vintages aged {th['target_vintage_age']} months")
        for name, levels in th["learning_curve"].items():
            cells = "  ".join(f"{k}: {v:.6f}" for k, v in levels.items())
            print(f"  {name:26s} quantile loss by history  {cells}")
        print(f"  {th['n_comparisons_all_levels']} comparisons, Holm-adjusted across all levels")
    elif result.get("experiment") == "scenarios":
        sc = result["scenarios"]
        print(sc["disclaimer"])
        for name, by_c in sc["summary"].items():
            print(f"  HYPOTHETICAL scenario {name}:")
            for c, s in by_c.items():
                print(f"    {c:26s} mean 12-month increase (median) {s['mean_increase_q0.5']:.6f}")
        moved = [c for c, r in sc["responds_to_scenario"].items() if r]
        print(f"  contestants that respond to the scenario: {', '.join(moved) or 'none'}")
    else:
        _print_scores(result["scores"])
    print(f"results: {out / 'results.json'}")
    return 0


def _seal(args: argparse.Namespace) -> int:
    cfg = cfgmod.load(args.config)
    lk = pipeline.seal(cfg, note=args.note, rehearsal=args.rehearsal)
    if args.rehearsal:
        print(pipeline.REHEARSAL_NOTE.strip())
    print(f"sealed {lk.path} (config hash {lk.config_hash[:12]})")
    print(f"next: git add {lk.path.as_posix()} data/checksums.sha256 && git commit, then run once")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    from reserve import doctor

    lines, ready = doctor.check(cfgmod.load(args.config))
    print("\n".join(lines))
    print("ready to prepare the Freddie Mac files" if ready else "NOT READY: fix the FIX lines")
    return 0 if ready else 3


def _checksums(args: argparse.Namespace) -> int:
    from reserve import checksums

    cfg = cfgmod.load(args.config)
    path, sums = checksums.write(cfg)
    print(f"wrote {path}: {len(sums)} files (methodology M42); commit it with the lock")
    return 0


def _monitor(args: argparse.Namespace) -> int:
    import polars as pl

    from reserve import months
    from reserve.monitoring import drift

    run_dir = Path(args.run_dir)
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    preds = pl.read_parquet(run_dir / "predictions.parquet")
    ev = result["config"]["evaluation"]
    table = drift.monitor_predictions(
        preds,
        args.contestant,
        months.parse(args.reference_end),
        list(ev["quantiles"]),
        int(ev["horizon"]),
        window=args.window,
    )
    if result.get("synthetic"):
        print("SYNTHETIC DATA: a pipeline check, not a result (methodology M17).")
    print(f"drift monitor for {args.contestant} in {result['run_id']} (methodology M36)")
    for r in table.iter_rows(named=True):
        print(
            f"  origin {r['origin']} known {r['known_at']}  {r['status']:5s}  "
            f"coverage {r['coverage_90']:.3f} ({r['coverage_status']})  "
            f"loss ratio {r['loss_ratio']:.2f} ({r['loss_status']})"
        )
    return 0


def _agent(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from reserve.agent import critic, llm, reporter, runner, tools

    if args.cassette:
        model = llm.CassetteLLM(args.cassette)
    else:
        try:
            backend = llm.AnthropicLLM()  # the SDK reads ANTHROPIC_API_KEY
        except ImportError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        model = llm.CassetteLLM(args.record, "record", backend) if args.record else backend
    box = tools.Toolbox(allow_test=args.allow_test)
    result = runner.run_agent(args.question, box, model, max_tool_calls=args.max_tool_calls)
    review = critic.review(result.final_text, box)
    challenge = critic.challenge(result.final_text, box, model, review)
    pack = reporter.build_pack(result, review, challenge, box)
    out = Path(args.out or f"artifacts/agent/{datetime.now(UTC):%Y%m%dT%H%M%SZ}_pack.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pack, encoding="utf-8")
    print(
        f"critic: {'passed' if review.passed else 'FAILED'}; challenge: "
        f"{'passed' if challenge.passed else 'FAILED'}"
    )
    print(f"evidence pack (draft for human review): {out}")
    return 0 if review.passed and challenge.passed else 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reserve", description="Reserve experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="build interim tables from raw data")
    p.add_argument("what", choices=["synthetic", "freddie", "macro"])
    p.add_argument("--config")
    p.add_argument("--out", default="data/raw/fred", help="macro only: output folder")
    p.add_argument("--latest", action="store_true", help="macro only: no ALFRED vintages")
    p.add_argument("--record", help="macro only: also save a cassette of the API responses")
    p.add_argument("--cassette", help="macro only: replay a saved cassette instead of the API")
    p.set_defaults(func=_prepare)
    r = sub.add_parser("run", help="score a config's evaluation period")
    r.add_argument("config")
    r.add_argument(
        "--period",
        choices=["dev", "recession"],
        help="score this unsealed period instead of the config's own (the test needs its "
        "sealed config)",
    )
    r.add_argument(
        "--experiment",
        choices=["thin_history", "scenarios"],
        help="a harness option on the dev period only (scenarios are HYPOTHETICAL)",
    )
    r.set_defaults(func=_run)
    s = sub.add_parser("seal", help="seal a config and its data before test scoring")
    s.add_argument("config")
    s.add_argument("--note", default="")
    s.add_argument(
        "--rehearsal",
        action="store_true",
        help="scripts/rehearse_real_path.py only: seal SYNTHETIC rehearsal data (refused for "
        "real data)",
    )
    s.set_defaults(func=_seal)
    d = sub.add_parser("doctor", help="check downloads, keys, extras, and git before a step")
    d.add_argument("--config", default="configs/freddie_test.yaml")
    d.set_defaults(func=_doctor)
    k = sub.add_parser("checksums", help="record SHA-256 of the raw downloads before sealing")
    k.add_argument("--config", required=True)
    k.set_defaults(func=_checksums)
    m = sub.add_parser("monitor", help="drift status of a finished run's forecasts")
    m.add_argument("run_dir")
    m.add_argument("--contestant", required=True)
    m.add_argument("--reference-end", required=True, help="last origin of the reference window")
    m.add_argument("--window", type=int, default=4, help="origins per monitoring window")
    m.set_defaults(func=_monitor)
    a = sub.add_parser("agent", help="run the experiment agent and draft an evidence pack")
    a.add_argument("question")
    a.add_argument("--allow-test", action="store_true", help="offer the sealed test period")
    a.add_argument("--cassette", help="replay a recorded model conversation (no API call)")
    a.add_argument("--record", help="record the live conversation to this cassette file")
    a.add_argument("--out", help="where to write the evidence pack")
    a.add_argument("--max-tool-calls", type=int, default=10)
    a.set_defaults(func=_agent)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (
        FileNotFoundError,
        ValueError,
        PermissionError,
        ImportError,
        macro.MissingKeyError,
    ) as err:
        # PermissionError covers the sealed-run guard (SealedRunError, LockMissingError).
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
