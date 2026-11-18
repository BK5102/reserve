"""Command line: `reserve prepare`, `reserve run`, `reserve seal`.

reserve prepare synthetic --config configs/synthetic_small.yaml
reserve prepare freddie --config configs/freddie_sample.yaml
reserve prepare macro [--out data/raw/fred] [--latest] [--record cassette.json]
reserve run configs/synthetic_small.yaml
reserve run configs/synthetic_small.yaml --experiment thin_history   # dev only
reserve run configs/synthetic_small.yaml --experiment scenarios      # HYPOTHETICAL, dev only
reserve seal configs/freddie_sample.yaml --note "plan for the 2013-2018 test"
"""

from __future__ import annotations

import argparse
import json
import sys

from reserve import config as cfgmod
from reserve import pipeline
from reserve.data import macro


def _prepare(args: argparse.Namespace) -> int:
    if args.what == "macro":
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
    lk = pipeline.seal(cfg, note=args.note)
    print(f"sealed {lk.path} (config hash {lk.config_hash[:12]}); commit it before a test run")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reserve", description="Reserve experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="build interim tables from raw data")
    p.add_argument("what", choices=["synthetic", "freddie", "macro"])
    p.add_argument("--config")
    p.add_argument("--out", default="data/raw/fred", help="macro only: output folder")
    p.add_argument("--latest", action="store_true", help="macro only: no ALFRED vintages")
    p.add_argument("--record", help="macro only: also save a cassette of the API responses")
    p.set_defaults(func=_prepare)
    r = sub.add_parser("run", help="score a config's evaluation period")
    r.add_argument("config")
    r.add_argument(
        "--experiment",
        choices=["thin_history", "scenarios"],
        help="a harness option on the dev period only (scenarios are HYPOTHETICAL)",
    )
    r.set_defaults(func=_run)
    s = sub.add_parser("seal", help="seal a config and its data before test scoring")
    s.add_argument("config")
    s.add_argument("--note", default="")
    s.set_defaults(func=_seal)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, cfgmod.ConfigError, macro.MissingKeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
