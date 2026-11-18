"""Command line: `reserve prepare`, `reserve run`, `reserve seal`.

reserve prepare synthetic --config configs/synthetic_small.yaml
reserve prepare freddie --config configs/freddie_sample.yaml
reserve prepare macro [--out data/raw/fred] [--latest] [--record cassette.json]
reserve run configs/synthetic_small.yaml
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


def _run(args: argparse.Namespace) -> int:
    cfg = cfgmod.load(args.config)
    out = pipeline.run(cfg)
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    if result["synthetic"]:
        print("SYNTHETIC DATA: a pipeline check, not a result (methodology M17).")
    comp = result["scores"]["comparison"]
    print(f"run {result['run_id']} ({result['role']}, {result['n_origins']} origins)")
    for name, s in result["scores"]["contestants"].items():
        print(
            f"  {name:22s} quantile loss {s['mean_quantile_loss']:.6f}  "
            f"90% coverage {s['coverage_90']['coverage']:.3f}"
        )
    print(f"  reference: {comp['reference']}; {comp['n_comparisons']} comparisons, Holm-adjusted")
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
