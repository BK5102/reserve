"""Can the data answer Reserve's question? Run before building on any dataset (methodology M19).

Usage:
    uv run python scripts/check_data_fit.py freddie data/raw/freddie [--fred data/raw/fred] [--out artifacts/data_fit.json]
    uv run python scripts/check_data_fit.py synthetic configs/synthetic_fit.yaml [--out ...]

The synthetic mode only proves the checks work. Its numbers are not real-data results.
Files read from a "freddie" folder are also labelled SYNTHETIC if their records carry the
simulator's markers (methodology M40).

Exit code: 0 if the data-fit gate passes, 3 if it fails (the go/no-go step in
docs/runbook.md), 1 on an error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from reserve import datafit
from reserve.data import loans, macro, synthetic


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("source", choices=["freddie", "synthetic"])
    ap.add_argument("path", help="raw Freddie Mac folder, or a synthetic YAML config")
    ap.add_argument("--fred", help="folder of saved FRED tables (from `reserve prepare macro`)")
    ap.add_argument("--out", help="write the full report as JSON here")
    args = ap.parse_args(argv)

    observations = None
    if args.source == "freddie":
        orig, perf, files = datafit.load_freddie(args.path)
        label = f"freddie:{args.path}"
        if args.fred:
            observations = macro.load_observations(args.fred)
    else:
        cfg = yaml.safe_load(Path(args.path).read_text(encoding="utf-8"))
        sim = synthetic.simulate(synthetic.SimConfig(**cfg["synthetic"]))
        orig = sim.orig
        perf = sim.perf.with_columns(loans.parse_delinquency(synthetic.pl.col("dlq_status")))
        files = []
        label = f"synthetic:{args.path}"
        observations = sim.macro
    report = datafit.check(orig, perf, label, file_report=files, macro_observations=observations)
    print(datafit.to_markdown(report))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nfull report: {args.out}")
    if not report["gate"]["passes"]:
        print("\nGATE FAILED: stop before any model run (docs/runbook.md, step 3).")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
