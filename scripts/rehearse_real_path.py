"""Dress rehearsal of the real-data path, on SYNTHETIC files (methodology M41).

Writes simulator files in the exact Freddie Mac sample layout into a throwaway workspace
(`data/raw/freddie/sample_orig_YYYY.txt` and `sample_perf_YYYY.txt`, plus FRED-style
tables in `data/raw/fred/`), copies the shipped Freddie Mac configs into it, makes it a
throwaway git repo, and then runs the owner's commands from docs/runbook.md there, as
separate processes, exactly as typed:

    reserve doctor --config configs/freddie_test.yaml
    reserve prepare freddie --config configs/freddie_sample.yaml
    python scripts/check_data_fit.py freddie data/raw/freddie --fred data/raw/fred --out ...
    reserve run configs/freddie_sample.yaml                       (dev)
    reserve run configs/freddie_sample.yaml --period recession    (descriptive)
    reserve checksums --config configs/freddie_test.yaml
    reserve seal configs/freddie_test.yaml                        (must refuse: synthetic)
    reserve seal configs/freddie_test.yaml --rehearsal            (rehearsal lock only)
    git add + commit the lock and checksums
    reserve run configs/freddie_test.yaml                         (the sealed run)
    reserve prepare macro --cassette tests/fixtures/fred/cassette.json --out ...

and checks every output: exit codes, the SYNTHETIC label on every result, the sealed-run
record, and that a changed plan is refused. Nothing is written inside the repo.

By default the TSFM list in the copied configs is replaced by the `fake` stand-in, so the
rehearsal runs without the `tsfm` extra (as in CI). `--with-tsfm` keeps the shipped list.

Usage:
    uv run python scripts/rehearse_real_path.py [--workdir DIR] [--loans 120] [--with-tsfm]

Exit code 0 means every step behaved as the runbook says; anything else names the step.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("freddie_sample.yaml", "freddie_test.yaml")
GIT = ["git", "-c", "user.name=rehearsal", "-c", "user.email=rehearsal@example.invalid"]
BANNER = "SYNTHETIC REHEARSAL: every number below comes from simulator files, not Freddie Mac."


class StepFailed(RuntimeError):
    pass


def _run(cmd: list[str], cwd: Path, expect: tuple[int, ...] = (0,)) -> str:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    shown = " ".join(c if " " not in c else f'"{c}"' for c in cmd[1:] if c != "-m")
    print(f"  [{time.perf_counter() - t0:5.1f}s] exit {proc.returncode}: {shown}")
    if proc.returncode not in expect:
        raise StepFailed(
            f"{shown} exited {proc.returncode}, expected {expect}\n{proc.stdout}\n{proc.stderr}"
        )
    return proc.stdout + proc.stderr


def reserve(*args: str) -> list[str]:
    return [sys.executable, "-m", "reserve.cli", *args]


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise StepFailed(message)


def _latest_result(ws: Path, name: str) -> dict:
    runs = sorted((ws / "artifacts" / "runs").glob(f"*_{name}_*"))
    _check(bool(runs), f"no run folder for {name}")
    return json.loads((runs[-1] / "results.json").read_text(encoding="utf-8"))


def build_workspace(ws: Path, loans: int, with_tsfm: bool) -> None:
    from reserve.data import synthetic

    sim = synthetic.simulate(
        synthetic.SimConfig(
            first_vintage="1999Q1",
            n_vintages=84,  # 1999Q1 to 2019Q4: covers the recession, dev, and test windows
            loans_per_vintage=loans,
            data_end="2020-12",
            seed=7,
        )
    )
    synthetic.write(sim, ws / "data" / "raw")  # -> data/raw/freddie and data/raw/fred
    (ws / "data" / "raw" / "truth.parquet").unlink()  # the owner's folder has no truth file
    (ws / "configs").mkdir()
    for name in CONFIGS:
        cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        if not with_tsfm:
            cfg["contestants"]["tsfm"] = ["fake"]
        (ws / "configs" / name).write_text(yaml.safe_dump(cfg, sort_keys=False), "utf-8")
    shutil.copy(ROOT / ".gitignore", ws / ".gitignore")
    (ws / "data" / "README.md").write_text("Rehearsal workspace. SYNTHETIC data only.\n", "utf-8")
    _run([*GIT, "init", "-q"], ws)
    _run([*GIT, "add", "-A"], ws)
    _run([*GIT, "commit", "-qm", "rehearsal workspace"], ws)


def rehearse(ws: Path, loans: int, with_tsfm: bool) -> None:
    print(BANNER)
    print(f"workspace: {ws}")
    print("1. synthetic files in the Freddie Mac sample layout")
    build_workspace(ws, loans, with_tsfm)
    files = sorted(p.name for p in (ws / "data" / "raw" / "freddie").iterdir())
    _check(files[0] == "sample_orig_1999.txt" and len(files) == 2 * 21, f"files: {files[:4]}")

    print("2. setup check")
    out = _run(reserve("doctor", "--config", "configs/freddie_test.yaml"), ws, expect=(0, 3))
    _check("21 sample years" in out, f"doctor did not find the files:\n{out}")

    print("3. prepare, data fit, dev and recession runs")
    out = _run(reserve("prepare", "freddie", "--config", "configs/freddie_sample.yaml"), ws)
    _check("SYNTHETIC DATA" in out, "prepare did not label the data SYNTHETIC")
    fit = [sys.executable, str(ROOT / "scripts" / "check_data_fit.py"), "freddie"]
    fit += ["data/raw/freddie", "--fred", "data/raw/fred", "--out", "artifacts/data_fit.json"]
    out = _run(fit, ws, expect=(0, 3))  # a small synthetic portfolio fails the events rule
    _check(out.startswith("SYNTHETIC DATA"), "data fit did not label the data SYNTHETIC")
    out = _run(reserve("run", "configs/freddie_sample.yaml"), ws)
    _check("SYNTHETIC DATA" in out and "reference:" in out, f"dev run output:\n{out}")
    dev = _latest_result(ws, "freddie_sample_dev")
    _check(dev["synthetic"] is True and dev["role"] == "development", "dev result labels")
    _check(dev["n_origins"] == 9, f"dev origins {dev['n_origins']}, expected 9")
    out = _run(reserve("run", "configs/freddie_sample.yaml", "--period", "recession"), ws)
    rec = _latest_result(ws, "freddie_sample_recession")
    _check(rec["synthetic"] is True and rec["role"] == "descriptive", "recession labels")
    _run(reserve("run", "configs/freddie_sample.yaml", "--period", "test"), ws, expect=(2,))

    print("4. checksums, seal, commit, sealed run")
    out = _run(reserve("seal", "configs/freddie_test.yaml"), ws, expect=(1,))
    _check("checksums" in out and "does not exist" in out, f"seal without checksums:\n{out}")
    _run(reserve("checksums", "--config", "configs/freddie_test.yaml"), ws)
    out = _run(reserve("seal", "configs/freddie_test.yaml"), ws, expect=(1,))
    _check("synthetic markers" in out, f"seal did not refuse synthetic data:\n{out}")
    out = _run(reserve("run", "configs/freddie_test.yaml"), ws, expect=(1,))
    _check("no lock" in out and "Traceback" not in out, f"run without a lock:\n{out}")
    out = _run(reserve("seal", "configs/freddie_test.yaml", "--rehearsal"), ws)
    _check("REHEARSAL" in out, "rehearsal seal did not say so")
    out = _run(reserve("run", "configs/freddie_test.yaml"), ws, expect=(1,))
    _check("not committed" in out, f"run with an uncommitted lock:\n{out}")
    _run([*GIT, "add", "locks", "data/checksums.sha256"], ws)
    _run([*GIT, "commit", "-qm", "seal the test plan (rehearsal)"], ws)
    out = _run(reserve("run", "configs/freddie_test.yaml"), ws)
    _check("SYNTHETIC DATA" in out, "sealed run output not labelled SYNTHETIC")
    test = _latest_result(ws, "freddie_test_test")
    _check(test["synthetic"] is True and test["role"] == "sealed test", "sealed result labels")
    _check(test["sealed"]["run_number"] == 1, f"sealed record {test['sealed']}")
    _check(test["n_origins"] == 24, f"test origins {test['n_origins']}, expected 24")
    uses = next((ws / "locks").glob("*.uses.jsonl")).read_text("utf-8").splitlines()
    _check([json.loads(u)["status"] for u in uses] == ["started", "finished"], "uses record")
    cfg_path = ws / "configs" / "freddie_test.yaml"
    plan = cfg_path.read_text("utf-8")
    cfg_path.write_text(plan.replace("horizon: 12", "horizon: 6"), "utf-8")
    _run([*GIT, "commit", "-qam", "change the plan after sealing"], ws)
    out = _run(reserve("run", "configs/freddie_test.yaml"), ws, expect=(1,))
    _check("no lock for config hash" in out, f"run of a changed plan:\n{out}")

    print("5. macro download path, replayed from the recorded cassette")
    cassette = ROOT / "tests" / "fixtures" / "fred" / "cassette.json"
    out = _run(reserve("prepare", "macro", "--cassette", str(cassette), "--out", "fred_c"), ws)
    _check(all((ws / "fred_c" / f"{s}.parquet").exists() for s in ("UNRATE", "MORTGAGE30US",
           "HPIPONM226S")), "macro tables missing")  # fmt: skip
    _check("cassette" in out, "prepare macro did not say it replayed a cassette")
    print("REHEARSAL PASSED: every step behaved as docs/runbook.md says (SYNTHETIC data).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workdir", help="empty folder to use (default: a new temporary folder)")
    ap.add_argument("--loans", type=int, default=120, help="synthetic loans per vintage")
    ap.add_argument("--with-tsfm", action="store_true", help="keep the configs' real TSFMs")
    ap.add_argument("--keep", action="store_true", help="keep the temporary workspace")
    args = ap.parse_args(argv)
    if args.workdir:
        ws = Path(args.workdir).resolve()
        if ws.exists() and any(ws.iterdir()):
            print(f"{ws} is not empty", file=sys.stderr)
            return 2
        ws.mkdir(parents=True, exist_ok=True)
    else:
        ws = Path(tempfile.mkdtemp(prefix="reserve_rehearsal_"))
    t0 = time.perf_counter()
    try:
        rehearse(ws, args.loans, args.with_tsfm)
    except StepFailed as err:
        print(f"REHEARSAL FAILED: {err}", file=sys.stderr)
        return 1
    finally:
        if not args.workdir and not args.keep:
            shutil.rmtree(ws, ignore_errors=True)
    print(f"total {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
