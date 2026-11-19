"""`reserve doctor`: is everything in place for the next step of docs/runbook.md? (M41)

Checks, without reading any data values: the Freddie Mac folder (sample pairs found, a
partner file missing, zip files not unzipped, files left in a subfolder), the saved FRED
tables, whether FRED_API_KEY and ANTHROPIC_API_KEY are set (never printing them), whether
each TSFM in the config has its packages, the checksum file, and the git tree.

Exit code 0 when the Freddie Mac files are ready for `reserve prepare`, 3 when something
blocks that. Everything else is reported as a note for the step that needs it.
"""

from __future__ import annotations

import os
import subprocess
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from reserve import checksums
from reserve.data import loans, macro


def _freddie(raw_dir: Path) -> tuple[list[str], list[str]]:
    lines, blocking = [], []
    if not raw_dir.exists():
        blocking.append(f"{raw_dir} does not exist: create it and unzip the sample files there")
        return lines, blocking
    pairs = loans.find_sample_files(raw_dir)
    years = [y for y, _, _ in pairs]
    if pairs:
        lines.append(f"OK    {raw_dir}: {len(pairs)} sample years ({years[0]} to {years[-1]})")
        gaps = sorted(set(range(years[0], years[-1] + 1)) - set(years))
        if gaps:
            lines.append(f"NOTE  years missing inside that range: {gaps}")
    else:
        blocking.append(f"{raw_dir} has no sample_orig_YYYY.txt / sample_perf_YYYY.txt pairs")
    paired = {p.name for _, o, pf in pairs for p in (o, pf)}
    for f in sorted(raw_dir.glob("sample_*_*.txt")):
        if f.name not in paired:
            blocking.append(f"{f.name} has no partner file (orig and perf come in pairs)")
    zips = sorted(raw_dir.glob("*.zip"))
    unzipped = {str(y) for y in years}
    waiting = [z.name for z in zips if z.stem.removeprefix("sample_") not in unzipped]
    if waiting:
        blocking.append(f"zip files not unzipped yet: {waiting[:5]}; unzip them into {raw_dir}")
    nested = sorted(p for p in raw_dir.rglob("sample_*_*.txt") if p.parent != raw_dir)
    if nested:
        blocking.append(
            f"{len(nested)} sample files are in subfolders "
            f"(e.g. {nested[0].relative_to(raw_dir)}); move them directly into {raw_dir}"
        )
    return lines, blocking


def _git_state() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "NOTE  not a git checkout: sealed runs need one"
    if dirty:
        return f"NOTE  git {commit}: uncommitted changes; a sealed run needs a clean tree"
    return f"OK    git {commit}: clean"


def check(cfg: dict[str, Any]) -> tuple[list[str], bool]:
    """(report lines, ready): ready means the Freddie Mac files can be prepared."""
    from reserve.models.tsfm import missing_packages

    data = cfg["data"]
    lines, blocking = _freddie(Path(data["raw_dir"]))
    fred_dir = Path(data.get("fred_dir", "data/raw/fred"))
    have = [s for s in macro.SERIES if (fred_dir / f"{s}.parquet").exists()]
    if len(have) == len(macro.SERIES):
        lines.append(f"OK    {fred_dir}: all {len(have)} FRED tables saved")
    else:
        lines.append(
            f"NOTE  {fred_dir}: {len(have)} of {len(macro.SERIES)} FRED tables; run "
            "`reserve prepare macro` (step 4)"
        )
    for env, step in (("FRED_API_KEY", "4, macro download"), ("ANTHROPIC_API_KEY", "13, agent")):
        state = "set" if os.environ.get(env, "").strip() else "not set"
        lines.append(f"{'OK   ' if state == 'set' else 'NOTE '} {env} {state} (step {step})")
    for name in cfg["contestants"].get("tsfm") or []:
        missing = missing_packages(name) if name != "fake" else []
        if missing:
            extra = "timesfm" if name == "timesfm_2p5" else "tsfm"
            lines.append(f"NOTE  TSFM {name} needs {missing}: uv sync --extra {extra}")
        else:
            lines.append(f"OK    TSFM {name}: packages installed")
    agent = "installed" if find_spec("anthropic") else "not installed (uv sync --extra agent)"
    lines.append(f"{'OK   ' if find_spec('anthropic') else 'NOTE '} agent extra {agent}")
    path = checksums.path_for(cfg)
    if path.exists():
        lines.append(f"OK    {path}: {len(checksums.read(path))} checksums recorded")
    else:
        lines.append(f"NOTE  {path} not written yet (`reserve checksums`, step 8)")
    lines.append(_git_state())
    lines += [f"FIX   {b}" for b in blocking]
    return lines, not blocking
