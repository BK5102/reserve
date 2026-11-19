"""Checksums of the raw downloads, kept in `data/checksums.sha256` (methodology M42).

The file uses the format `sha256sum` writes and checks: one line per file, the 64-digit
SHA-256 in hex, two spaces, then the path relative to the repo root with forward slashes.
Lines starting with "#" are comments. `reserve checksums --config CONFIG` writes it from
the files actually present; `reserve seal` refuses to seal unless every raw file is listed
and still matches, so the lock always names data someone can check.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from cleanroom import provenance

from reserve.data import loans, macro

DEFAULT_PATH = "data/checksums.sha256"
LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")
HEADER = (
    "# SHA-256 of the raw downloads (methodology M42), written by `reserve checksums`.\n"
    "# Check by hand with: sha256sum -c data/checksums.sha256\n"
)


def path_for(cfg: dict[str, Any]) -> Path:
    return Path(cfg["data"].get("checksums", DEFAULT_PATH))


def raw_files(cfg: dict[str, Any]) -> dict[str, list[Path]]:
    """The raw files a Freddie Mac config reads: sample pairs, plus saved FRED tables."""
    data = cfg["data"]
    freddie = [p for _, o, pf in loans.find_sample_files(data["raw_dir"]) for p in (o, pf)]
    fred_dir = data.get("fred_dir")
    fred = []
    if fred_dir:
        fred = [Path(fred_dir) / f"{s}.parquet" for s in macro.SERIES]
        fred = [p for p in fred if p.exists()]
    return {"freddie": freddie, "fred": fred}


def _key(path: Path) -> str:
    return path.as_posix()


def write(cfg: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    """Hash every raw file and write the checksum file. Returns (its path, {path: sha256})."""
    files = raw_files(cfg)
    if not files["freddie"]:
        raise FileNotFoundError(
            f"no sample_orig_YYYY.txt / sample_perf_YYYY.txt pairs in {cfg['data']['raw_dir']}; "
            "nothing to checksum (docs/runbook.md, step 1)"
        )
    sums = {_key(p): provenance.file_sha256(p) for p in files["freddie"] + files["fred"]}
    out = path_for(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{digest}  {name}\n" for name, digest in sums.items())
    out.write_text(HEADER + body, encoding="utf-8", newline="\n")
    return out, sums


def read(path: str | Path) -> dict[str, str]:
    """{path: sha256} from a checksum file; raises ValueError on a malformed line."""
    out = {}
    for i, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE.match(line)
        if not m:
            raise ValueError(f"{path} line {i} is not '<sha256>  <path>'")
        out[m.group(2)] = m.group(1)
    return out


def problems(cfg: dict[str, Any]) -> list[str]:
    """Why the recorded checksums do not cover the raw data exactly ([] if they do)."""
    path = path_for(cfg)
    if not path.exists():
        return [f"{path} does not exist; run `reserve checksums --config <config>` first"]
    try:
        recorded = read(path)
    except ValueError as err:
        return [str(err)]
    files = raw_files(cfg)
    found: list[str] = []
    if not files["freddie"]:
        found.append(f"no Freddie Mac sample files in {cfg['data']['raw_dir']}")
    if not any(name.rsplit("/", 1)[-1].startswith("sample_") for name in recorded):
        found.append(f"{path} lists no Freddie Mac sample file; run `reserve checksums`")
    current = {_key(p): p for p in files["freddie"] + files["fred"]}
    for name, p in current.items():
        if name not in recorded:
            found.append(f"{name} is not in {path}; run `reserve checksums` again")
        elif provenance.file_sha256(p) != recorded[name]:
            found.append(f"{name} changed since its checksum was recorded")
    for name in recorded:
        if name not in current:
            found.append(f"{path} lists {name}, which is not in the raw folders")
    return found
