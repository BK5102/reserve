"""`reserve doctor` and `reserve prepare macro --cassette` (methodology M41)."""

import zipfile
from pathlib import Path

import polars as pl

from reserve import cli, doctor

FIX = Path(__file__).parent / "fixtures"


def _cfg(tmp_path: Path) -> dict:
    return {
        "data": {
            "raw_dir": str(tmp_path / "freddie"),
            "fred_dir": str(tmp_path / "fred"),
            "checksums": str(tmp_path / "checksums.sha256"),
        },
        "contestants": {"tsfm": ["fake"]},
    }


def test_doctor_blocks_until_the_files_are_unzipped_and_flat(tmp_path, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    cfg = _cfg(tmp_path)
    lines, ready = doctor.check(cfg)
    assert not ready and any("does not exist" in x for x in lines)
    raw = tmp_path / "freddie"
    raw.mkdir()
    with zipfile.ZipFile(raw / "sample_2005.zip", "w") as z:  # downloaded, not unzipped
        z.writestr("sample_orig_2005.txt", "x")
    nested = raw / "sample_2006"
    nested.mkdir()
    (nested / "sample_orig_2006.txt").write_text("x")  # unzipped into a subfolder
    (raw / "sample_perf_2007.txt").write_text("x")  # partner missing
    lines, ready = doctor.check(cfg)
    text = "\n".join(lines)
    assert not ready
    assert "not unzipped" in text and "subfolders" in text and "no partner" in text
    assert "FRED_API_KEY not set" in text and "sk-" not in text


def test_doctor_is_ready_with_flat_pairs_and_never_prints_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "secret-value-123")
    raw = tmp_path / "freddie"
    raw.mkdir()
    for name in ("sample_orig_2005.txt", "sample_perf_2005.txt"):
        (raw / name).write_text((FIX / "freddie" / name).read_text())
    lines, ready = doctor.check(_cfg(tmp_path))
    text = "\n".join(lines)
    assert ready and "1 sample years (2005 to 2005)" in text
    assert "FRED_API_KEY set" in text and "secret-value-123" not in text


def test_prepare_macro_replays_a_cassette_without_a_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    out = tmp_path / "fred"
    cassette = str(FIX / "fred" / "cassette.json")
    assert cli.main(["prepare", "macro", "--cassette", cassette, "--out", str(out)]) == 0
    assert "replayed from cassette" in capsys.readouterr().out
    unrate = pl.read_parquet(out / "UNRATE.parquet")
    assert unrate.height > 0 and unrate["series_id"].unique().to_list() == ["UNRATE"]
    assert cli.main(["prepare", "macro", "--cassette", cassette, "--record", "x.json"]) == 2
