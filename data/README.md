# data/

Nothing in this folder is committed except this file and `checksums.sha256`.

- `raw/freddie/`: Freddie Mac Single-Family Loan-Level Dataset sample files, downloaded by hand after free registration (`docs/runbook.md`). Expected names: `sample_orig_YYYY.txt` and `sample_perf_YYYY.txt`, unzipped from `sample_YYYY.zip` directly into this folder (no subfolders); `reserve doctor` checks this.
- `raw/fred/`: one Parquet table per series (every ALFRED vintage) saved by `reserve prepare macro` (needs `FRED_API_KEY`).
- `synthetic/`: files written by the synthetic simulator, in the Freddie Mac layout. Regenerated from a seed; never used for a headline.
- `interim/`: Parquet tables built by `reserve prepare`, plus `prepare.json`, which records the raw files read and whether any record carried the simulator's SYNTHETIC markers (methodology M40).
- `checksums.sha256` (tracked): SHA-256 of every raw file, written by `reserve checksums` before sealing and committed with the lock; check by hand with `sha256sum -c data/checksums.sha256` (methodology M42).

No real customer personal data is used. The Freddie Mac files are public, anonymized loan records, used under Freddie Mac's dataset terms and never redistributed here.
