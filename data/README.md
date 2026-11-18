# data/

Nothing in this folder is committed except this file and `checksums.sha256`.

- `raw/freddie/`: Freddie Mac Single-Family Loan-Level Dataset sample files, downloaded by hand after free registration (see the README, "Manual steps"). Expected names: `sample_orig_YYYY.txt` and `sample_perf_YYYY.txt`, unzipped from `sample_YYYY.zip`.
- `raw/fred/`: FRED and ALFRED JSON responses saved by `reserve prepare macro` (needs `FRED_API_KEY`).
- `synthetic/`: files written by the synthetic simulator, in the Freddie Mac layout. Regenerated from a seed; never used for a headline.
- `interim/`: Parquet tables built by `reserve prepare`.

No real customer personal data is used. The Freddie Mac files are public, anonymized loan records, used under Freddie Mac's dataset terms and never redistributed here.
