# Methodology: every choice, why it was made, and what it costs

This file is the reasoning behind Reserve. Each decision has the same four parts: **what** was chosen, **why**, **what else was considered**, and the **limitation** it brings. Decision ids (M1, M2, ...) are stable so other docs can point to them. `docs/decisions.md` is the dated diary of when things were built; this file is the argument for why they were built that way.

Rule for contributors (also in `CLAUDE.md`, rule 13): any new method, variable, threshold, default, or dataset choice gets an entry here in the same commit, and any new limitation goes in the register at the end.

Status: no real data has been loaded yet. Every number in this file comes from documentation or from synthetic test data, and says so.

---

## 1. The question and the setup

**M1. The question is "does a zero-shot TSFM forecast the next 12 months of vintage delinquency curves better than the classical methods, with honest intervals?"**
- What: a vintage is all loans originated in one calendar quarter. Its curve is the cumulative share of those loans that have ever been 90 or more days late or defaulted, by loan age in months. At each forecast origin (a calendar month), every contestant forecasts each vintage's curve for the next 12 months as quantiles.
- Why: reserves and lending strategy are set at portfolio level, and a vintage curve is a time series, which is what a TSFM is built for (PLAN.md Part 2). A school tracking each graduating class is the analogy.
- Considered: loan-level default scoring. Rejected for v0.1: a different, well-studied problem, and not what TSFMs are for (PLAN.md non-goals).
- Limitation: the answer is about portfolio curves on prime, conforming US mortgages, not about all consumer credit.

**M2. Data: Freddie Mac Single-Family Loan-Level Dataset sample files, plus FRED and ALFRED macro series.**
- Why: both are free and public. The Freddie Mac sample is a simple random sample of 50,000 loans per full vintage year with the same fields as the full dataset ([General User Guide, Release 48, October 2026](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf), "Single-Family Loan-Level Dataset Sample"). That is small enough for a laptop and spans 1999 to the latest release, which covers the 2007 to 2009 recession. FRED is free with an API key, and its ALFRED archive keeps earlier versions of revised series, so forecasts can use the values that were known at the time.
- Considered: Fannie Mae Single-Family Loan Performance Data (also free with registration). Kept as a possible second dataset; one dataset first, as in the agreed plan (`NEXT_PROJECTS_PLAN.md`). The full Freddie Mac dataset (about 49.5 million loans in Release 48, per the Release Notes) is too large for the first pass and is not needed for quarterly vintage curves.
- Limitation: Freddie Mac loans are prime, conforming, fully amortizing fixed-rate mortgages in the Standard Dataset (User Guide, "Loan Selection Criteria"), not all consumer credit. The owner has not registered yet, so no real data exists in this repo; every pipeline is tested on synthetic data written in the exact file layout (M7).

**M3. CPU only, for every run including the full run.**
- Why: the owner has no GPU, and Tells showed that a GPU-only step leaves results resting on one hard-to-repeat run (Tells limitations L15 and L16). PyTorch, when the optional `tsfm` extra is installed, comes from the CPU-only wheel index.
- Considered: planning a GPU run for larger TSFMs. Rejected by the owner (NEXT_PROJECTS_PLAN.md, decision 6).
- Limitation: larger TSFMs may be too slow to run on every origin; the small Chronos-Bolt-tiny model is the first TSFM for that reason (M27).

**M4. The Cleanroom is imported from the Tells repo at tag `cleanroom-v0.1`, never copied.**
- What: `pyproject.toml` pins `cleanroom` to the git tag `cleanroom-v0.1` (subdirectory `packages/cleanroom`). Reserve uses its `split`, `lock`, `provenance`, and `stats` modules.
- Why: one shared, tested toolkit means a leak fix in one project fixes all three (Tells `docs/decisions.md` D1).
- Considered: copying the code. Rejected: copies drift.
- Limitation: v0.1 lacks forecast losses, a Diebold-Mariano test, rolling origins, and an as-of join. Reserve has small local stand-ins for the first three pieces it needs now, each marked `TODO(cleanroom-v0.2)` (M30).

---

## 2. Loading the loan files

**M5. The file layout comes from Freddie Mac's own documents, checked on 2026-11-17; each field is marked verified or unverified.**
- Sources (all linked from the [SFLLD page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset), downloaded 2026-11-17; the first 16 hex digits of each file's SHA-256 are in brackets):
  - [General User Guide](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf). Its title page says "Release 48, October 2026", although the file name says July 2026 [0884a958ebfd58af].
  - [File Layout spreadsheet](https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx): field position, data type, maximum length [e4ee4ba00b5b1a8d].
  - [File header files](https://www.freddiemac.com/fmac-resources/research/docs/file_headers_july_2026.zip): the official attribute names in file order [db1390531ca9ea9c].
  - [SFLLD Disclosure Changes Effective July 2026](https://www.freddiemac.com/fmac-resources/research/pdf/disclosure-changes-summary.pdf), version 1.2, last updated 6/2026: old and new positions for Release 47 [e513ccc06fca1388].
  - [Release Notes, Release 48](https://www.freddiemac.com/fmac-resources/research/pdf/release_notes.pdf): origination and performance cutoff 2026-06-30 [8c5382463d67e36b].
- Verified from those documents, and enforced by `tests/test_loans.py`:
  - Pipe delimited, no header row. Sample files are named `sample_orig_YYYY.txt` and `sample_perf_YYYY.txt` inside `sample_YYYY.zip`. The sample is a simple random sample of 50,000 loans per full vintage year with the same fields as the full dataset.
  - Current layout (Release 47 and later): 31 origination fields and 35 performance fields, in the order of the official header files. A test compares the loader's field list with the header text character by character.
  - Loan Identifier format PYYQnXXXXXXX, where YYQn is the origination year and quarter. Reserve takes the vintage from it.
  - Delinquency status codes: 00 (current or under 30 days), 01 (30 to 59 days), 02 (60 to 89), 03 (90 to 119), and so on, capped at 99; RA for REO Acquisition (the lender took the house back); XX for not available. For periods up to September 2025, 00 could also mean "cycle before first payment date".
  - Zero Balance Codes 01 (prepaid or matured), 02 (third party sale), 03 (short sale or charge off), 09 (REO disposition), 15 (whole loan sales), 16 (reperforming loan securitizations), 96 (confirmed underwriting or servicing defect).
  - Not-available codes: 9999 for both credit scores, 999 for MI percent, CLTV, LTV, DTI, and ELTV, 99 for units and number of borrowers, 7 for property valuation method. The loader turns these into nulls.
  - Since Release 47, recoveries and gains are negative and expenses and losses are positive.
  - Loan Age in the performance file restarts from the modification first payment date when a loan is modified (User Guide, "Loan Age"). Reserve therefore computes its own age from the first payment date (M8).
- Unverified, and how the code handles it:
  - The legacy (pre-Release-47) layout: 31 origination fields with Servicer Name at position 25 and no VantageScore, and the first 32 performance fields. This is read from the "current" column of the disclosure-changes PDF, whose extracted text is partly garbled (the old location of the MI cancellation indicator is unreadable). The loader supports it and detects it from field 31 (Y or N in legacy files, digits in current ones). Check one real legacy file before relying on it.
  - Unpadded legacy delinquency codes (0, 1, 2) and blank for "not available": inferred from the disclosure-changes "current" column. Both forms are accepted.
  - Whether real files end lines with a trailing pipe: not stated. The loader accepts both.
  - Whether the sample files on Clarity follow the Release 47 layout today: the User Guide says the sample has the same fields as the full dataset, but no real file has been opened. `scripts/check_data_fit.py` prints the detected layout of every file it reads.
- Considered: reading the field list from Freddie Mac's header file at run time. Rejected: the header file is a separate download, and hard-coding it with a character-level test against the official text catches a change equally well.
- Limitation: until the owner registers and downloads a file, the loader has only been run on hand-built fake records (`tests/fixtures/freddie/`).

**M6. Every field is read as text first and cast explicitly, with strict casts.**
- Why: a silent type guess could turn "03" into 3 in one file and keep "RA" as text in another. Strict casts make a layout surprise fail loudly instead of producing wrong numbers.
- Considered: letting the CSV reader infer types. Rejected for the reason above.
- Limitation: one malformed value stops the whole file from loading. The error names the column, which is the point.

---

## Limitations register

The honest list. Each item says what could be wrong and what was done about it.

| # | Limitation | Effect | What was done |
|---|---|---|---|
| L1 | No real data yet | Nothing about real loans is known; all checks ran on synthetic data | Stated everywhere; `scripts/check_data_fit.py` is ready to run once the owner registers |
| L2 | Prime, conforming fixed-rate mortgages only | Results do not cover subprime, cards, or auto loans | Stated up front (M2) |
| L3 | Cleanroom v0.1 lacks forecast tools | Local stand-ins could differ from the shared versions | Marked `TODO(cleanroom-v0.2)`; switch when the tag exists (M30) |
| L4 | Legacy (pre-Release-47) file layout partly unverified | Older downloads could be misread | Layout detected from field 31 and the field count; flagged in M5; check a real legacy file first |
| L5 | Loader tested only on hand-built fake records | Real-file quirks are unknown | `check_data_fit.py` reports the layout, counts, and reconciliation on the first real run |
