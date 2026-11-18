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

## 3. Vintage curves

**M7. The event is "ever 90 or more days delinquent, or defaulted", the first time it happens.**
- What: a loan's event month is the first month in which its delinquency status is 03 or higher (90 days or more), its status is RA (REO acquisition, meaning the lender took the house back), or it leaves the data with a credit-related zero balance code: 02 third party sale, 03 short sale or charge off, 09 REO disposition, 15 whole loan sale. Each loan has at most one event.
- Why: 90 days late is the usual "serious delinquency" line and comes before most defaults, so it is the early-warning signal a lender watches. Adding the credit-related exits catches loans that were sold or settled before reaching 90 days. Counting only the first time makes the curve cumulative: it can only go up, like a class's share of students who have ever failed a course. The owner approved this target (NEXT_PROJECTS_PLAN.md, decision 3).
- Why code 15 counts: Freddie Mac computes Actual Loss for codes 02, 03, 09, and 15 only (User Guide, "Actual Loss"), which marks 15 as a credit event. Code 16 (reperforming loan securitization) and 96 (defect) are not counted as events; a loan's earlier 90-day delinquency, if any, is already counted.
- Considered: 60+ days (more events, but more loans cure); "currently 90+" (goes up and down, so it is not a clean curve); default only (fewer events, and the timing depends on slow foreclosure processes).
- Limitation: during COVID forbearance in 2020 and 2021, borrowers who paused payments with permission could still be reported as delinquent, so the event then partly measures forbearance rather than distress. Those months are excluded from the headline (M12).

**M8. Vintage age counts months since the end of the origination quarter, not each loan's own age.**
- What: age 1 of vintage 2005Q1 is calendar month 2005-04 for every loan in it. Any event recorded before age 1 is counted at age 1.
- Why: with one calendar month per (vintage, age) cell, "what was known at forecast date t" is exact: a cell is known once its month has passed. Each loan's own age would put a vintage's loans at different ages in the same calendar month, and a cell would only be complete after its latest loan reached that age. Freddie Mac's Loan Age field also restarts after a loan modification (User Guide, "Loan Age"), so it cannot be used as is.
- Considered: months since each loan's first payment date (closer to "loan age", but breaks the one-month-per-cell property); quarters instead of months (fewer, noisier steps).
- Limitation: loans originated early in a quarter are up to two months older than loans originated late in it at the same vintage age. Events before age 1 are shifted later, never earlier, so nothing leaks from the future, but the first cell can hold a few early events.

**M9. The denominator is the original loan count; prepaid and other exiting loans stay in it.**
- What: `cum_rate = loans with an event by this age / loans originated in the vintage`. Prepaid loans (zero balance code 01), reperforming sales (16), defects (96), and loans that vanish from the data without a code all remain in the denominator and simply can no longer have an event.
- Why: this is the standard "cumulative default rate by original count" view, the share of the original class that ever went bad. It never decreases and needs no assumption about what a prepaid loan would have done. The owner approved it (decision 3).
- Considered: a life-table estimate that divides each month's events by the loans still at risk and compounds them. That answers a different question ("if no one prepaid"), depends on the prepayment pattern, and is used inside the hazard model instead (M10).
- Limitation: a vintage that prepays fast (for example in a refinancing boom) shows a lower curve partly because fewer loans remained exposed. The curve mixes credit quality and prepayment speed, by design.
- Cells stop at the data cutoff (the last performance month), so unobserved months are missing, not zero. Vintages with no complete month are dropped. A loan that disappears before the cutoff without a zero balance code is counted in `exits_unexplained` and reported by the reconciliation.

**M10. Each cell also records the loans at risk.**
- What: `at_risk` is the number of loans observed and event-free at the start of an age: a loan is at risk from its first observed month through its event month or, without an event, its last observed month. Each cell also counts new events, prepayments, other exits, and unexplained exits.
- Why: the hazard regression (contestant C) needs the true denominator for "chance of going bad this month, given not bad yet", and the counts let a test check the identity at_risk(next age) = at_risk - events - exits + loans first seen at the next age.
- Limitation: loans enter the data when Freddie Mac acquires them, sometimes a few months after the first payment. Before that they are not observed, and an event in that gap would be missed. Freddie Mac rarely buys a loan that is already 90 days late, so this is expected to be small; the data-fit check reports how many loans are first seen after age 1.

**M11. Reconciliation runs on every build.**
- What: `vintages.build` raises an error unless: there is one outcome row per originated loan; no performance loan lacks an origination record; no (loan, month) row is duplicated; each vintage's denominator equals its loan count; the events in the curves equal the loans with an event; no curve decreases; and the at-risk identity holds in every cell.
- Tests: hand-computed values on the fake fixture, a brute-force loan-by-loan count on random portfolios, and a past-only test that curves built from data up to month t equal the full curves restricted to months up to t.
- Limitation: reconciliation proves internal consistency, not that Freddie Mac's data is right.

---

## 4. Macro series

**M13. Three FRED series: UNRATE, MORTGAGE30US, and HPIPONM226S.**
- What, checked on each FRED series page on 2026-11-17:
  - [UNRATE](https://fred.stlouisfed.org/series/UNRATE): Unemployment Rate, percent, monthly, seasonally adjusted, from the Bureau of Labor Statistics (Employment Situation release).
  - [MORTGAGE30US](https://fred.stlouisfed.org/series/MORTGAGE30US): 30-Year Fixed Rate Mortgage Average in the United States, percent, weekly ending Thursday, not seasonally adjusted, from Freddie Mac's Primary Mortgage Market Survey.
  - [HPIPONM226S](https://fred.stlouisfed.org/series/HPIPONM226S): Purchase Only House Price Index for the United States, index January 1991 = 100, monthly, seasonally adjusted, from the Federal Housing Finance Agency (FHFA).
- Why: job loss, the cost of refinancing, and house prices are the three standard drivers of mortgage delinquency. A borrower who loses a job cannot pay, one whose rate is above the market rate may refinance away, and one with falling house prices cannot sell to escape. These are PLAN.md FR3's three series.
- Considered: the FHFA all-transactions index (USSTHPI), which is quarterly; monthly was preferred to match the monthly curves. State-level unemployment and house prices would match loans to local conditions but need a loan-to-state mix per vintage; deferred.
- Limitation: national series only. A vintage concentrated in one region feels local conditions the national series miss. HPIPONM226S's first observation month and how far back its ALFRED vintages go were not checked (the FRED page shows recent values only); `check_data_fit.py` reports both once the key exists.

**M14. As-of mode: every forecast uses the version of each series that was published by the end of the forecast month.**
- What: one API request per series asks for every vintage (`realtime_start=1776-07-04`, `realtime_end=9999-12-31`, [FRED API docs](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)). Each row then carries the dates during which that value was the published one. For month t, Reserve keeps the rows valid on the last day of t, so a later revision cannot leak back. This is the "as-of join".
- Why: unemployment and house prices are revised after first release. Using today's revised numbers would give a forecast information nobody had at the time (CLAUDE.md rule 1). Analogy: grading a weather forecaster against the radar they actually had that morning, not a cleaned-up map from a week later.
- Considered: downloading one request per forecast date with `vintage_dates`; more requests for the same result.
- Limitation: FRED's missing-value marker is "." in practice, but the API page does not say so (unverified; the parser treats "." as missing). Whether ALFRED vintages exist for every month of every series back to 1999 is unverified until the key exists.

**M15. Latest mode is a fallback, and it is always flagged.**
- What: when as-of data is not available, Reserve uses today's values and treats observation month m as known at the end of month m + lag: lag 1 for UNRATE, 0 for MORTGAGE30US, 2 for HPIPONM226S. Every row then carries `revisions_ignored = true`, and that flag is carried into results.
- Why the lags: on the FRED pages checked 2026-11-17, UNRATE's October 2026 value was published 2026-11-06 (one month), HPIPONM226S's latest value was August 2026 at the 2026-10-27 update (two months), and MORTGAGE30US is published weekly for the week just ended.
- Considered: no lag at all. Rejected: it would use a month's unemployment before it was published.
- Limitation: revisions leak in latest mode. A test shows the revised value being used as if known earlier, so the flag is not decorative.

**M16. Four past-only macro features, as known at the end of each month.**
- What: newest known unemployment rate; its change over the 12 months to that newest month (both from the same vintage); newest known monthly average 30-year mortgage rate (weekly values averaged within the calendar month); and the log change in the house price index over the 12 months to its newest known month.
- Why: levels and 12-month changes are the usual macro inputs to credit models; changes capture "conditions are worsening" better than levels. Using the newest month known, rather than a fixed calendar month, means each feature uses exactly what was published.
- Leak test: for every month t, features computed from all vintages equal features computed after deleting, or multiplying by 100, every value published after the end of t.
- Considered: more series or lags. Deferred: with about 25 years of history, a small set keeps the hazard regression stable.
- Limitation: the features describe the economy up to the forecast date only. Contestant C must hold them fixed over the forecast horizon (M26), because future macro values are unknown at the origin.

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
| L6 | COVID forbearance shows up as delinquency in 2020 and 2021 | The event then partly measures permitted payment pauses | Forbearance share measured by the data-fit check; those months excluded from the headline (M12) |
| L7 | Vintage age is calendar-aligned, not loan-level | Loans in a quarter differ by up to two months in true age | Stated (M8); the alternative breaks the exact "known at t" property |
| L8 | Curves by original count mix credit and prepayment speed | Fast-prepaying vintages look better partly because fewer loans stayed exposed | Stated (M9); the hazard model uses at-risk counts instead |
| L9 | Events before Freddie Mac acquires a loan are not observed | A few early events could be missed | Data-fit check reports loans first seen after age 1 |
| L10 | National macro series only | Regional shocks are missed | Stated (M13); state-level series deferred |
| L11 | Latest mode leaks revisions | Forecasts could use revised values unknown at the time | As-of mode by default; latest mode sets `revisions_ignored` on every row and result (M15) |
| L12 | ALFRED vintage coverage and the "." missing marker are unverified | As-of mode may have gaps for early years | Checked by the data-fit script once the FRED key exists (M14) |
