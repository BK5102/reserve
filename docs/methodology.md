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

**M4. The Cleanroom is imported from the Tells repo at tag `cleanroom-v0.2`, never copied.**
- What: `pyproject.toml` pins `cleanroom` to the git tag `cleanroom-v0.2` (subdirectory `packages/cleanroom`). Reserve uses `split` (Split, Window, `rolling_origins`), `asof` (`join_as_known`), `forecast` (pinball loss, weighted quantile loss, interval coverage, interval score, relative error), `stats` (Diebold-Mariano with the Harvey-Leybourne-Newbold correction, Holm), `lock` and `guard` (sealed runs), and `provenance`.
- Why: one shared, tested toolkit means a leak fix in one project fixes all three (Tells `docs/decisions.md` D1).
- Considered: copying the code. Rejected: copies drift. Reserve started on `cleanroom-v0.1`, which lacked the forecasting pieces; it was bumped to v0.2 on 2026-11-17, as soon as the tag existed, before any local stand-in was written.
- Limitation: Reserve's behaviour now depends on a package in another repo. The tag is fixed, so it changes only when Reserve bumps it on purpose.
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
- Limitation: during COVID forbearance in 2020 and 2021, borrowers who paused payments with permission could still be reported as delinquent, so the event then partly measures forbearance rather than distress. Those months are excluded from the headline (M20).

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

**M12. Headline horizon: 12 months ahead, from origins every 3 months.**
- What: each forecast covers the next 12 months of every eligible vintage's curve; the headline scores the 12-month-ahead point, and losses at every horizon from 1 to 12 are reported next to it. Origins are quarter ends, every 3 months.
- Why: a year is the usual planning horizon for credit loss reserves and the horizon in the agreed plan (NEXT_PROJECTS_PLAN.md, "the next 12 months"). Quarterly origins match how vintages are formed and keep the number of correlated forecasts manageable.
- Considered: 24 months (PLAN.md mentions 12 to 24), which would halve the number of independent test windows (M21); monthly origins, which add forecasts that are almost copies of their neighbours.
- Limitation: a 12-month horizon says nothing direct about lifetime losses, which reserving standards also consider.

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
- How: the selection is the shared `cleanroom.asof.join_as_known` (latest release per observation date whose release date is on or before the query date; `inclusive=True` because a FRED vintage date is the day the value became available). A latest release of "." counts as not available.
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

## 5. Synthetic data

**M17. A synthetic portfolio with known answers, used only for tests and pipeline checks, never for a headline.**
- What: `src/reserve/data/synthetic.py` writes fake loans in the exact Freddie Mac file layout (current release), plus FRED-style macro tables and the true expected curves. Every synthetic record is marked: seller and servicer "SYNTHETIC", property state "ZZ".
- Why: no real data exists yet, and a simulator whose true answer is known is the only way to check that the pipeline computes what it claims. Tells used real public benchmark data; Reserve cannot until the owner registers.
- Considered: waiting for real data before building anything. Rejected: the loader, builder, models, and checks can all be tested now, and the data-fit check is ready the day the files arrive.
- Limitation: a simulator only contains the patterns written into it. A model that wins on synthetic data proves the plumbing works, nothing about real loans. No synthetic number appears in `docs/findings.md`.

**M18. The simulator's true hazard has age, season, cohort, and macro parts, plus a recession shock.**
- What: an event-free, active loan of vintage v at vintage age a in month p goes 90+ days late with chance exp(base + age(a) + season(p) + cohort(v) + beta x (unemployment(p) - 5)), capped at 0.5. `age(a)` is a hump peaking at 30 months and zero chance below age 3; `cohort(v)` is a random vintage quality effect (sd 0.25); season is a small yearly wave; unemployment is 5% plus a recession bump (+5 points peaking 2009-10 by default). Loans also prepay with a fixed 1.5% monthly chance. After an event, 30% cure and the rest are disposed of six months later (zero balance code 09).
- Known answers, each tested: the vintage builder recovers every loan's simulated event age exactly; observed events by age 24 are within binomial noise of the exact expected curve; the true hazard ratio between recession peak and a calm month equals exp(beta x rise) after removing the seasonal part; a forbearance window produces events flagged "F".
- Unemployment is published FRED-style: first with noise, then revised a month later, so as-of features differ from the truth just as real first releases do.
- Considered: an aggregate-only simulator that produces curves directly. Rejected: it would skip the loader and the builder, which most need testing.
- Limitation: prepayment does not depend on rates, and there are no regional effects. Both are simplifications of real mortgages.

---

## 6. Data fit

**M19. Measure whether the data can answer the question before building models on it.**
- What: `scripts/check_data_fit.py` reports, for a folder of Freddie Mac sample files: each file's detected layout and row count; loans per origination quarter; cumulative events by age 12 per vintage and their relative binomial standard error; whether a simple class-average forecast is already at the noise floor; recession coverage; the COVID forbearance check (M20); every reconciliation count; power (M21); the latest month available; and, with saved FRED tables, each macro series' first observation, number of ALFRED vintages, and missing values.
- Gate (from NEXT_PROJECTS_PLAN.md): at least 60 usable vintages, and at least 20 events per vintage by age 12. "Usable" means observed through age 24 (12 months of history plus a 12-month forecast). The plan's "20 events per cell at age 12" is applied to the median usable vintage's cumulative events by age 12, because the target is cumulative and single-month counts at age 12 are far smaller; the share of vintages meeting 20 and the single-month counts are reported next to it.
- Noise-floor check: for each vintage, the 12-month increment from age 12 to 24 is forecast by the mean increment of the previous four vintages, and the root mean squared error is divided by the binomial standard deviation of the increment. A ratio near 1 means even a perfect model would gain little, the same trap as Tells' TabFormer ceiling.
- Why: Tells' biggest lesson was that two popular datasets could not test its idea at all, which only measurement revealed (Tells `docs/data-fit.md`).
- Considered: going straight to models on the sample. Rejected for that reason.
- Limitation: run so far only on synthetic data (`docs/data-fit.md`). The 20-event rule and the four-vintage class average are judgment calls, both settings in `FitSettings`.

**M20. The COVID forbearance check, and why 2020 to 2021 is excluded from the headline.**
- What: for events dated 2020-03 to 2021-12, the check reports how many there were, the share flagged as forbearance (Borrower Assistance Plan "F" or Delinquency Due to Disaster "Y" in the event month; both fields are populated from January 2014 per the User Guide), and the largest monthly event count against the 2019 monthly mean.
- Why: under COVID relief, borrowers could pause payments with permission and still be reported as delinquent, so a spike in "90+ days late" then measures relief uptake as much as distress. Like the end-of-file artifact Tells removed from AMLWorld, it would reward a model for learning a one-off pattern. The owner approved excluding 2020 to 2021 from the headline and keeping the 2007 to 2009 recession descriptive (NEXT_PROJECTS_PLAN.md, decision 4).
- Considered: dropping only forbearance-flagged events. Rejected for the headline: the flags may be incomplete, and unflagged loans were also affected by moratoria; it is kept as a possible sensitivity check.
- Limitation: the share flagged is only as good as servicer reporting, which is unverified until real data is read.

**M21. Power is reported as a minimum detectable effect in standard-deviation units.**
- What: MDE = (z(1 - alpha / (2m)) + z(power)) / sqrt(n), with alpha 0.05, power 0.8, and m = 3 comparisons (Bonferroni as a conservative stand-in for Holm). n is reported twice: the number of quarterly test origins from 2013-03 to 2018-12 whose 12-month target is observed (24), and the number of non-overlapping 12-month windows in that span (6). The truth lies between: forecasts from origins less than 12 months apart share outcome months.
- Why: it says in advance how large a consistent advantage would have to be before the test could see it, so a null result is not over-read. Analogy: before a race, know how many seconds apart two runners must be for your stopwatch to tell them apart.
- Considered: a simulation-based power analysis on the synthetic portfolio. Deferred: it would only describe the simulator.
- Limitation: the MDE is in units of the standard deviation of per-origin loss differences, which is unknown until models run. With 6 independent windows, only a large effect (about 1.3 standard deviations) is reliably detectable. The test window (2013-03 to 2018-12) is from the agreed plan.

---

## 7. Contestants and the forecasting harness

**M22. Rolling forecast origins from `cleanroom.split.rolling_origins`, on a month axis.**
- What: an origin is a calendar month t. Everything up to and including t is known; the forecast covers t+1 to t+12. Origins step every 3 months. Each origin is a Cleanroom `Split` with an expanding training window (all months up to t) and a 12-month test window, and `Split.check` re-verifies the order on the rows every time (CLAUDE.md rule 2). Contestants receive only the training part; the test part is used only to look up outcomes afterwards.
- How: `rolling_origins` steps by a fixed `timedelta`, and calendar months differ in length. `src/reserve/evaluation/origins.py` therefore places cells on a "month axis" where one month is exactly one day after a fixed epoch, so a 3-day step is exactly 3 months. A test checks the round trip and the window edges.
- Which vintages: every vintage observed at the origin with at least 6 months of curve (`min_history`). Younger vintages are too short for the TSFM or the persistence baseline to say anything.
- Considered: building splits by hand. Rejected: the shared function is tested once for every project. Asking the Cleanroom for a calendar-month step is a reasonable future change, done in the Tells repo if wanted.
- Limitation: with a 3-month step and a 12-month horizon, test windows overlap, so neighbouring origins are not independent (handled in M29).

**M23. Every contestant returns seven quantiles: 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, cleaned up the same way.**
- What: the 0.25 to 0.75 range is the 50% interval and 0.05 to 0.95 the 90% interval. Every forecast is then made coherent: clipped between the last known value and 1, made non-decreasing in the horizon, and sorted so quantiles never cross.
- Why: a reserve depends on the range, not only the middle (PLAN.md FR9). A cumulative share that decreases, or a 5% quantile above the 95% quantile, is impossible, and leaving such forecasts in would penalize a contestant for an artifact rather than for its judgement.
- Considered: scoring raw outputs. Rejected for that reason; the clean-up is identical for all contestants.
- Limitation: clipping at the last known value moves mass to the lower end, which slightly favours intervals that were too low.

**M24. Contestant A, "last increment persists".**
- What: the vintage keeps going bad at its own average monthly pace over its last 3 months: c(a0 + k) = c0 + k x d. Quantiles treat the future event count as Poisson with mean k x d x loans.
- Why: the lazy guess every forecaster must beat, like "tomorrow will be like today". PLAN.md listed "seasonal naive"; vintage curves are indexed by age, so persistence of the recent pace is the natural naive here (NEXT_PROJECTS_PLAN.md, "A last change persists").
- Considered: last value (zero further increase), which is always too low for a cumulative curve; seasonal naive, which has no meaning on an age axis.
- Limitation: its interval reflects only counting noise, not uncertainty about the pace, so it is expected to be too narrow. When the last 3 months had no events, the interval collapses to a point.

**M25. Contestant B, "average age curve" (chain-ladder style, additive).**
- What: for each future age, the mean monthly increment of the 8 newest vintages that have already reached that age, added up over the horizon. The interval is normal with variance equal to the sum over ages of those vintages' increment variance. Ages no vintage has reached yet reuse the oldest age available.
- Why: this is the class-average rule lenders use, adapted from the actuarial chain ladder (which projects each cohort with development patterns of older cohorts). Additive increments are used because multiplicative development factors c(a+1)/c(a) explode when the curve is near zero at early ages.
- Considered: multiplicative development factors; more or fewer donor vintages (8 vintages is two years, a balance between recency and noise).
- Limitation: it ignores the economy and the vintage's own quality, and summing variances assumes increments are independent across ages, which understates the spread when a vintage is consistently better or worse than average. On synthetic data its 90% interval covered well below 90%, which is consistent with that (not a result; the simulator has cohort effects by design).

**M26. Contestant C, hazard regression on age plus past-only macro.**
- What: a binomial logistic regression of each cell's new events out of loans at risk on a linear age spline (knots at 6, 12, 24, 36, 60 months, plus an indicator for age 3 or less) and four macro features known at the end of that cell's month (M16): unemployment level and 12-month change, mortgage rate, and 12-month house price growth. Features are standardized on the training cells, with a small ridge penalty. The forecast holds the macro features at their values known at the origin, projects the loans at risk forward with the pooled exit rate of the last 24 months, and takes quantiles from 200 Monte Carlo draws of the coefficients (from their estimated covariance) and of binomial event and exit counts.
- Why: it is what an analyst would build, and the only classical contestant that uses the economy. The hazard uses at-risk counts, so prepayment does not bias it. Holding macro values flat is the honest choice: future unemployment is unknown at the origin, and using realized future values would leak (CLAUDE.md rule 1). Scenario paths for the economy belong to the hypothetical stress module, labelled as such (PLAN.md FR12).
- Considered: a Cox proportional hazards model (same idea, harder to project forward); vintage random effects (would capture vintage quality, deferred to keep the baseline simple and explainable); using realized future macro (rejected as a leak).
- Limitation: no vintage quality term and no seasonality, and flat macro paths miss turning points by construction, which is expected to hurt C most at the start of a recession.

**M27. One interface for every TSFM; the first real one is Chronos-Bolt-tiny, pinned, and not run yet.**
- What: a `QuantileForecaster` takes plain number sequences and returns (series, horizon, quantiles). The TSFM contestant feeds each vintage's own cumulative curve from age 1 to the origin, indexed by age only, so the model never sees a calendar date; its output gets the same clean-up as every other contestant (M23). A deterministic fake forecaster (last value plus recent pace, spread from recent steps) tests the whole path in CI. The Chronos-Bolt-tiny adapter (amazon/chronos-bolt-tiny, Apache-2.0, pinned to revision a0e552de83495b5c28c14c71c374f3e33280b340) sits behind the optional `tsfm` extra and downloads nothing unless constructed.
- Why: small enough for a CPU (Tells measured it at milliseconds per series on CPU, Tells `docs/tsfm-notes.md`), already pinned and used in the sibling project, and it outputs quantiles natively. Feeding the cumulative curve keeps the model's job identical to the other contestants'.
- Considered: feeding monthly increments and summing (quantiles of sums are not sums of quantiles); Chronos-2 and TimesFM 2.5 (planned next, one adapter each, NEXT_PROJECTS_PLAN.md).
- Limitation: the adapter is a stub. Its quantile levels outside the trained range (0.05 and 0.95), context length limits, and pretraining cutoff are not verified (`docs/tsfm-notes.md`). No TSFM result exists.

**M28. A past-only leak test for every input.**
- What: (1) for every contestant, forecasts at an origin are identical when every curve cell after the origin is multiplied by 7 and shifted by 3; (2) contestant C's forecasts are identical when every macro value published after the origin month is corrupted; (3) the vintage builder and the macro features have their own past-only tests (M11, M16).
- Why: CLAUDE.md rule 1. Corrupting the future and checking nothing moves catches leaks that a code review can miss.
- Limitation: the tests prove the code paths are past-only for the origins tested, not for every possible origin; they run on synthetic data.

**M29. Scoring with the shared Cleanroom tools, against the best classical baseline.**
- What: all losses use the change since the origin (outcome minus the known value). Reported per contestant at the 12-month horizon: mean pinball loss over the seven quantiles, weighted quantile loss on the change, the mean absolute error of the median, 50% and 90% interval coverage with Wilson intervals, and the 90% interval score; and mean pinball loss at every horizon. The reference is the classical contestant (A, B, or C) with the lowest overall loss in the same table. Each other contestant gets a relative error (geometric mean over origins of the ratio of its per-origin loss to the reference's) and a Diebold-Mariano test on per-origin losses, with the Harvey-Leybourne-Newbold correction and lags up to ceil(12 / 3) - 1 = 3 because forecasts from origins less than a year apart share outcome months. Holm's correction adjusts the p-values, and the number of comparisons is reported (CLAUDE.md rule 8).
- Why: this is the plan's headline, "quantile loss ratio against the best classical baseline at 12 months, Diebold-Mariano plus Holm" (NEXT_PROJECTS_PLAN.md). Choosing the best baseline after seeing the table can only favour the baselines, which makes any TSFM win more convincing, not less.
- Considered: averaging the ratio arithmetically (treats "twice as good" and "twice as bad" asymmetrically); pooling all vintage forecasts into one DM test (vintages at the same origin are strongly correlated, so that would overstate the evidence).
- Limitation: the Wilson coverage intervals treat forecasts as independent, which they are not; the true uncertainty is larger. Choosing the reference by its loss in the same table is a mild selection step, reported in every result.

---

## 8. Configs, runs, and the lock

**M30. One YAML config per experiment; the whole config is hashed, and every result carries provenance.**
- What: `configs/*.yaml` sets the data source, macro mode, evaluation period and origin windows, horizon, step, quantiles, excluded target windows, contestants and their settings, and seeds. `reserve prepare` builds `curves.parquet`, `loans.parquet`, and `macro_features.parquet`; `reserve run` writes `results.json` and `predictions.parquet` under `artifacts/runs/<run id>/`. Each result stores the config and its hash, the git commit and whether the tree was dirty, the SHA-256 of every input table, package versions, each contestant's settings and version, whether macro revisions were ignored, and the run's role (development, sealed test, or descriptive) (CLAUDE.md rule 12).
- Why: anyone can trace a number back to exact inputs, and any change to the plan changes the hash.
- Synthetic configs write the fake portfolio in the Freddie Mac layout and read it back through the real loader, so the synthetic check exercises the real data path. `configs/synthetic_small.yaml` runs prepare plus run in about 10 seconds on a laptop CPU.
- Considered: command-line flags instead of files. Rejected: a file can be hashed, sealed, and committed.
- Limitation: settings that live in code (for example the age-spline knots) are pinned by the git commit, not by the config hash, as in Tells (Tells L13).

**M31. Three evaluation periods, and only the test period is sealed.**
- What: `dev` (Freddie config: origins 2010-03 to 2012-03) for building and debugging; `test` (2013-03 to 2018-12, from NEXT_PROJECTS_PLAN.md) for the headline; `recession` (2007-03 to 2009-03) for the descriptive stress exam. Scoring `test` happens only inside `cleanroom.guard.sealed_run`, which requires a lock for exactly this config and data checksum (`reserve seal`), a clean git tree, and a committed lock, and records every attempt as started and finished. Synthetic configs can never be scored as `test`. Target months 2020-01 to 2021-12 are excluded from every score (M20); the headline test's last target is 2019-12 anyway.
- Why the dev window ends in 2012-03: its 12-month outcomes then end by 2013-03, the first test origin, so tuning on dev never shows a test-period outcome. The config loader refuses any config that breaks this.
- Why the recession is descriptive: there is one recession episode in the data, so it can illustrate behaviour under stress but cannot support a general claim (PLAN.md Part 7; owner decision 4). Its origins end 2009-03 so its outcomes end by 2010-03, the start of dev.
- Considered: sealing the recession too. Not done: development runs use months up to 2012 as history, so the recession is not unseen in any case; it is labelled "descriptive" in every result instead.
- Limitation: the dev period sees the recession's curves as history, so choices made on dev are informed by it. The lock fixes the plan, not the code (M30).

---

## Limitations register

The honest list. Each item says what could be wrong and what was done about it.

| # | Limitation | Effect | What was done |
|---|---|---|---|
| L1 | No real data yet | Nothing about real loans is known; all checks ran on synthetic data | Stated everywhere; `scripts/check_data_fit.py` is ready to run once the owner registers |
| L2 | Prime, conforming fixed-rate mortgages only | Results do not cover subprime, cards, or auto loans | Stated up front (M2) |
| L3 | Evaluation tools live in another repo | A Cleanroom bug would affect Reserve | Pinned to a tag; bumped only on purpose (M4) |
| L4 | Legacy (pre-Release-47) file layout partly unverified | Older downloads could be misread | Layout detected from field 31 and the field count; flagged in M5; check a real legacy file first |
| L5 | Loader tested only on hand-built fake records | Real-file quirks are unknown | `check_data_fit.py` reports the layout, counts, and reconciliation on the first real run |
| L6 | COVID forbearance shows up as delinquency in 2020 and 2021 | The event then partly measures permitted payment pauses | Forbearance share measured by the data-fit check; those months excluded from the headline (M20) |
| L7 | Vintage age is calendar-aligned, not loan-level | Loans in a quarter differ by up to two months in true age | Stated (M8); the alternative breaks the exact "known at t" property |
| L8 | Curves by original count mix credit and prepayment speed | Fast-prepaying vintages look better partly because fewer loans stayed exposed | Stated (M9); the hazard model uses at-risk counts instead |
| L9 | Events before Freddie Mac acquires a loan are not observed | A few early events could be missed | Data-fit check reports loans first seen after age 1 |
| L10 | National macro series only | Regional shocks are missed | Stated (M13); state-level series deferred |
| L11 | Latest mode leaks revisions | Forecasts could use revised values unknown at the time | As-of mode by default; latest mode sets `revisions_ignored` on every row and result (M15) |
| L12 | ALFRED vintage coverage and the "." missing marker are unverified | As-of mode may have gaps for early years | Checked by the data-fit script once the FRED key exists (M14) |
| L13 | Simulator is far simpler than real mortgages | Pipeline checks on it say nothing about real performance | Used only for tests and pipeline checks; never in findings (M17) |
| L14 | Only about 6 independent 12-month windows in the test period | Small real gains may be undetectable | MDE reported before any result (M21); negative results stay in |
| L15 | The events gate and noise-floor settings are judgment calls | A borderline dataset could pass or fail on the setting | All settings in `FitSettings`, reported in every data-fit output |
| L16 | Overlapping 12-month test windows | Neighbouring origins are correlated | Diebold-Mariano with lags up to 3; independent-window count in the data-fit report (M21, M29) |
| L17 | Contestant C holds macro flat over the horizon | It cannot foresee turning points | By design, since future macro is unknown; stress paths are a separate, hypothetical module (M26) |
| L18 | Contestant B's interval assumes independent increments | Likely too narrow when vintage quality persists | Coverage reported for every contestant (M25, M29) |
| L19 | Chronos-Bolt-tiny adapter is a stub | No TSFM result exists; quantile range and limits unverified | Fake forecaster tests the path; checks listed in `docs/tsfm-notes.md` (M27) |
| L20 | Coverage intervals assume independent forecasts | Coverage uncertainty is understated | Stated with every coverage number (M29) |
| L21 | Dev runs see the recession as history | Choices made on dev are informed by it | Recession results are labelled descriptive and never headline (M31) |
| L22 | The lock hashes the config, not the code | Code could change between seal and test | Sealed runs need a clean git tree; the commit is recorded on every result (M30, M31) |
