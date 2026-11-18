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

## Limitations register

The honest list. Each item says what could be wrong and what was done about it.

| # | Limitation | Effect | What was done |
|---|---|---|---|
| L1 | No real data yet | Nothing about real loans is known; all checks ran on synthetic data | Stated everywhere; `scripts/check_data_fit.py` is ready to run once the owner registers |
| L2 | Prime, conforming fixed-rate mortgages only | Results do not cover subprime, cards, or auto loans | Stated up front (M2) |
| L3 | Cleanroom v0.1 lacks forecast tools | Local stand-ins could differ from the shared versions | Marked `TODO(cleanroom-v0.2)`; switch when the tag exists (M30) |
