# Data fit check

Reserve needs two things from a dataset before any model is built. First, enough vintages with enough events that a vintage curve is more than noise: if each quarter has only a handful of loans going bad by age 12, the curve is mostly luck and no model can forecast it better than a simple average. Second, enough history to cover a recession and enough later origins for an honest test period. Analogy: you cannot judge a weather forecaster in a town where it rains twice a year.

The checks are in `scripts/check_data_fit.py` (logic in `src/reserve/datafit.py`; reasoning in `docs/methodology.md` M19 to M21).

## Status

**Real-data numbers: PENDING.** The owner has not yet registered for the Freddie Mac Single-Family Loan-Level Dataset or obtained a FRED API key, so no real file has been read. Every number in the "Synthetic run" section below comes from fake data and says nothing about Freddie Mac loans.

To fill in the real numbers (see README, "Manual steps"):

```
uv run python scripts/check_data_fit.py freddie data/raw/freddie --fred data/raw/fred --out artifacts/data_fit_freddie.json
```

## Gate (agreed in NEXT_PROJECTS_PLAN.md)

At least 60 usable vintages, and at least 20 events per vintage by age 12. "Usable" means observed through at least age 24 (12 months of history plus a 12-month forecast). The events rule is applied to the median usable vintage's cumulative events by age 12 (methodology M19), and the share of vintages meeting it is reported too.

One piece of arithmetic, not a measurement: the sample holds 50,000 loans per full vintage year (User Guide), about 12,500 per quarter. Twenty events by age 12 then needs a 12-month cumulative rate of at least 20 / 12,500 = 0.16%. Whether real vintages reach that is exactly what the check will measure.

## Real-data results

| Check | Result |
|---|---|
| Files found, layout detected per file | PENDING |
| Last performance month (Release 48 notes say 2026-06-30) | PENDING |
| Vintages and loans per quarter | PENDING |
| Cumulative events by age 12 (median over usable vintages), relative binomial standard error | PENDING |
| Class-average 12-month error divided by binomial noise | PENDING |
| Recession (2007-12 to 2009-06): vintages observed, events | PENDING |
| COVID window (2020-03 to 2021-12): events, share flagged forbearance | PENDING |
| Reconciliation: loans without performance, first seen after age 1, vanishing without a code | PENDING |
| Power: test origins, independent windows, minimum detectable effect | PENDING |
| Macro: first observation, number of ALFRED vintages, first vintage date, missing values | PENDING |
| Gate | PENDING |

## Synthetic run (proves the checks work; NOT real data)

Command: `uv run python scripts/check_data_fit.py synthetic configs/synthetic_fit.yaml --out artifacts/data_fit_synthetic.json`, run 2026-11-17. The simulator (methodology M17, M18) made 92 fake vintages (1999Q1 to 2021Q4) of 600 loans each, a fake recession peaking in 2009-10, and a fake forbearance window from 2020-04 to 2021-06.

| Check | Synthetic result |
|---|---|
| Last performance month | 2022-06-01 |
| Vintages, loans per quarter | 92, median 600 |
| Cumulative events by age 12 (median over usable vintages) | 1; relative binomial standard error 0.999 |
| Class-average 12-month error / binomial noise | 1.53 |
| Recession window | 41 vintages observed, 138 events |
| COVID window | 285 events, 71.6% flagged forbearance |
| Power | 24 quarterly test origins with targets, 6 independent 12-month windows, minimum detectable effect 0.66 to 1.32 standard deviations |
| Gate | 86 usable vintages (need 60), median 1 event by age 12 (need 20): FAIL |

What this shows about the checks, not about real data: with 600 loans per quarter and a low early hazard, the events rule fails, which is the behaviour the gate exists for; the recession and forbearance windows are found; and the forbearance flag is picked up from the Borrower Assistance Plan field. The power numbers depend only on the test window and the data end, so they will be the same on real data whose last month is after 2019-12: with only 6 non-overlapping 12-month windows, only a large, consistent advantage could be detected.
