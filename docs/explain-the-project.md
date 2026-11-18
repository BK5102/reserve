# Explaining Reserve: the project step by step

This is the guide to read before talking about Reserve with a recruiter, a hiring manager, or an interviewer. It tells the story in the order the work happened. Every step has the same parts:

- **What I did**
- **Why**
- **What else I considered**
- **Cause and effect**: what happened because of this step, including problems it exposed
- **How to say it**: one or two sentences you can say out loud

The technical detail behind each step is in `docs/methodology.md` (numbered M1, M2, ...), and the dated diary is in `docs/decisions.md`. No result exists yet, so nothing here claims one.

## Elevator pitch (draft, no results yet)

"Lenders watch each quarter's batch of loans, called a vintage, the way a school tracks a graduating class: what share has gone seriously late by each age. I am testing whether a pretrained time series model forecasts the next year of those curves better than the methods lenders use, with honest uncertainty ranges, on public Freddie Mac mortgage data. The pipeline is built and tested on synthetic data; the real-data check is waiting on a free data registration."

---

## Part 1. Setup

### Step 1. Scaffold the repo and reuse the shared testing toolkit

- **What I did:** set up the project (uv, ruff, pytest, CI on Python 3.11 and 3.12) and installed the Cleanroom, the honest-testing toolkit built in my Tells project, from a fixed git tag.
- **Why:** the same toolkit across projects means the same exam rules everywhere, and a fix in one place fixes all of them.
- **What else I considered:** copying the toolkit into this repo. Rejected because copies drift apart.
- **Cause and effect:** the first tag (v0.1) had no forecasting tools. Version 0.2 added them (forecast scores, a statistical test for comparing forecasters, rolling forecast start dates, an as-of join), and Reserve moved to it before writing its own versions, so there are none to drift.
- **How to say it:** "I reuse one tested evaluation toolkit across my projects, pinned to a version, so every project sits the same honest exam."

## Part 2. Data

### Step 2. Read Freddie Mac's loan files exactly as documented

- **What I did:** wrote a loader for Freddie Mac's free loan-level mortgage files: one origination file (what each loan looked like when it was made) and one monthly performance file (each loan's status every month).
- **Why:** everything later depends on reading the right column as the right thing. A delinquency code read from the wrong column would silently produce wrong curves.
- **What else I considered:** letting the reader guess column types. Rejected: "03" (90 days late) and "RA" (the lender took the house back) must never be confused.
- **Cause and effect:** Freddie Mac changed the layout in July 2026 (35 performance fields instead of 32, some new names). The loader handles both, detects which one a file uses, and flags the older layout as only partly verified.
- **How to say it:** "I built the loader from the data provider's own field list and test it against their header file character by character, so a layout change fails loudly instead of quietly corrupting the data."

### Step 3. Turn loan records into vintage curves

- **What I did:** grouped loans by origination quarter (a vintage) and, for each month after the quarter ended, counted the share of the original loans that had ever been 90 or more days late or defaulted.
- **Why:** this cumulative curve is the time series every contestant forecasts. Counting each loan once, at its first bad month, makes the curve only go up.
- **What else I considered:** dividing by the loans still active each month (a life table). It answers "what if nobody prepaid" and depends on prepayment, so I kept it for the hazard model only.
- **Cause and effect:** the build checks itself every time. A duplicated row, a loan without an origination record, or a count that does not add up stops it with an error instead of producing a wrong curve.
- **How to say it:** "Each quarter's loans become one curve, the share of the original class that ever went seriously late by each month, and the build refuses to run if any count fails to reconcile with the loan-level data."

### Step 4. Load the economy as it was known at the time

- **What I did:** wrote a loader for unemployment, mortgage rates, and house prices from FRED, using its archive of earlier versions (ALFRED) so each forecast date sees only the numbers published by then.
- **Why:** unemployment and house prices get revised. A forecast that uses revised numbers has information nobody had at the time, which flatters it.
- **What else I considered:** using today's numbers with a publication delay. Kept only as a fallback that labels every row "revisions ignored".
- **Cause and effect:** a leak test deletes or corrupts everything published after each month and checks that nothing changes. Tests run on a fake recorded response, so they need no key and no network. The real download waits on a free FRED key.
- **How to say it:** "Every macro input is what was published by the forecast date, taken from the Fed's archive of past data versions, and a test proves that later revisions cannot leak in."

### Step 5. Build a fake portfolio where the right answer is known

- **What I did:** wrote a simulator that creates fake loans in the exact Freddie Mac format, from a rule I chose: risk rises with loan age to a peak, varies by vintage, has a small season, and jumps when unemployment rises in a fake recession.
- **Why:** with no real data yet, it is the only way to check that the pipeline computes what it claims. Because I wrote the rule, I know the true answer.
- **What else I considered:** waiting for the real data. Rejected: everything except the data itself can be tested now.
- **Cause and effect:** the tests show the vintage builder finds every fake loan's event month exactly and the curves match the known truth within random noise. The simulator is never used for a result.
- **How to say it:** "Before touching real data I built a simulator with a known answer and checked that my pipeline recovers it, the way you test a scale with a known weight."

### Step 6. Check whether the data can answer the question before modeling

- **What I did:** wrote a data-fit script that measures, for each quarter's loans, how many go bad by month 12 compared with random noise, whether a simple average is already as good as it gets, whether the 2007 to 2009 recession and the COVID period are covered, whether all counts reconcile, and how large an advantage the test could detect.
- **Why:** in Tells, two popular datasets turned out unable to test the idea at all, which only measurement revealed. The gate (60 usable quarters, 20 events by month 12) is agreed before seeing real data.
- **What else I considered:** going straight to models. Rejected for that reason.
- **Cause and effect:** on synthetic data the script fails the events rule, as designed for thin data, and finds the recession and forbearance windows. It also shows the test period has only about six independent years, so only a large advantage could be detected; that is stated before any result.
- **How to say it:** "Before building models I measured whether the data could even answer the question, including how big an effect the test could detect, so a null result would not be over-read."

## Part 3. Contestants and the honest exam

### Step 7. Build the contestants and the forecasting harness

- **What I did:** built three classical forecasters (A: the vintage keeps its recent pace; B: it follows the average path of recent older vintages at the same ages; C: a hazard regression on loan age and the economy as known at the time), one interface for pretrained time series models with a fake model for testing, and a harness that runs all of them on the same forecast dates.
- **Why:** a pretrained model only matters if it beats the methods a lender already uses, under identical inputs.
- **What else I considered:** using the actual future economy in C. Rejected: nobody knows next year's unemployment at the forecast date; that would be a leak.
- **Cause and effect:** each forecast date is a split from the shared toolkit, a test corrupts everything after each date and checks no forecast changes, and the scores (quantile loss, interval coverage, a statistical test with a correction for several comparisons) come from the shared toolkit. On synthetic data B's 90% range covered too few outcomes, which shows why coverage is reported for everyone.
- **How to say it:** "Every model gets exactly the data known on the forecast date, and I prove it by corrupting the future and checking nothing moves; the pretrained model has to beat the best classical method, chosen after the fact, which can only make the bar higher."
