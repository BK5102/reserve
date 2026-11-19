# Explaining Reserve: the project step by step

This is the guide to read before talking about Reserve with a recruiter, a hiring manager, or an interviewer. It tells the story in the order the work happened. Every step has the same parts:

- **What I did**
- **Why**
- **What else I considered**
- **Cause and effect**: what happened because of this step, including problems it exposed
- **How to say it**: one or two sentences you can say out loud

The technical detail behind each step is in `docs/methodology.md` (numbered M1, M2, ...), and the dated diary is in `docs/decisions.md`. No result exists yet, so nothing here claims one.

## Elevator pitch (draft, no results yet)

"Lenders watch each quarter's batch of loans, called a vintage, the way a school tracks a graduating class: what share has ever gone seriously late by each month of age. I am testing whether a pretrained time series model forecasts the next year of those curves better than the methods lenders use, with honest uncertainty ranges, on public Freddie Mac mortgage data. The pipeline, the classical baselines, and the honesty checks are built and tested against a simulator with known answers; the real-data check is waiting on a free data registration, and I have no result to claim yet."

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

### Step 8. Make every run reproducible and seal the exam

- **What I did:** added config files, a command line (`reserve prepare`, `reserve run`, `reserve seal`), and result files that record the config hash, git commit, data checksums, and package versions.
- **Why:** any number in a report must trace back to exact inputs, and the test period must be scored under a plan fixed in advance.
- **What else I considered:** command-line flags instead of config files. Rejected: a file can be hashed and sealed.
- **Cause and effect:** while setting the windows I found that a development period ending in 2012-12 would have shown outcomes from 2013, the test period. The development window now ends in 2012-03, and the config loader rejects any overlap.
- **How to say it:** "The test plan is hashed and committed before any test score exists, the code refuses to score the test without it, and my development window ends early enough that I never see a test answer while tuning."

## Part 4. Pretrained models

### Step 9. Make the first pretrained model real

- **What I did:** connected Chronos-Bolt-tiny, a small pretrained forecaster from Amazon, to the harness, checked what it actually outputs against its own code and settings file, and ran it on the synthetic portfolio.
- **Why:** the earlier adapter was a stub, and a contestant that has never run cannot be trusted on real data.
- **What else I considered:** letting the library fill in the uncertainty levels it was not trained on, or inventing tails for them. I map the levels in my own code instead and hold the outer ones flat, so nothing is added that the model did not say.
- **Cause and effect:** the model is trained only on the 10% to 90% levels, so the range I call "90%" is really its 80% range, and its 90% coverage will look low for that reason alone. That is now stated wherever its coverage appears. It costs about 3 milliseconds per vintage forecast on my laptop CPU (synthetic data), so speed is not a constraint.
- **How to say it:** "Before trusting the pretrained model I checked which uncertainty levels it was actually trained on; it stops at 10% and 90%, so I report its outer range honestly instead of stretching it."

### Step 10. Line up the next two pretrained models without pretending they ran

- **What I did:** added Chronos-2 (Amazon, larger) and TimesFM 2.5 (Google) behind the same interface as stubs that refuse to run, with every fact about them marked as checked or not checked, and where.
- **Why:** the plan asks for three pretrained models, and the facts that decide whether a comparison is fair (uncertainty levels, context length, training data dates, license) must be checked at the source before any code relies on them.
- **What else I considered:** writing the full code from the documentation now. Rejected because code that has never run is untested, and TimesFM's install route is not clear yet.
- **Cause and effect:** checking showed Chronos-2 is trained on 21 uncertainty levels including all of mine, so its 90% range needs no shortcut, while TimesFM 2.5 has Chronos-Bolt's 10% to 90% limit and was trained on some data from as late as 2023, after my test period, which is a contamination risk to state if it is ever run.
- **How to say it:** "I verify each pretrained model's settings at the source before using it, and the two I have not run yet are clearly marked as stubs rather than quietly counted."

## Part 5. Experiments on the development period

### Step 11. Test what happens when the portfolio is new

- **What I did:** added a thin-history option that pretends the loan product started only 6, 12, or 24 months before each forecast date, by removing every older vintage, and scores every contestant on the same 6-month-old vintages at each level.
- **Why:** the main reason a pretrained model might help is that a new product has little history of its own, so the classical methods have little to learn from.
- **What else I considered:** shortening every vintage's own history instead, which answers a different question (what if old records were deleted), and letting the forecast targets change with the history length, which would mix two effects.
- **Cause and effect:** by design, the methods that look only at the vintage's own curve (A and the pretrained model) score the same at every level, while B and C, which learn from older vintages, change. On SYNTHETIC data (a pipeline check, not a result) C got much worse with 6 or 12 months, because it had to guess the shape of the curve at ages it had never seen. Corrections for multiple comparisons now run across all levels together, and it runs on the development period only.
- **How to say it:** "I simulate a brand-new loan product by hiding all older vintages, and keep the forecast targets fixed so the only thing that changes is how much history each method can learn from."

### Step 12. Add hypothetical economic scenarios, labelled as such

- **What I did:** added baseline, adverse, and severe what-if paths for unemployment and house prices, fed them to the macro regression (C), and report how its forecasts move. Everything is labelled HYPOTHETICAL and nothing is scored.
- **Why:** reviewers ask how a model behaves in a downturn. A scenario is a crash test, not a prediction, and it cannot be scored because the economy never followed it.
- **What else I considered:** calibrating to official supervisory scenarios (not consulted, so not claimed), and building exact paths from the real unemployment history (more complex for little gain at this stage).
- **Cause and effect:** only C responds, because the other contestants, including the zero-shot pretrained model, do not use the economy at all. That is a finding about the design worth stating: in a downturn, the pretrained model as used here cannot react to a forecast of rising unemployment.
- **How to say it:** "My stress scenarios are labelled hypothetical everywhere and never scored, and they showed that only the regression with economic inputs reacts to them, which is exactly the gap a pretrained model with covariates would have to close."

## Part 6. Monitoring

### Step 13. Add a drift monitor and test it on drift I planted

- **What I did:** built a monitor that rates three warning lights (are the 90% ranges still catching 90% of outcomes, are errors growing compared with the backtest, and is the economy outside what the model was trained on) green, amber, or red, and a `reserve monitor` command that applies it to any finished run.
- **Why:** a model is not finished when it is built; reviewers expect ongoing checks that it still works, and the plan asks for a simple monitor with thresholds.
- **What else I considered:** the population stability index, a standard credit-scoring drift score, which needs many items per period while a portfolio adds only a few vintages a year.
- **Cause and effect:** on fake forecasts that are right by construction it stays non-red, and when I shift the outcomes from a known date every light turns red within a year of origins. One honest weakness: the outcome of a 12-month forecast arrives 12 months later, so drift is always seen late.
- **How to say it:** "I test the monitor the way you test a smoke alarm: I plant the drift at a known date and check that it goes off, and that it stays quiet when nothing is wrong."

## Part 7. The agent

### Step 14. Build the agent around four buttons, a referee, and a report written by code

- **What I did:** built the agent layer as in Tells: a tool layer with exactly four tools (run an experiment, load results, chart data, statistics at another horizon), a runner that executes only those, a critic that runs fixed checks before asking a model for the three strongest objections to each claim, and a reporter that writes an eight-section evidence pack with every number inserted by code.
- **Why:** an agent that can only press labelled buttons cannot leak the test or invent numbers, and a validation-style pack is only useful if a reviewer can trace each number to a file.
- **What else I considered:** letting the agent write the pack, or decide on its own to spend the sealed test. Both rejected: numbers come from files, and spending the test is a person's decision, behind a flag and a committed lock.
- **Cause and effect:** the tests replay hand-written model conversations, so the whole loop runs with no API key and no cost. They show the agent being refused when it asks for a file or the test period, the critic failing an answer that types a number or claims a win the statistics do not support, and the challenger's objections being rejected unless each points to a value that exists in a results file. No live model call has been made yet.
- **How to say it:** "The agent can only use four tools, a referee checks its answer with fixed rules before a second model argues against it, and every number in the final report is inserted by code from the result files."

## Part 8. Getting ready for real data

### Step 15. Label fake data by what is in it, not where it sits

- **What I did:** made every loan the simulator writes carry two markers (seller "SYNTHETIC", state "ZZ"), and made the pipeline and the data-fit script label any result SYNTHETIC if a single record carries one, even when the files sit in the real-data folder.
- **Why:** before the real files arrive, I rehearse the owner's exact commands on fake files placed where the real ones will go. The first rehearsal printed a normal-looking result with no SYNTHETIC label, because the label came from the config's name for the data source, not from the data.
- **What else I considered:** labelling by folder or file name. Rejected: copying fake files into the real folder would defeat it.
- **Cause and effect:** a fake number can no longer look real by accident. The data-fit script also now stops with a non-zero exit code when the gate fails, so the go/no-go step is a command, not a line someone has to notice.
- **How to say it:** "My rehearsal caught that fake data in the real folder produced unlabelled results, so now the label travels inside every record and nothing built from them can lose it."
