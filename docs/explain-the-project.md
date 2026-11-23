# Explaining Reserve: the project step by step

This is the guide to read before talking about Reserve with a recruiter, a hiring manager, or an interviewer. It tells the story in the order the work happened. Every step has the same parts:

- **What I did**
- **Why**
- **What else I considered**
- **Cause and effect**: what happened because of this step, including problems it exposed
- **How to say it**: one or two sentences you can say out loud

The technical detail behind each step is in `docs/methodology.md` (numbered M1, M2, ...), and the dated diary is in `docs/decisions.md`. No result exists yet, so nothing here claims one.

## Elevator pitch (draft, no results yet)

"Lenders watch each quarter's batch of loans, called a vintage, the way a school tracks a graduating class: what share has ever gone seriously late by each month of age. I am testing whether a pretrained time series model forecasts the next year of those curves better than the methods lenders use, with honest uncertainty ranges, on public Freddie Mac mortgage data. The pipeline, the classical baselines, three pretrained models, and the honesty checks are built and tested against a simulator with known answers, and every real-data command has been rehearsed end to end; the real run is waiting on the data download, and I have no result to claim yet."

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

### Step 16. Give the sealed test its own config and make sealing refuse anything unready

- **What I did:** added `configs/freddie_test.yaml` (the development config with the period set to the 2013 to 2018 test), a `reserve checksums` command that records the SHA-256 of every raw download, and a list of checks that `reserve seal` must pass first: real Freddie Mac files, every file checksummed and unchanged, tables built from those exact files, all three classical baselines, and only real, installed pretrained models.
- **Why:** the rehearsal showed the documented path could never work. I had told myself to seal the development config and then change its period to "test", but the period is part of the sealed plan, so the edited file no longer matched its lock. Separately, a sealed run that crashes still counts as an attempt, so sealing a plan whose model is not installed would waste the one honest test.
- **What else I considered:** a `--period test` flag on one config. Rejected for the same reason: it changes the plan's fingerprint.
- **Cause and effect:** a test checks the two configs differ only in name and period, so what I tune on is what I seal. Sealing now fails with a list of every problem at once instead of producing a lock that cannot be used. Errors from the guard now print as messages instead of Python tracebacks.
- **How to say it:** "My dress rehearsal found that my own instructions for sealing the test were a dead end, so the test now has its own config, and the seal command refuses until the data is real, fingerprinted, and every model in the plan can actually run."

### Step 17. Rehearse the owner's exact commands, every time the code changes

- **What I did:** wrote a script that builds a throwaway copy of the project with simulator files where the real Freddie Mac files will go, then runs every command I will type on real data, in order, as separate programs: check the setup, prepare, check the data fit, development and recession runs, record checksums, seal, commit the lock, the one sealed run, and a deliberately changed plan that must be refused. CI runs it on every push. I also added `reserve doctor`, which checks the download folder and the API keys first, and a way to replay saved FRED responses so the macro download path can be rehearsed without a key.
- **Why:** each function had tests, but a sequence of commands typed by a person can still fail. The rehearsal is the only test of the path as it will actually be used.
- **What else I considered:** running the rehearsal inside the test suite (it would double the test time) and inside the repo itself (it would leave fake files and a fake lock where the real ones go).
- **Cause and effect:** the first rehearsal found the three faults described in Steps 15 and 16 (unlabelled fake results, a sealing path that could never work, and tracebacks instead of messages). Now any change that breaks the real path fails CI before real data exists. What it cannot test is anything only real files have, such as their size or unexpected codes.
- **How to say it:** "Before the real data arrived I ran my exact runbook end to end on simulated files in a throwaway repo, including sealing and the one test run; it caught three bugs, and it now runs in CI so the real path cannot quietly break."

### Step 18. Make the second pretrained model real

- **What I did:** turned the Chronos-2 stub into a working adapter, checked at load time that it reports the 21 uncertainty levels it was trained on, and ran it on the synthetic portfolio next to Chronos-Bolt-tiny.
- **Why:** the plan asks for three pretrained models, and Chronos-2 is the only one trained on the 5% and 95% levels, so its 90% range is genuinely a 90% range rather than a stretched 80% one.
- **What else I considered:** letting it share information across all the vintages in a batch (its cross-learning option). Rejected: a vintage's forecast would then depend on which other vintages happened to be in the batch.
- **Cause and effect:** it costs about 18 milliseconds per vintage forecast on my laptop's CPU, ten times the tiny model and still negligible for the real test. It has only run on synthetic data, and its training data cutoff is undisclosed, which I state with any result.
- **How to say it:** "My second pretrained model is the only one trained on the exact uncertainty levels I score, so its 90% range is honest by construction, and I checked its output against the library's own before trusting it."

### Step 19. Make the third pretrained model real, installed from its official source

- **What I did:** added TimesFM 2.5 (Google) as an optional install taken straight from Google's code repository at a fixed version, tried it first in a separate throwaway environment, and once it ran there on my laptop's CPU, replaced the stub with a working adapter and tests.
- **Why:** the plan asks for three pretrained models from more than one lab, and the model's own page was unclear about which published package contains this version, so installing the exact tagged commit from the official source removes the guesswork.
- **What else I considered:** the published package (probably identical, but the model page points to the source), the newer TimesFM 3.0 (its weights are licensed for non-commercial use only), and leaving it as a stub.
- **Cause and effect:** it works, but it is by far the slowest model, so I set its batch size to match how many vintages each forecast date has, because the library pads every batch. Two parts of its training data come from after my test period, so any TimesFM result carries a contamination warning. Whether it enters the sealed test is the owner's one-line choice before sealing.
- **How to say it:** "I verified the third model in a throwaway environment before letting it near the project, pinned it to an exact commit of the official code, and I state up front that part of its training data postdates my test period."

### Step 20. Put a hard spend cap on the agent and record its first live run

- **What I did:** added a settings file for the agent with a 10-dollar cap per run, code that checks before every model call what the worst case of that call could cost and shrinks or stops the call to stay under the cap, and a `--record` option that saves the live conversation so it can be replayed later for free.
- **Why:** an agent in a loop can keep calling the model, and the bill is real money. A cap that only adds up the cost after each call can be broken by the one call that crosses it; checking the worst case first cannot.
- **What else I considered:** counting tokens exactly with an extra request before each call (more calls for little gain; my estimate deliberately overcounts), and ignoring the fallback model (it can cost more than the main one, so I price it at the highest listed rate).
- **Cause and effect:** the first live run will produce a real recording, replacing my hand-written test conversations as the evidence that the request shape is right. The spend figure is an estimate from listed prices, so the account's own limit stays as a second guard. No live call has been made yet.
- **How to say it:** "My agent cannot overspend: before every call it prices the worst case, including a fallback to a pricier model, and stops before the cap rather than after it, and its first live run is recorded so every later test replays it for free."

### Step 21. Write the runbook for the real run

- **What I did:** wrote `docs/runbook.md`, the exact commands in order from "the data is downloaded" to "the sealed result is written up": check the download, the data-fit gate, the macro download, development runs, the rule that picks the reference baseline, checksums, sealing, committing the lock, the one sealed run, a summary table printed from the results file, the agent's evidence pack, and the findings.
- **Why:** the real run happens once and cannot be redone quietly, so every step, and what each step must print, is written down and rehearsed in advance.
- **What else I considered:** leaving the steps scattered across the README and the docs, which is how the dead-end sealing instruction survived until the rehearsal.
- **Cause and effect:** writing it found one more fault: the rule meant to let small result files be committed never worked, so the sealed results could not have been kept in the repo; it is fixed. I also added `reserve summary`, which prints the headline table from a results file with its run id and config hash, so the findings can quote numbers without retyping them.
- **How to say it:** "The real run is a checklist I rehearsed in CI before the data existed, so on the day it is a sequence of commands, not a series of decisions."

### Step 22. Make the data loader fit in a laptop's memory

- **What I did:** the real Freddie Mac sample arrived (all 28 years, 1999 to 2026, about 9 GB of text), and the first data-fit check was stopped for running out of memory. I changed the loader to shrink each year to one row per loan as soon as it is read, instead of stacking every month of every loan first.
- **Why:** the checks only need a few facts per loan (when it was first and last seen, when it first went 90 days late), and every loan lives in one year's file, so the monthly rows can be dropped year by year.
- **What else I considered:** loading only some years (a choice I would have to defend), or a bigger machine (the project must run on a laptop CPU).
- **Cause and effect:** the largest year peaks at about 2.6 GB while being read and shrinks to about 2 MB. A test builds the tables both ways and finds them identical, so no number changes. It is also the first fault that only the real data could reveal: the rehearsal's small fake files never tested size.
- **How to say it:** "The real data was too big to hold at once, so I summarize each year to one row per loan as it loads, and a test proves the summary gives exactly the same curves."

### Step 23. The real data passes its check, and three choices are made before any model runs

- **What I did:** the data-fit gate passed on the real sample: 102 usable vintages (60 needed) and a median of 38.5 defaults by age 12 (20 needed), with all 1,375,000 loans reconciled. Then I made three choices, each before any model had touched the real data:
  - Macro data: use archived as-known values where they exist, and flagged current values only before the archive starts.
  - Include TimesFM 2.5 as a third pretrained model.
  - Approve the plan's defaults.
- **Why:** the macro download showed the archive starts in 2010 for mortgage rates and 2012 for house prices. Strictly as-known data would have left the macro baseline unable to forecast from the development and recession origins. Settling all three choices now means none can be tuned toward a test result later.
- **What else I considered:**
  - Using unemployment only, which is cleaner but drops the two mortgage-specific features and changes the design after looking at the data.
  - Using current values everywhere, which ignores revisions even where the archive exists.
  - Leaving TimesFM out, because it is slower and some of its training data overlaps the test years.
- **Cause and effect:**
  - Every test origin still uses only as-known macro values.
  - The flagged fallback affects only older training months. I measured its cost: mortgage rates were never revised, and house-price growth can be off by about 0.4 of a typical swing before 2012.
  - The test now compares three pretrained models from two labs.
- **How to say it:** "Where the historical record of what was known exists, I use it. Where it does not, I use today's value and flag it, I measured how much that could matter, and I made the choice before any model saw the data."

### Step 24. Development runs on the real data, then freezing the plan

- **What I did:** ran every contestant on the development period, the 2007-2009 recession (descriptive only), the thin-history experiment and the hypothetical scenarios, all on the real Freddie Mac data. Then I wrote down the development winner before sealing (`docs/decisions.md` D29).
- **Why:** development runs catch broken plumbing on real data, and writing down the practice result first makes any change on the test visible.
- **What else I considered:** tuning settings after these runs. I did not, because that would fit the plan to what I had seen.
- **Cause and effect:**
  - **Point forecasts:** on development, all three pretrained models matched or beat the best simple baseline (`A_last_increment`, the reference).
  - **Intervals:** the pretrained models' 90% intervals covered 82 to 88% of outcomes, against 31 to 42% for the classical methods.
  - **Recession:** coverage dropped to 39 to 67% and the edge mostly disappeared. The sealed test checks whether the development edge holds on 2013 to 2018.
- **How to say it:** "On practice data the pretrained models matched the best simple baseline and had far more honest intervals, but not in the 2008 recession, and I wrote that down before sealing so the test result could not be spun."

### Step 25. The sealed test, once

- **What I did:** sealed the plan with the raw-file checksums, committed the lock, and ran the 2013 to 2018 test once (run `20261123T022625Z_freddie_test_test_d6120cf1`).
- **Why:** a result only counts if the plan existed before it, and one run means no picking the best of several tries.
- **What else I considered:** running the optional agent write-up, which needs paid API calls. I skipped it to spend nothing.
- **Cause and effect:**
  - **Accuracy:** no pretrained model was significantly more accurate than the simple "last change continues" reference. With about six independent years of outcomes, a small edge cannot be told from zero.
  - **Intervals:** their 90% ranges were honest (83 to 89% coverage). The classical methods' ranges covered only about half the outcomes.
  - **Macro model:** the one model that reads the economy did worst.
- **How to say it:** "On sealed Freddie Mac data the pretrained forecasters were no more accurate than a simple baseline, but they knew how uncertain they were far better, and I report both, along with the catch that the baselines' ranges were never recalibrated."
