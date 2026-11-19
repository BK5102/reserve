# Decisions

One entry per decision or feature: what was decided, why, and an analogy. These are the owner's talking points. Newest at the bottom. The full reasoning for each choice is in `docs/methodology.md`.

## Scaffold (2026-11-17)

### D1. Repo scaffold with the shared Cleanroom installed from its git tag

Reserve is a uv project with ruff, pytest, and CI on Python 3.11 and 3.12, and it installs the Cleanroom testing toolkit from the Tells repo at the tag `cleanroom-v0.1` instead of keeping its own copy. One shared copy means a leak bug fixed once is fixed for every project, and the claim that all projects sit the same honest exam stays true. Analogy: three schools using the same sealed exam board instead of each photocopying and editing the rulebook.

### D2. CPU only

Every run, including the full one, must work on a laptop CPU, so the PLAN.md line allowing a GPU for the full run was removed and CLAUDE.md gained a CPU-only rule. Tells showed that a step needing a GPU ends up resting on one hard-to-repeat run. Analogy: plan the trip for the car you own, not one you might borrow.

## Data (2026-11-17)

### D3. Freddie Mac loader built from the official file layout, tested on fake records

The loader reads the pipe-delimited origination and monthly performance files using the field order in Freddie Mac's official header files (31 and 35 fields since Release 47, July 2026), and a test compares the code's field list with that header text exactly. No real data is available yet, so the loader is tested on a few hand-built fake loans written in the exact layout: a prepayment, a loan that goes 90 days late and ends in foreclosure, and a month with status "not available". Analogy: checking a key against the locksmith's blueprint before the door arrives.

### D4. Vintage curves by original loan count, reconciled on every build

Each origination quarter becomes a curve: the share of its original loans that have ever been 90+ days late or defaulted, by month since the quarter ended, with prepaid loans kept in the denominator so the curve never goes down. Every build checks that each loan is counted once, that the events in the curves equal the loans that went bad, and that the "loans still at risk" count adds up month to month, and a test proves that curves built from data up to any month match the full curves cut at that month. Analogy: a school's yearbook counting, for each graduating class, how many have ever dropped out by each year, where students who transferred away still count as members of their class.

### D5. Macro series as they were known at the time

The macro loader fetches unemployment, the 30-year mortgage rate, and a house price index from FRED, including every past version of each series from the ALFRED archive, and gives each forecast month only the numbers that had been published by the end of that month. If only today's numbers are available, it uses fixed publication delays and labels every row "revisions ignored". Tests replay a fake recorded response file, so they never call the network or need a key, and a leak test proves that deleting or corrupting anything published later does not change any feature. Analogy: judging an investor on the newspaper they could read that day, not on the corrected edition printed a month later.

### D6. A synthetic portfolio with known answers

The simulator writes fake loans in the exact Freddie Mac layout from a known rule (an age curve, a vintage quality effect, a seasonal wave, and unemployment with a recession shock), so the true answer is known and tests can check that the pipeline finds it. It is used only to test the plumbing and never for a result. Analogy: a pilot practising in a flight simulator where the instructor knows exactly where the wind will come from.

### D7. A data-fit check before any model

`scripts/check_data_fit.py` measures whether the data can answer the question: loans and events per vintage against random noise, whether a simple class average is already as good as possible, recession and COVID coverage, reconciliation counts, and how big an advantage the test could detect. It runs on synthetic data now, where it correctly fails the events rule, and is ready for the real files the day they are downloaded; `docs/data-fit.md` marks every real number as pending. Analogy: checking that a thermometer can read the range you care about before using it to judge a fever.

### D8. Cleanroom bumped to v0.2 before building the forecasting harness

The shared Cleanroom gained forecast scores, the Diebold-Mariano test, rolling forecast origins, an as-of join, and a sealed-run guard in `cleanroom-v0.2`, so Reserve moved to that tag and its macro loader now uses the shared as-of join instead of its own filter. Bumping before writing the harness meant no local copy of those tools was ever written. Analogy: waiting for the shared lab's new instrument instead of building a home-made one that would later disagree with it.

### D9. Three classical contestants, one TSFM interface, scored with the shared tools

Contestant A assumes each vintage keeps going bad at its own recent pace, B gives it the average path that recent older vintages took at the same ages, and C is a hazard regression on loan age and the economy as it was known at the forecast date. A TSFM plugs in through one interface that sees each vintage's curve indexed by age only, with a fake model for tests and a pinned Chronos-Bolt-tiny adapter that is not yet run. Every forecast is cut with the shared Cleanroom splits, a test corrupts everything after each forecast date and checks that no forecast moves, and scores come from the shared forecast tools with a Diebold-Mariano test and Holm's correction. Analogy: a race with a walker, a pacer who copies last year's runners, and a coach who also checks the weather, before the new runner is allowed on the track.

### D10. Configs, a CLI, and a sealed test period

`reserve prepare` builds the vintage and macro tables from a config and `reserve run` scores one evaluation period and writes results with full provenance; the synthetic config runs end to end in about 10 seconds. The headline test period can only be scored after `reserve seal` has locked the config and data checksum and the lock is committed, and every attempt is counted. The development window ends a full year before the test window starts, so no development outcome is a test answer. Analogy: the exam paper is sealed in an envelope with a signature across the flap, and every opening is written in a log.

### D11. Docs that a reviewer can check, and an empty findings file

The README gives a synthetic quickstart and the two manual steps that block real data (Freddie Mac registration and a FRED API key in an environment variable), `docs/findings.md` is an explicit placeholder with no numbers, and every real-data entry in `docs/data-fit.md` says PENDING. Writing down what is not known yet is as important as what is built, because a portfolio project is judged on whether its claims can be traced. Analogy: a lab notebook where blank pages are labelled "not yet measured" instead of being filled in from memory.

## Pretrained models (2026-11-18)

### D12. Chronos-Bolt-tiny is real, with its quantile levels mapped in Reserve's own code

The adapter now loads the pinned model, checks that it was trained on the nine levels 0.1 to 0.9, and maps those to Reserve's seven levels by reading between them, holding 0.05 and 0.95 at the 0.1 and 0.9 values instead of inventing tails. It ran end to end on the synthetic portfolio at about 3 ms of wall time per vintage forecast on a laptop CPU, and tests that run only when the optional extra is installed check its output against the library's own. Analogy: a ruler marked from 1 to 9 cm lets you read 2.5 cm between marks, but you write "1 cm or less" rather than guessing 0.5 cm.

### D13. Chronos-2 and TimesFM 2.5 as honest stubs

Both models sit behind the same quantile interface with their ids, licenses, sizes, quantile levels, and training data written down from their official pages and code, each fact marked verified or unverified, and both refuse to run. Chronos-2 turned out to be trained on 21 quantile levels that include all of Reserve's, while TimesFM 2.5 stops at 10% and 90% and lists training data up to 2023, after the test period. Analogy: two new players are on the team sheet with their measurements checked, but they are marked "not yet played" instead of being given made-up scores.

### D14. A thin-history experiment that simulates a new loan product

`reserve run CONFIG --experiment thin_history` hides every vintage older than 6, 12, or 24 months at each forecast date and scores every contestant on the same 6-month-old vintages, with Holm's correction across all levels together, on the development period only. Methods that read only the vintage's own curve do not change across levels, while the ones that learn from older vintages do, which is exactly what the experiment is meant to expose. Analogy: asking how well each coach predicts a first-term student's grades at a school that opened six months, one year, or two years ago.

### D15. Hypothetical macro scenarios, never scored

Baseline, adverse, and severe what-if paths for unemployment and house prices feed the macro regression, and the result reports how far its forecasts move and which contestants respond at all; everything carries the label HYPOTHETICAL and runs on the development period only. Only the regression responds, because the other contestants do not use the economy. Analogy: a crash test shows how a car behaves in a crash, not whether you will have one.

## Monitoring (2026-11-18)

### D16. A three-light drift monitor

`reserve monitor` rates each later forecast origin on 90% interval coverage, the loss compared with a backtest window, and (as a function) whether the macro inputs are outside their training range, each green, amber, or red with fixed thresholds. It was tested by planting a shift in synthetic outcomes at a known date and checking that every light turned red within a year of origins while staying quiet without drift. Analogy: testing a smoke alarm by lighting a match under it, and checking it stays silent when nobody is cooking.

## The agent (2026-11-18)

### D17. An agent with four tools, a two-stage critic, and an evidence pack written by code

`reserve agent "question"` lets Claude Opus 5.5 plan and call four tools (run an experiment, load results, chart data, statistics at another horizon), cannot score the test period unless a person passes `--allow-test` and a lock is committed, and hands its answer to a critic that checks numbers, labels, split order, the lock, the correction, and claim directions before a second pass lists three objections per claim, each pointing to a value in a results file. The reporter then writes an eight-section evidence pack, labelled "inspired by validation practice, not a compliance document", with every number inserted by code. Analogy: an intern who may use four named instruments, whose lab report is checked by a referee and argued against by a rival, with the numbers copied in by the machine rather than typed by hand.

## Getting ready for real data (2026-11-18)

### D18. The SYNTHETIC label follows the data, not the folder

A dress rehearsal of the real-data commands on simulator files in the Freddie Mac layout showed that those runs carried no SYNTHETIC label, because the label came from the config. Now every simulated record carries markers, `reserve prepare` writes a manifest saying whether any record is marked, and every run and data-fit report built from marked records is labelled SYNTHETIC. Analogy: a practice exam paper is stamped PRACTICE on every page, not only on the folder it came in.

### D19. A separate sealed test config, and a seal that checks before it locks

The test period is scored from `configs/freddie_test.yaml`, identical to the development config except its name and period, because a lock fixes the whole config and editing the period afterwards breaks the match. `reserve checksums` writes `data/checksums.sha256`, and `reserve seal` refuses until the data is real, every raw file is checksummed and unchanged, and every contestant is a real, installed model. Analogy: an exam board will not seal the envelope until the paper is the final version, every page is numbered, and the exam room is booked.

### D20. A dress rehearsal of the real-data commands, run in CI

`scripts/rehearse_real_path.py` puts simulator files where the Freddie Mac download will go, inside a throwaway git repo, and runs every command of the runbook in order, including sealing, committing the lock, the sealed run, and a refused change of plan; CI runs it on every push. Its first run found three faults that unit tests had missed, and `reserve doctor` now checks the download folder (unzipped, flat, paired) and the keys before anything runs. Analogy: a fire drill through the real exits, not a check that each door opens on its own.

### D21. Chronos-2 is a real contestant

The Chronos-2 adapter now loads the pinned model, checks it reports its 21 trained quantile levels, and keeps Reserve's seven, all of which it was trained on, so its 90% range needs no shortcut; tests that run only with the `tsfm` extra compare it with the library's own answer, with different batch sizes, and on 6-point and all-zero curves. On the synthetic portfolio it costs about 18 ms of wall time per vintage forecast on a laptop CPU, ten times Chronos-Bolt-tiny and still negligible. Analogy: the second runner has now actually run the practice course, timed, before being allowed into the real race.

## Where things stand

Built and tested on synthetic data: the loader, the vintage builder, the macro loader, the data-fit check, three classical contestants, the TSFM interface with a fake model and a real Chronos-Bolt-tiny adapter (run end to end on synthetic data), Chronos-2 and TimesFM 2.5 stubs, the harness, scoring, configs, the sealed test guard, the thin-history experiment, hypothetical macro scenarios, the drift monitor, and the agent (tools, runner, critic, reporter) on hand-written cassettes. Blocked on the owner: Freddie Mac registration and download, a FRED API key, and an Anthropic API key for a live agent run. Not started: TSFMs with macro covariates, the ensemble contestant, calibration plots, the toy reserve translation, and slice analysis.

How to say it: "The whole pipeline, the experiments, and the agent are built and tested against a simulator with known answers; the real-data check runs the day the free data registration comes through, and the results file is deliberately empty until then."
