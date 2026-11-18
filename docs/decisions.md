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
