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
