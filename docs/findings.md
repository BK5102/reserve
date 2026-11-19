# Findings and limits

**No results yet.** Nothing in this file is a measurement. It is a placeholder with the structure from `PROMPT.md` section 8, to be filled only from result files under `artifacts/runs/`, each number tagged with its run id and config hash.

Why there is nothing here: the Freddie Mac files have not been downloaded and no FRED API key is set yet, so no real data has been read (`docs/data-fit.md`). Runs on the synthetic portfolio, including the dress rehearsal of the real-data commands, exist only to check the pipeline and are never reported here (methodology M17, M41). `docs/runbook.md` step 14 says how this file is filled: from `reserve summary` output and results files only.

## 1. Question

Does a zero-shot time series foundation model forecast the next 12 months of Freddie Mac vintage delinquency curves (cumulative share ever 90+ days late or defaulted) better than the best classical baseline, with honest 50% and 90% intervals? (methodology M1)

## 2. Setup

[PLACEHOLDER: data release, commit of `data/checksums.sha256`, vintages used, origins, contestants and versions (whether TimesFM 2.5 was included), config hash, lock commit, number of sealed runs.]

## 3. Headline table

[PLACEHOLDER: quantile loss relative to the best classical baseline at 12 months, Diebold-Mariano p-values with Holm's correction, number of comparisons, 50% and 90% coverage with intervals, per contestant. Sealed test, 2013-03 to 2018-12 origins.]

## 4. What worked

[PLACEHOLDER]

## 5. What did not work

[PLACEHOLDER. Negative results stay in.]

## 6. Recession, descriptive only

[PLACEHOLDER: 2007-03 to 2009-03 origins. One episode; no general claim.]

## 7. What could not be tested

[PLACEHOLDER. Known in advance: TSFM pretraining cutoffs are undisclosed (`docs/tsfm-notes.md`); the test period has about six independent 12-month windows (methodology M21).]

## 8. Next steps

[PLACEHOLDER]
