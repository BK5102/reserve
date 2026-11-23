# Findings and limits

**One sealed result exists (2026-11-23).** Every number below is copied from `reserve summary` output for the results files named next to it, with the run id and config hash. Development numbers are in `docs/decisions.md` D29 and are not results.

## 1. Question

Does a zero-shot time series foundation model forecast the next 12 months of Freddie Mac vintage delinquency curves (cumulative share ever 90+ days late or defaulted) better than the best classical baseline, with honest 50% and 90% intervals? (methodology M1)

**Short answer:**
- **Accuracy:** no reliable difference. The best pretrained model, Chronos-2, had a 5.6% lower relative error than the simple reference, but no comparison was close to significant (every Holm-adjusted p-value is 1).
- **Intervals:** yes. All three pretrained models' 90% intervals covered 83 to 89% of outcomes, against 41 to 54% for the classical methods.

## 2. Setup

- **Data:** the Freddie Mac Single-Family Loan-Level Dataset sample, 1999Q1 to 2026Q2 origination files, performance through 2026-06 (`docs/data-fit.md`).
  - 1,375,000 loans in 110 quarterly vintages.
  - Macro inputs from FRED/ALFRED in `asof_fallback` mode (D26): as-known values where ALFRED has them, flagged latest values before that. `C_hazard_macro` used the fallback for 157 training months.
- **Checksums and lock:** raw-file checksums in `data/checksums.sha256` (59 files). They were committed with the lock in commit `92d78ea` on 2026-11-22, before the run.
- **Sealed run:** run `20261123T022625Z_freddie_test_test_d6120cf1`, config hash `d6120cf1b45a`, git `92d78ea3` (clean tree).
  - Sealed run number 1 under its lock; there were no other attempts.
  - The results file is kept at `artifacts/samples/20261123T022625Z_freddie_test_test_d6120cf1/results.json`.
- **Test design:** 24 quarterly origins, 2013-03 to 2018-12, horizon 12 months, 9,576 scored rows. Targets in 2020-01 to 2021-12 (COVID) are excluded (M20).
- **Contestants:**
  - Classical: `A_last_increment` (window 3), `B_age_curve` (8 donors) and `C_hazard_macro`.
  - Pretrained, zero-shot on CPU: `chronos_bolt_tiny` (amazon/chronos-bolt-tiny@a0e552de, chronos-forecasting 2.3.2), Chronos-2 and TimesFM 2.5 (D27).
  - Reference: the M29 rule picked `A_last_increment`, the same winner as on development (D29).
  - 5 comparisons, Holm-adjusted.

## 3. Headline table

Sealed test, run `20261123T022625Z_freddie_test_test_d6120cf1`, config hash `d6120cf1b45a`, 24 origins, horizon 12 months.

| Contestant | Mean quantile loss | Relative error vs reference | DM p | Holm p | 50% coverage [95% CI] | 90% coverage [95% CI] |
|---|---|---|---|---|---|---|
| A_last_increment (reference) | 0.0003113 | 1 |  |  | 0.235 [0.215, 0.256] | 0.479 [0.455, 0.504] |
| B_age_curve | 0.0003601 | 1.025 | 0.6339 | 1 | 0.225 [0.205, 0.246] | 0.541 [0.517, 0.566] |
| C_hazard_macro | 0.0005226 | 1.613 | 0.2141 | 1 | 0.222 [0.202, 0.243] | 0.409 [0.385, 0.433] |
| D_tsfm_chronos2 | 0.0002746 | 0.944 | 0.3228 | 1 | 0.467 [0.442, 0.491] | 0.888 [0.871, 0.902] |
| D_tsfm_chronos_bolt_tiny | 0.0003431 | 1.176 | 0.3747 | 1 | 0.420 [0.396, 0.445] | 0.843 [0.824, 0.860] |
| D_tsfm_timesfm_2p5 | 0.0003072 | 1.057 | 0.8912 | 1 | 0.591 [0.567, 0.615] | 0.831 [0.812, 0.848] |

Caveats printed with the table, and part of the result:
- **Narrower native intervals:** Chronos-Bolt-tiny and TimesFM 2.5 are trained only on levels 0.1 to 0.9, so their 90% interval is their own 80% interval, and their 90% coverage is low by construction (M32).
- **Independence:** coverage intervals treat forecasts as independent, which they are not (L20), so they are too narrow.

## 4. What worked

- **Honest 90% intervals:** Chronos-2's 90% coverage interval, [0.871, 0.902], includes 0.90. The other two pretrained models reached 0.84 and 0.83 with what are really 80% intervals.
- **Accuracy without training:** the pretrained models matched the classical methods with no training on loan data. Chronos-2 had the lowest loss of all six contestants, but not by a significant margin.
- **The plan held:** the reference was chosen by the pre-written rule and was the same as on development, and the test was scored once under a committed lock.

## 5. What did not work

- **No significant accuracy gain:** no pretrained model beat the reference after Holm's correction. Two of the three, Chronos-Bolt-tiny (1.176) and TimesFM 2.5 (1.057), had a higher relative error than the reference.
- **The macro model lost:** `C_hazard_macro`, the only contestant that reads the economy, had the highest loss (relative error 1.613) and the lowest 90% coverage (0.409).
- **Classical intervals far too narrow:** the classical methods' 90% intervals covered only 0.41 to 0.54 of outcomes, and their 50% intervals 0.22 to 0.24.
  - The drift monitor on the reference (`reserve monitor`, reference end 2014-12) shows coverage red at every origin from 2015-03 to 2018-12. Its loss ratio is green throughout.
  - Part of the pretrained models' interval advantage is therefore that the classical intervals are poorly calibrated, not only that the pretrained ones are good.
- **50% intervals off in both directions:** Chronos-2 slightly under-covered (0.467) and TimesFM 2.5 over-covered (0.591).

## 6. Recession, descriptive only

Origins 2007-03 to 2009-03, run `20261122T025409Z_freddie_sample_recession_a78402a8`, config hash `a78402a83575` (development config, labelled descriptive; one episode, so no general claim; M31).
- **Reference:** `A_last_increment` again, loss 0.001482, 90% coverage 0.371.
- **Chronos-2:** loss 0.001472 (relative error 1.039, Holm p 0.8739) and 90% coverage 0.667.
- **The other two pretrained models:** worse, at relative errors 1.916 (Chronos-Bolt-tiny) and 1.593 (TimesFM 2.5), with 90% coverage 0.390 and 0.565.
- **The macro model:** `C_hazard_macro` had relative error 1.425 (DM p 0.01131, Holm p 0.05657).
- **Summary:** in the one bad economy, every method's intervals were too narrow. Only Chronos-2 stayed level with the reference.

## 7. What could not be tested

- **Power:** the test period holds about six independent 12-month windows (M21). A small real difference in loss cannot be told from zero, so "no significant difference" is not "no difference".
- **Contamination:** pretraining cutoffs are undisclosed (`docs/tsfm-notes.md`), and TimesFM 2.5's training data extends past the test period (L25). The pretrained models may have seen series like these.
- **Fair interval comparison:** the classical intervals were not recalibrated (for example with conformal intervals), so the interval comparison is against uncalibrated baselines.
- **Revisions:** macro revisions are ignored before ALFRED's first vintages (D26), which affects only `C_hazard_macro`'s training months.
- **Not built:** TSFMs with macro inputs, an ensemble, and the agent's evidence pack (optional, live API calls; not run, to avoid spending).

## 8. Next steps

- **Fairer interval test:** give the classical baselines calibrated intervals (conformal or empirical residual quantiles) and test again on new, unseen origins. That checks whether the pretrained models' interval advantage survives a fair comparison.
- **More power:** a longer test period, or the full dataset rather than the sample.
- **More pretrained variants:** pretrained models with macro covariates, and a stricter contamination check.
