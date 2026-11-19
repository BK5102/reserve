# TSFM notes

Capabilities of each time series foundation model, checked against the model's own documentation at build time (CLAUDE.md). Nothing here is assumed; "not stated" means the source did not say, and "unverified" means it must be checked before the first real run.

| Model | Id and revision | License | Size | Quantiles | Covariates | Context | Pretraining cutoff | Source, date checked |
|---|---|---|---|---|---|---|---|---|
| Chronos-Bolt-tiny | `amazon/chronos-bolt-tiny` at `a0e552de83495b5c28c14c71c374f3e33280b340` | Apache-2.0 | 9M (model card table), 8.65M (Safetensors metadata) | Yes. Trained levels 0.1, 0.2, ..., 0.9 (`chronos_config.quantiles` in `config.json` at the pinned revision, checked 2026-11-18) | Not stated for the base model (the card mentions covariates only for SageMaker endpoints and AutoGluon regressors) | 2048 steps (`chronos_config.context_length`); one pass predicts 64 steps (`prediction_length`) | Not stated; the card says about 100 billion observations, with no date range: treat as undisclosed | [Hugging Face model card](https://huggingface.co/amazon/chronos-bolt-tiny), 2026-11-17; revision from Tells `docs/tsfm-notes.md` |
| Chronos-2 | `amazon/chronos-2` at `29ec3766d36d6f73f0696f85560a422f50e8498c` (Hugging Face API head on 2026-11-18, same as Tells' pin) | Apache-2.0 | 119,477,664 (Safetensors metadata) | Yes. Trained on 21 levels: 0.01, 0.05, 0.1 to 0.9 in steps of 0.05, 0.95, 0.99 (`config.json` at the pin, from the local cache). All seven Reserve levels are native; the loaded pipeline reports the same 21 levels (checked at load, 2026-11-19) | Past-only and known-future covariates per Tells' reading of the card; not rechecked; unused | 8192 steps (`config.json`) | Undisclosed (Tells, from the paper) | Hugging Face API, cached `config.json`, chronos-forecasting 2.3.2 source, 2026-11-18; run on CPU 2026-11-19 |
| TimesFM 2.5 (STUB) | `google/timesfm-2.5-200m-pytorch`; head commit `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` (last modified 2025-10-02), not pinned | Apache-2.0 (card and GitHub README) | 231,289,280 (Safetensors metadata; card says 0.2B) | Yes. Output (series, horizon, 10): the mean, then 0.1 to 0.9 (card). Same flat-tail mapping as Chronos-Bolt would apply | Through XReg, per the GitHub README (added back 2025-10-29); not checked in code | Card example `max_context=1024`, `max_horizon=256`; README says up to 16k. Unverified which applies | Partly disclosed: GiftEvalPretrain (no date), Wikimedia pageviews to 2023-11, Google Trends to end of 2022, synthetic data. Treat as undisclosed | [Model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch), [GitHub README](https://github.com/google-research/timesfm), PyPI, 2026-11-18 |

## Chronos-Bolt-tiny in Reserve

Status: adapter real and tested (`src/reserve/models/tsfm/chronos_bolt.py`); run end to end on the synthetic portfolio only. Install with `uv sync --extra tsfm` (CPU-only PyTorch 2.15.0 and chronos-forecasting 2.3.2 in `uv.lock`). Each vintage's cumulative curve, indexed by age with no dates, is one univariate series.

The four questions this file listed before the first run, answered from chronos-forecasting 2.3.2 (`chronos/chronos_bolt.py`, `chronos/utils.py`) and the model's `config.json`, 2026-11-18:

1. Trained quantile levels: 0.1 to 0.9 in steps of 0.1. Asked for 0.05 or 0.95, the library logs a warning and returns its 0.1 or 0.9 value. Reserve asks for the nine native levels and maps them itself (methodology M32): 0.05 becomes the 0.1 value and 0.95 the 0.9 value, so Chronos-Bolt's "90% interval" is its own 80% interval. A test checks Reserve's mapping equals the library's.
2. Series of different lengths: yes. `predict_quantiles` accepts a list of 1-D tensors and left-pads them with NaN (`left_pad_and_stack_1D`). It has no batch size, so the adapter batches 64 series at a time, sorted by length; a test shows batch size does not change the forecasts.
3. Context: 2048 steps, longer inputs cut to the last 2048. Reserve's curves are at most about 300 months, so nothing is cut. The 12-month horizon fits in one 64-step pass, so there is no recursive roll-out; the adapter refuses horizons above 64.
4. CPU time: measured below.

### CPU cost (SYNTHETIC data, a pipeline check)

Run `20261118T162732Z_synthetic_tsfm_dev_41b911fd` (`configs/synthetic_tsfm.yaml`, 16 quarterly origins, 360 vintage forecasts of 12 months each, curve lengths 6 to about 90 months), on the owner's laptop (Intel64 family 6 model 142, 8 logical cores), PyTorch using 4 threads, 2026-11-18, one run:

| Model | Wall time per series | CPU time per series (all threads) | Whole model time |
|---|---|---|---|
| Chronos-Bolt-tiny | 2.9 ms | 10.7 ms | 1.05 s wall for 360 series |
| Fake stand-in (reference) | 0.14 ms | 0.43 ms | 0.05 s |

Times cover the model calls only (`TSFMContestant.cost`, stored in every run's `contestants` entry), not loading the weights. The run was made from an uncommitted working tree (marked dirty in its provenance). At this rate the Freddie Mac dev period (about 100 vintages at each of 9 origins) needs a few seconds of model time, so CPU cost does not limit Reserve.

Contamination: the pretraining corpus and cutoff are not disclosed on the card, so Freddie Mac curves or the macro history may be in it. The input is date-free, which removes calendar cues but not memorized shapes. This must be stated with any result (CLAUDE.md rule 9).

## Chronos-2 in Reserve

Status: adapter real and tested (`src/reserve/models/tsfm/chronos2.py`, methodology M43); run end to end on the synthetic portfolio only. Same `tsfm` extra as Chronos-Bolt; the weights at the pinned revision were already in the local Hugging Face cache (downloaded earlier by Tells), so nothing new was downloaded.

What was checked against the installed chronos-forecasting 2.3.2 and the loaded model, 2026-11-19:

1. Quantile levels: the pipeline reports 21 levels, 0.01, 0.05, 0.1 to 0.9 in steps of 0.05, 0.95, 0.99. The adapter asks for all 21 and keeps Reserve's seven, which are all native, so there is no interpolation and no flat tail: Chronos-2's 90% interval is its own 90% interval, unlike Chronos-Bolt's (M32). A test shows this equals asking the library for the seven levels directly.
2. Output: `predict_quantiles` returns one tensor per series shaped (variates, horizon, levels); Reserve's input is one variate. `batch_size` is a library argument (default 256); a test shows it does not change the forecasts.
3. Short and flat inputs: 6-point curves and all-zero curves give finite forecasts. With 6 points the interval is wide and its lower end falls below the last known value, which the shared clean-up clips (M23).
4. CPU time: measured below.

### CPU cost (SYNTHETIC data, a pipeline check)

Run `20261119T175607Z_synthetic_tsfm_dev_af26dda0` (`configs/synthetic_tsfm.yaml`, the same 16 origins and 360 vintage forecasts as the Chronos-Bolt run above), same laptop, PyTorch using 4 threads, 2026-11-19, one run, made from an uncommitted working tree (marked dirty in its provenance):

| Model | Wall time per series | CPU time per series (all threads) | Whole model time |
|---|---|---|---|
| Chronos-2 (batch 256) | 18.2 ms | 66.3 ms | 6.5 s wall for 360 series |
| Chronos-Bolt-tiny (batch 64), same run | 1.8 ms | 5.7 ms | 0.65 s |
| Fake stand-in (reference), same run | 0.10 ms | 0.43 ms | 0.04 s |

Loading the weights is not included. At this rate the Freddie Mac test period (24 origins of roughly 80 vintages) needs well under a minute of Chronos-2 time, so CPU cost does not limit it. Chronos-Bolt-tiny was faster here than in the 2026-11-18 run (1.8 against 2.9 ms per series); one run each, so the difference is not interpreted.

Contamination: the pretraining cutoff is undisclosed, as for Chronos-Bolt; the same statement applies to any result.

## TimesFM 2.5: stub

It lives behind the same interface (`src/reserve/models/tsfm/timesfm25.py`) and refuses to be constructed. Before it runs:

1. TimesFM 2.5: find which install provides `timesfm.TimesFM_2p5_200M_torch` (the card says install from source; PyPI's newest `timesfm` is 3.0.2, unchecked), pin a revision, check whether the context limit is 1024 or larger under the chosen `ForecastConfig`, and add it to a separate optional extra so its dependencies cannot change the Chronos versions.
2. Time it on CPU as was done for the Chronos models.
