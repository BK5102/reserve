# TSFM notes

Capabilities of each time series foundation model, checked against the model's own documentation at build time (CLAUDE.md). Nothing here is assumed; "not stated" means the source did not say, and "unverified" means it must be checked before the first real run.

| Model | Id and revision | License | Size | Quantiles | Covariates | Context | Pretraining cutoff | Source, date checked |
|---|---|---|---|---|---|---|---|---|
| Chronos-Bolt-tiny | `amazon/chronos-bolt-tiny` at `a0e552de83495b5c28c14c71c374f3e33280b340` | Apache-2.0 | 9M (model card table), 8.65M (Safetensors metadata) | Yes: the card says it outputs quantile forecasts, shape [series, quantiles, horizon]. Native training levels not stated on the card; unverified | Not stated for the base model (the card mentions covariates only for SageMaker endpoints and AutoGluon regressors) | Not stated (the card's benchmark used 512) | Not stated; the card says about 100 billion observations, with no date range: treat as undisclosed | [Hugging Face model card](https://huggingface.co/amazon/chronos-bolt-tiny), 2026-11-17; revision from Tells `docs/tsfm-notes.md` |

## Chronos-Bolt-tiny in Reserve

Status: adapter written (`src/reserve/models/tsfm/chronos_bolt.py`), never run; needs `uv sync --extra tsfm`, which installs CPU-only PyTorch and `chronos-forecasting`. Each vintage's cumulative curve, indexed by age with no dates, is one univariate series. The call (`predict_quantiles(series, prediction_length, quantile_levels)` returning quantiles shaped (series, horizon, levels)) follows Tells' adapter, which ran with chronos-forecasting 2.3.2.

To check before the first real run:

1. Which quantile levels the model was trained on, and what the library does for 0.05 and 0.95 (Reserve's outer levels).
2. Whether `predict_quantiles` accepts a list of 1-D tensors of different lengths (vintages have different ages at an origin).
3. Maximum useful context length; Reserve's longest curves will be about 300 months.
4. CPU time per origin on the real number of vintages.

Contamination: the pretraining corpus and cutoff are not disclosed on the card, so Freddie Mac curves or the macro history may be in it. The input is date-free, which removes calendar cues but not memorized shapes. This must be stated with any result (CLAUDE.md rule 9).
