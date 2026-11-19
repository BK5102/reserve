# CLAUDE.md: Reserve

Read this first in every session. It tells you what this project is, how it is laid out, and the rules you must not break.

## What this project is

Reserve tests whether time series foundation models (TSFMs) can forecast delinquency curves for vintages of loans better than classical methods, including in a held-out recession and with thin history, and whether the forecast intervals are honest. An AI agent runs the experiments and drafts a validation-style evidence pack, including an adversarial critique of its own findings. See `PLAN.md` for diagrams, requirements, and the 30-day plan.

Plain-English summary: lenders track groups of loans issued in the same period (vintages) the way a school tracks a graduating class. Reserve asks whether a pretrained forecaster predicts how each class will fare, including in a bad economy, and whether it knows how uncertain it is.

## Who you are working with

The owner is a data science and economics student who is new to credit risk and TSFMs. Explain every new term in one plain sentence with a real-world analogy the first time it appears. Keep messages short and direct.

Writing style for all docs, comments, commit messages, and chat:

- Concise, specific, plain language.
- No em-dashes. No emojis.
- Prose over bullet points when prose reads fine.
- Make edits precisely. Do not rewrite surrounding content that was not asked about.
- Cite sources for factual claims about the field. If you cannot source it, say so.

## Repo layout

```
reserve/
  CLAUDE.md  PROMPT.md  PLAN.md  README.md
  pyproject.toml  uv.lock
  src/reserve/
    data/        loans.py  vintages.py  macro.py
    models/      baselines.py  tsfm/ (one adapter per TSFM, quantile output)
    evaluation/  origins.py  harness.py  scoring.py (Cleanroom splits and forecast scores)
    stress/      scenarios.py
    monitoring/  drift.py
    agent/       planner.py  runner.py  critic.py  reporter.py  tools.py  llm.py  spend.py
    datafit.py  pipeline.py  config.py  cli.py  doctor.py  checksums.py  summary.py
  tests/
  configs/
  docs/          runbook.md  methodology.md  explain-the-project.md  tsfm-notes.md  decisions.md  data-fit.md  findings.md
  data/          gitignored. Download scripts and checksums only.
  artifacts/     gitignored except small sample outputs
```

The testing toolkit comes from the shared `cleanroom` package (split, lock, metrics, bootstrap, multiple-comparison correction, canary), built in the sibling project Tells under `packages/cleanroom` and installed here from a git tag (currently `cleanroom-v0.2`, pinned in `pyproject.toml` under `[tool.uv.sources]`). Import it. Do not copy it. If something is missing, add it to `cleanroom` in the Tells repo in a domain-neutral way, release a new `cleanroom-vX.Y` tag, then bump the version here.

## Commands

```
uv sync                                       # extras in one command: --extra tsfm (Chronos-Bolt, Chronos-2) --extra timesfm --extra agent
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run reserve prepare synthetic --config configs/synthetic_small.yaml   # fake data, seconds
uv run reserve run configs/synthetic_small.yaml                          # pipeline check, not a result
uv run python scripts/check_data_fit.py synthetic configs/synthetic_fit.yaml
uv run reserve doctor --config configs/freddie_test.yaml   # downloads, keys, extras, git: ready?
uv run python scripts/rehearse_real_path.py   # the real-data commands on SYNTHETIC files (CI runs it)
uv run reserve prepare macro                  # needs FRED_API_KEY in the environment
uv run reserve prepare macro --cassette tests/fixtures/fred/cassette.json --out <dir>   # replay, no key
uv run reserve prepare freddie --config configs/freddie_sample.yaml      # after the Freddie Mac download
uv run python scripts/check_data_fit.py freddie data/raw/freddie --fred data/raw/fred
uv run reserve run configs/freddie_sample.yaml --period recession   # descriptive only
uv run reserve summary artifacts/runs/<run>   # headline table from a results file, with run id and config hash
uv run reserve checksums --config configs/freddie_test.yaml   # raw-file SHA-256 into data/checksums.sha256
uv run reserve seal configs/freddie_test.yaml   # refuses until real data is checksummed; commit lock + checksums
uv run reserve run configs/synthetic_small.yaml --experiment thin_history   # dev only: 6, 12, 24 months of history
uv run reserve run configs/synthetic_small.yaml --experiment scenarios      # dev only: hypothetical macro paths, never scored
uv run reserve monitor artifacts/runs/<run> --contestant <name> --reference-end <date>   # drift lights for a finished run
uv sync --extra tsfm --extra agent && uv run reserve agent "do TSFMs help when history is only 12 months?" --record   # needs ANTHROPIC_API_KEY; capped by configs/agent.yaml
uv run reserve agent "..." --cassette artifacts/agent/cassettes/<time>.json   # replay a recorded run, no API call
```

If a command above does not exist yet, create it as part of the day's task and update this file. The full real-data order is in `docs/runbook.md`.

## Rules you must not break

1. **No future information.** Forecasts at origin t use only data known at t. This includes macro series: use as-known-then values (FRED's ALFRED archive) where possible, and flag clearly when revisions are ignored.
2. **All splits go through `cleanroom.split`.** No ad hoc splits.
3. **No test scoring without a lock.** The config hash must be recorded first. Never bypass it.
4. **Report numbers come from result files.** The reporter inserts numbers programmatically. An LLM writes narrative only.
5. **Public data only.** No real customer data. Do not commit raw data.
6. **Hypothetical means hypothetical.** Stress scenarios are labeled "hypothetical" in code, plots, and reports. Never present them as forecasts of the economy.
7. **Never claim compliance.** Evidence packs are "inspired by validation practice, not a compliance document." Say so in every generated pack.
8. **Multiple-comparison correction is mandatory** whenever more than one contestant or variant is compared. Report how many were tried.
9. **State contamination limits.** Record each TSFM's documented pretraining cutoff or "undisclosed." Run the date-masked variant. Say what remains uncontrolled.
10. **Never invent results, datasets, or citations.** Use clearly marked placeholders.
11. **Negative results stay in.**
12. **Record provenance** on every result: config hash, git commit, model name and version, data checksum.
13. **Every choice is explained.** Any new method, variable, threshold, default, or dataset choice gets an entry in `docs/methodology.md` (what, why, what else was considered, limitation) in the same commit, and any new limitation goes in its limitations register. The owner must be able to defend every choice in an interview.
14. **Keep the story current.** After every step that changes the project or produces a result, update `docs/explain-the-project.md`: what was done, why, what else was considered, cause and effect, and a one-sentence way to say it in an interview.
15. **CPU only.** Every run, including the full run, must work on a laptop CPU. Do not plan for or require a GPU.

## Agent safety rules (for `src/reserve/agent/`)

- Tool allow-list only: `run_forecast_experiment`, `load_results`, `plot`, `compute_stats`. No shell, no network, no arbitrary file access.
- The Runner cannot read test-period actuals before the lock exists.
- The Critic runs deterministic checks first, then an adversarial LLM pass that must list the three strongest objections per headline claim, each pointing to evidence in the results.
- LLM calls sit behind an interface. Tests use recorded cassettes.

## Workflow

- One shippable feature per day, as listed in `PLAN.md` Part 6. Small commits, each with a test.
- Commit style: `type: short summary` using `feat`, `fix`, `test`, `docs`, `chore`.
- Before finishing a task: run tests and lint, then state what changed and what you did not verify.
- When a task is ambiguous, ask one specific question instead of guessing.
- After each feature, add a 3-sentence plain-English note to `docs/decisions.md` with an analogy. These become the owner's talking points.
- Check TSFM capabilities (quantile output, covariate support, context length, license) against the model's own documentation at build time and record them in `docs/tsfm-notes.md`. Do not assume.

## Definition of done for any feature

Tests pass, lint passes, a leak or reconciliation test exists if the feature touches data, splits, or vintages, docs are updated if behavior changed, and the change is explained in plain English.
