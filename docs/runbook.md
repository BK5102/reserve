# Runbook: from the manual tasks to a sealed, written-up result

This is the exact order of commands once the owner has finished the manual tasks. Every command runs from the repo root on a laptop CPU. The same sequence, on SYNTHETIC files in a throwaway repo, runs in CI on every push (`scripts/rehearse_real_path.py`, methodology M41), so the commands themselves are known to work; what is not yet known is how real files behave.

Commands are shown for PowerShell; in bash, set variables with `export NAME=value` instead of `$env:NAME = "value"`.

## The manual tasks this runbook assumes are done

1. **Freddie Mac sample files.** Registered at Freddie Mac's Clarity Data Intelligence (linked as "Access Historical Data" on the [dataset page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)), downloaded every `sample_YYYY.zip`, and unzipped them so that `data/raw/freddie/` directly holds `sample_orig_YYYY.txt` and `sample_perf_YYYY.txt` pairs (no subfolders). The files are never committed.
2. **FRED API key** from [FRED API keys](https://fred.stlouisfed.org/docs/api/api_key.html), kept in the environment only: `$env:FRED_API_KEY = "your-key"`.
3. **Anthropic API key** for the agent, environment only: `$env:ANTHROPIC_API_KEY = "your-key"`. Every live agent run is capped by `configs/agent.yaml` (10 dollars per run by default; methodology M45).
4. **TimesFM 2.5 decision.** Recommended: install it from the official source repo at the pinned commit, which is already declared as the `timesfm` extra (methodology M44). The switch is one flag on the install line in step 0, plus adding `timesfm_2p5` to the TSFM list in both Freddie Mac configs in step 7. Leaving it out keeps two TSFMs, which is stated in the findings.

## Step 0. Install once

`uv sync` installs exactly the extras named on its command line and removes the others, so name them all in one command:

```
uv sync --extra tsfm --extra agent                    # Chronos-Bolt-tiny, Chronos-2, the agent
uv sync --extra tsfm --extra timesfm --extra agent    # the same plus TimesFM 2.5 (recommended)
uv run pytest -q                                       # everything passes before real data
```

The first model run downloads the weights from Hugging Face at their pinned revisions (Chronos-Bolt-tiny is about 9 million parameters, Chronos-2 about 119 million, TimesFM 2.5 about 231 million).

## Step 1. Check the download location

```
uv run reserve doctor --config configs/freddie_test.yaml
```

It must end with "ready to prepare the Freddie Mac files" (exit code 0). Any `FIX` line names the problem: zip files not unzipped, files left in a subfolder, an origination file without its performance partner. `NOTE` lines are for later steps (keys, extras, checksums, git).

## Step 2. Data fit on the loans

```
uv run python scripts/check_data_fit.py freddie data/raw/freddie --out artifacts/data_fit_freddie.json
```

It prints the detected layout of every file, loans and events per vintage, the noise floor, recession and COVID coverage, reconciliation counts, and power (methodology M19 to M21). It must not print "SYNTHETIC DATA"; if it does, simulator files are in the folder (M40).

## Step 3. The go/no-go gate

The script exits 0 when the gate passes (at least 60 usable vintages and a median of at least 20 events per vintage by age 12) and 3 when it fails ("GATE FAILED").

- **Pass:** copy the numbers into the "Real-data results" table of `docs/data-fit.md` (replacing PENDING), then `git add docs/data-fit.md` and `git commit -m "docs: data-fit results on the Freddie Mac sample"`.
- **Fail:** stop here. Record the numbers the same way, write in `docs/findings.md` that the sample cannot answer the question and why, and do not run any model. A failed gate is a result.

## Step 4. Macro series as they were known at the time

```
$env:FRED_API_KEY = "your-key"
uv run reserve prepare macro
uv run python scripts/check_data_fit.py freddie data/raw/freddie --fred data/raw/fred --out artifacts/data_fit_freddie.json
```

The first command saves every ALFRED vintage of UNRATE, MORTGAGE30US, and HPIPONM226S to `data/raw/fred/`. The second repeats the data fit with the macro tables included; the `macro` section of `artifacts/data_fit_freddie.json` (it is in the JSON report, not the printed summary) gives each series' first observation, number of vintages, first vintage date, and missing values. Add them to `docs/data-fit.md`. If ALFRED has no vintages for the early years of a series, write that down (limitation L12) before going on.

## Step 5. Build the vintage tables

```
uv run reserve prepare freddie --config configs/freddie_sample.yaml
```

It writes `data/interim/freddie_sample/` (curves, loans, macro features, and `prepare.json`) and reconciles every count on the way; any reconciliation error stops it. It must not print "SYNTHETIC DATA".

## Step 6. Development runs (never the test period)

```
uv run reserve run configs/freddie_sample.yaml                                  # dev: origins 2010-03 to 2012-03
uv run reserve run configs/freddie_sample.yaml --period recession               # descriptive only
uv run reserve run configs/freddie_sample.yaml --experiment thin_history        # 6, 12, 24 months of history
uv run reserve run configs/freddie_sample.yaml --experiment scenarios           # HYPOTHETICAL, never scored
uv run reserve summary artifacts/runs/<dev run id>
uv run reserve monitor artifacts/runs/<dev run id> --contestant <reference> --reference-end 2011-03
```

Each run writes `artifacts/runs/<run id>/results.json` with its provenance. Use these runs to find bugs and to look at the data, not to tune anything toward a better test score. Development and recession numbers may appear in the findings, labelled as such.

## Step 7. The reference baseline, and freezing the plan

The reference is not chosen by hand. The rule, written before any result (methodology M29), is: in each scored table, the classical contestant (A, B, or C) with the lowest mean quantile loss is the reference, and every other contestant is compared with it by relative error and a Diebold-Mariano test with Holm's correction. `reserve summary` prints which one the rule picked. Note which contestant won on dev in `docs/decisions.md`, so a different winner on the test is visible, not surprising.

If development changed anything in `configs/freddie_sample.yaml`, make the identical change in `configs/freddie_test.yaml`. For TimesFM, add `timesfm_2p5` to the `tsfm` list of both files now if the `timesfm` extra is installed; never after sealing. Then:

```
uv run pytest -q tests/test_pipeline.py -k same_plan   # the two configs differ only in name and period
git add configs
git commit -m "chore: freeze the test plan"
```

## Step 8. Record the raw-file checksums

```
uv run reserve checksums --config configs/freddie_test.yaml
```

It writes `data/checksums.sha256` (`sha256sum` format) for every Freddie Mac file and every saved FRED table. Run it after the last `reserve prepare macro`, because a new FRED download changes those checksums (M42).

## Step 9. Seal the test plan

```
uv run reserve seal configs/freddie_test.yaml --note "2013Q1-2018Q4 origins, horizon 12, plan frozen before any test score"
```

It refuses, listing every reason at once, unless the data is real (no synthetic markers), every raw file is checksummed and unchanged, the tables were prepared from exactly those files, all three classical baselines are in the plan, and every TSFM is real and installed. Fix what it lists and run it again.

## Step 10. Commit the lock

```
git add locks data/checksums.sha256
git commit -m "chore: seal the 2013-2018 test plan"
git push
uv run reserve doctor --config configs/freddie_test.yaml   # the git line must say "clean"
```

The commit date is the evidence that the plan existed before the result.

## Step 11. The sealed run, once

```
uv run reserve run configs/freddie_test.yaml
```

It refuses without a committed lock for exactly this config and data, or with uncommitted changes. It records the attempt as "started" before scoring and "finished" after; an attempt that crashes still counts. Then keep the record and the result:

```
git add locks
git commit -m "chore: record sealed test run 1"
mkdir artifacts/samples/<run id>
copy artifacts/runs/<run id>/results.json artifacts/samples/<run id>/   # bash: cp
git add artifacts/samples
git commit -m "chore: keep the sealed test results file"
git push
```

Do not run it again to get a different number. If a rerun is ever needed (for example after a crash), it is counted, and the findings must say how many sealed runs there were and why.

## Step 12. Evaluate

```
uv run reserve summary artifacts/runs/<test run id>
uv run reserve summary artifacts/runs/<recession run id>
uv run reserve monitor artifacts/runs/<test run id> --contestant <reference> --reference-end 2014-12
```

`reserve summary` prints the headline table (mean quantile loss, relative error against the reference, Diebold-Mariano and Holm p-values, 50% and 90% coverage with intervals) with the run id and config hash, straight from the results file. Its caveats are part of the result: Chronos-Bolt-tiny's and TimesFM's 90% interval is their own 80% interval (M32), and coverage intervals treat forecasts as independent (L20).

## Step 13. The agent's evidence pack

```
$env:ANTHROPIC_API_KEY = "your-key"
uv run reserve agent "Did any TSFM beat the best classical baseline on the sealed 2013-2018 test, and are its intervals honest?" --allow-test --record
```

`--allow-test` lets the agent read the sealed test run (it still cannot score the test without the committed lock). `--record` saves the conversation to `artifacts/agent/cassettes/<time>.json`; replay it at no cost with `--cassette <that file>`. The run stops before passing the cap in `configs/agent.yaml` and writes the estimated spend next to the pack. Read the pack in `artifacts/agent/` and every objection in it before using any sentence from it; it is a draft, inspired by validation practice, not a compliance document.

## Step 14. Write the findings

Fill `docs/findings.md` from the `reserve summary` output and the results files only, each number with its run id and config hash: setup (data release, checksum file commit, lock commit, number of sealed runs), the headline table, what worked, what did not (negative results stay in), the recession as a single descriptive episode, what could not be tested (pretraining cutoffs, about six independent test windows), and next steps. Then update the status lines in `README.md`, `docs/explain-the-project.md`, and `docs/decisions.md`, and commit.
