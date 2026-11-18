# Explaining Reserve: the project step by step

This is the guide to read before talking about Reserve with a recruiter, a hiring manager, or an interviewer. It tells the story in the order the work happened. Every step has the same parts:

- **What I did**
- **Why**
- **What else I considered**
- **Cause and effect**: what happened because of this step, including problems it exposed
- **How to say it**: one or two sentences you can say out loud

The technical detail behind each step is in `docs/methodology.md` (numbered M1, M2, ...), and the dated diary is in `docs/decisions.md`. No result exists yet, so nothing here claims one.

## Elevator pitch (draft, no results yet)

"Lenders watch each quarter's batch of loans, called a vintage, the way a school tracks a graduating class: what share has gone seriously late by each age. I am testing whether a pretrained time series model forecasts the next year of those curves better than the methods lenders use, with honest uncertainty ranges, on public Freddie Mac mortgage data. The pipeline is built and tested on synthetic data; the real-data check is waiting on a free data registration."

---

## Part 1. Setup

### Step 1. Scaffold the repo and reuse the shared testing toolkit

- **What I did:** set up the project (uv, ruff, pytest, CI on Python 3.11 and 3.12) and installed the Cleanroom, the honest-testing toolkit built in my Tells project, from a fixed git tag.
- **Why:** the same toolkit across projects means the same exam rules everywhere, and a fix in one place fixes all of them.
- **What else I considered:** copying the toolkit into this repo. Rejected because copies drift apart.
- **Cause and effect:** the tagged version (v0.1) does not yet have forecasting tools (forecast losses, rolling forecast start points). Reserve has small local stand-ins marked to be replaced when v0.2 is tagged.
- **How to say it:** "I reuse one tested evaluation toolkit across my projects, pinned to a version, so every project sits the same honest exam."
