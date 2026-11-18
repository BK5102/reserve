# Decisions

One entry per decision or feature: what was decided, why, and an analogy. These are the owner's talking points. Newest at the bottom. The full reasoning for each choice is in `docs/methodology.md`.

## Scaffold (2026-11-17)

### D1. Repo scaffold with the shared Cleanroom installed from its git tag

Reserve is a uv project with ruff, pytest, and CI on Python 3.11 and 3.12, and it installs the Cleanroom testing toolkit from the Tells repo at the tag `cleanroom-v0.1` instead of keeping its own copy. One shared copy means a leak bug fixed once is fixed for every project, and the claim that all projects sit the same honest exam stays true. Analogy: three schools using the same sealed exam board instead of each photocopying and editing the rulebook.

### D2. CPU only

Every run, including the full one, must work on a laptop CPU, so the PLAN.md line allowing a GPU for the full run was removed and CLAUDE.md gained a CPU-only rule. Tells showed that a step needing a GPU ends up resting on one hard-to-repeat run. Analogy: plan the trip for the car you own, not one you might borrow.
