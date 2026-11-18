# PROMPT.md: Reserve

Prompts to paste into Claude Code. Put `CLAUDE.md`, `PROMPT.md`, and `PLAN.md` at the repo root first. Build the Cleanroom in Tells before starting.

---

## 1. Kickoff prompt (paste once, on day 1)

```
You are my pair-programmer and teacher. I am a data science and economics student who is new to credit risk and time series foundation models (TSFMs).

Read CLAUDE.md and PLAN.md in full before doing anything. Then:

1. Summarize the project back to me in 5 sentences of plain English with one real-world analogy (for example vintages as graduating classes), so I can check we agree.
2. Install the shared `cleanroom` package from the Tells repo tag (`uv add "cleanroom @ git+https://github.com/BK5102/tells@cleanroom-v0.1#subdirectory=packages/cleanroom"`) and check it imports. If the tag does not exist or the import fails, stop and tell me exactly what is missing. Never copy Cleanroom code into this repo.
3. Do ONLY Day 1 of PLAN.md Part 6: repo scaffold with uv, ruff, pytest, CI, the cleanroom dependency, and a README skeleton that embeds Diagram 1 from PLAN.md. Do not start Day 2.
4. Create docs/decisions.md and docs/tsfm-notes.md as empty templates.
5. Run tests and lint. Show me the output.
6. End with: what you built (3 sentences, plain English, one analogy), what you did not verify, and the exact first command I should run.

Rules: follow every rule in CLAUDE.md. Do not invent datasets, results, or citations. If you are unsure about something, ask me one specific question. No em-dashes, no emojis.
```

## 2. Daily prompt (paste every day, change the number)

```
Today is Day N of the Reserve 30-day plan.

1. Read CLAUDE.md and the Day N row in PLAN.md Part 6. Read docs/decisions.md for yesterday's notes.
2. Restate today's task in two sentences and list the files you will touch. Wait for my "go" if the task changes the Cleanroom or the agent's permissions; otherwise proceed.
3. Write the test first (a reconciliation or leak test if it touches data, vintages, macro series, or splits). Then the code.
4. Run pytest and ruff. Fix failures.
5. Append a 3-sentence note to docs/decisions.md: what I built, why it matters, and a real-world analogy.
6. Propose a commit message in the form "type: short summary".
7. Finish with: what changed, what you did NOT verify, and one question that tests whether I understood today's feature.
```

## 3. Data sanity prompt (run at Days 4, 5, 6)

```
Audit the data pipeline like a skeptic. Read src/reserve/data and the latest config.

Check, with evidence from the code and by running small examples:
- Is each loan counted exactly once per vintage and month?
- Do vintage totals reconcile with the loan-level data?
- Is any macro value used at an origin date later than when it was actually published? If as-known-then data is not available, is that flagged in the output?
- Does the split guarantee that no forecast origin sees later data?
- Are delinquency definitions (30, 60, 90+ days, default) consistent and documented?

Report each check as PASS or FAIL with file and line references. Do not fix anything yet. List fixes in priority order.
```

## 4. Calibration and stress review prompt (run at Days 17, 18)

```
Act as an independent model reviewer. Read the latest results.

1. For each contestant, report 50 and 90 percent interval coverage overall, by horizon, and in the recession holdout. Flag any contestant whose coverage is more than 10 points off nominal.
2. State in plain English what under-coverage would mean for a lender setting reserves.
3. Check that stress scenarios are labeled hypothetical everywhere they appear.
4. Check that multiple-comparison correction was applied and that the number of variants tried is reported.
5. List what the recession holdout cannot tell us (for example, it is a single episode).

Output a short table, then a paragraph in plain English. Do not fix anything yet.
```

## 5. Explain-it-to-me prompt (run after each week)

```
Quiz me. Ask me, one at a time, five questions about what we built this week, starting with the simplest. After each answer, tell me what I got right, what I got wrong, and give a better 2-sentence version using a real-world analogy. Finish by writing a 60-second spoken explanation of the project as it stands today, using only things that actually exist in the repo.
```

## 6. Recruiter rehearsal prompt (run on Days 20, 28, 30)

```
Play a recruiter at a fintech company, then a hiring manager on a credit risk team, then an ML interviewer. Ask me the questions they would really ask about Reserve, one at a time. After each answer, score it from 1 to 5 for clarity and honesty, point out any claim I cannot back with something in the repo, and suggest a tighter version. Use the "Questions you will probably get" and "What not to say" sections of PLAN.md Part 4 as the base, but add new questions, including one about why I did not use loan-level default models.
```

## 7. Evidence pack challenge prompt (Day 27)

```
Read the generated evidence pack and the results files. For each headline claim, write the three strongest objections a skeptical model validator would raise. For each objection, point to the specific result or missing test that supports it, and say whether the pack already addresses it. Do not soften the objections. Output them as a table: claim, objection, evidence, addressed in pack (yes, partly, no).
```

## 8. Findings write-up prompt (Day 28)

```
Read artifacts/results (all runs). Write docs/findings.md with these sections: question, setup, headline table, what worked, what did not work, what could not be tested, and next steps. Every number must be copied from a results file and tagged with its config hash. Include negative results and the recession-holdout limits. Label stress scenarios hypothetical. Do not use causal language the experiments cannot support. No em-dashes.
```

## 9. When something breaks

```
Do not guess. Reproduce the failure, show me the smallest failing case, explain the cause in plain English, propose the minimal fix, and add a regression test before changing code.
```
