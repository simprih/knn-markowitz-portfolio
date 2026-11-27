# CONTRIBUTIONS - ownership matrix

One owner per module. Each owner has to be able to explain every function, parameter and library call
of their own module at the oral (rubric item 5). Each module also has a reviewer from the team.

| Module | Files | Owner | Reviewer | Status |
|---|---|---|---|---|
| 1. Data pipeline, features, EDA, feature ablation | `src/data.py`, `src/eda.py`, `src/selection.py`, `tests/test_data.py` | Sacha Audouy | Pierre Rançon | data + EDA v0.1 done (27 Nov); ablation in week 3 |
| 2. Walk-forward engine, optimizer, baseline (+ window tuning), 1/N, metrics | `src/backtest.py`, `src/baseline.py` | Pierre Rançon | Simon Prihnenko | week 2 |
| 3. Models: k-NN, decision tree, Lasso, Voting ensemble, grid search | `src/knn.py`, `src/forecasters.py`, `src/ensemble.py` | Simon Prihnenko | Sacha Audouy | week 3 |
| 4. Model outside the course: Gaussian HMM of regimes (Ang & Bekaert, 2002) | `src/hmm.py` | Pierre Rançon | Simon Prihnenko | week 3 |
| 5. Report + oral | `report/` | all | | week 4 |

If we have time at the end, we'll add one extension: the cost sensitivity at 5 bps.

Everyone writes at least one entry in `PROMPT_LOG.md` and one in `AI_AUDIT.md` (otherwise -2 pts per student).

## Change log

| Date | Who | What |
|---|---|---|
| 2026-11-27 | Pierre (with AI help, see PROMPT_LOG #1) | First version of module 1 for the 6-ETF universe; week 1 specification corrected. Sacha to review it and take ownership. |
| 2026-11-27 | Simon | Back to 3 owners, one per student: tree, Lasso and Voting go to module 3, feature ablation to module 1, the HMM (required outside-the-course model) to Pierre. EDA statistics now computed on 2005-2016 only, with the target distribution and prevalence added. |
