# AI_AUDIT - flaws caught in AI outputs ("catch and correct")

Goal: at least 4 real flaws, at least one per student, each with the root cause, the fix and a proof
that the fix works. Claiming the AI made zero errors scores 0/5. We only log flaws that really happened,
and we keep the wrong AI output as evidence (commit, screenshot or quote).

Template:

```
## #N - <flaw> (<date>, caught by <student>)
AI output (quote / commit):
Why it is wrong (root cause):
Impact if not caught:
Fix (commit / file:line):
Verification:
```

---

## #1 - Optimizer silently stopped at its starting point (2026-11-27, caught by Pierre)

AI output: `src/backtest.py` (first version), `min_variance_weights` called `scipy.optimize.minimize`
(SLSQP) on `w @ Sigma @ w` with default tolerances.

Why it is wrong (root cause): weekly variances are ~1e-4, below SLSQP's default stopping tolerance
(`ftol = 1e-6`). The solver stopped almost immediately and returned weights close to its starting point
(1/N) with no error or warning.

How it was caught: a sanity check on the validation table. The "minimum-variance" portfolio had a
higher volatility (8.3 %) than the max-Sharpe portfolio (6.8 %), which is impossible, and a turnover
close to 1/N's.

Impact if not caught: every minimum-variance result (and the fallback of max-Sharpe) would be wrong,
and the min-variance vs mean-variance ablation (O5) would be meaningless.

Fix: Sigma rescaled by its average variance before optimizing, and `ftol = 1e-12`
(`src/backtest.py`, `min_variance_weights` and `_solve`). Min-variance volatility drops to 4.7 %.

Verification: `tests/test_backtest.py::test_optimizers_beat_a_random_search` checks both optimizers
against 20,000 random feasible portfolios.

---

---

## Flaws found in the AI first draft of the week 1 specification (to split into one entry each)

The first draft of the week 1 report was written with an AI tool. We found these flaws when reviewing it,
and they are all fixed in `report/week1/week1_report.tex` (27 Nov). Each one still needs its own entry,
with the name of the student who caught it.

* The Sharpe ratio was maximized without the risk-free rate and without any weight cap, which leads to
  corner portfolios driven by a noisy mu.
* Transaction costs were left for later, even though the portfolio is rebalanced every week.
* The neighbour pool was frozen at 2016, so test weeks could never use 2017-2026 analogues like 2020 or 2022.
* Feature redundancy was never checked, while vix_level and spy_vol_12w have a correlation of 0.86 (2005-2016).

---

## What we still need to watch (typical AI mistakes for this project)

These are checks, not entries: an entry only exists once the AI has actually made the mistake.

* `StandardScaler` or Ledoit-Wolf fitted on the whole sample instead of on data up to t.
* `GridSearchCV` / `cross_val_score` with the default shuffled K-fold on time series, instead of the fixed
  validation period (`PredefinedSplit`) or `TimeSeriesSplit`.
* A stacking meta-model trained on K-fold out-of-fold predictions instead of walk-forward ones.
* k-NN neighbours whose target r_{s+1} isn't known yet at t (s + 1 > t).
* A Sharpe ratio without the risk-free rate, or annualized with sqrt(252) on weekly data.
* Transaction costs charged on the weights instead of the turnover |w_t - w_{t-1}|.
