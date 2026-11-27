# Week 2 - Baseline results (validation 2010-2016, net of 10 bps)

Produced by `notebooks/02_baseline.ipynb` (code in `src/backtest.py`, `src/baseline.py`, `src/metrics.py`).
Test period 2017-2026 untouched.

## Protocol

Weekly walk-forward on 366 validation weeks. At each Friday close *t*: estimate mu and Sigma with data <= t,
choose long-only weights with a cap w_max, hold them during week t+1 and pay 10 bps per unit of turnover
(turnover measured from the drifted weights). Two optimizer modes:

* **max-Sharpe** in excess of r_f (13-week bill), uses mu and Sigma; falls back to minimum variance if no
  ETF is expected to beat r_f;
* **minimum-variance**, uses Sigma only.

Engine tests: 11 passing (no look-ahead, caps, costs, return of week t+1, optimizer optimality against a
random search).

## Grid search of the baseline (window x cap x mode, ranked by validation Sharpe)

| window | cap | mode | ann. return | ann. vol | Sharpe | max DD | turnover / week |
|---|---|---|---|---|---|---|---|
| **104** | **40 %** | **min-variance** | **7.2 %** | **4.6 %** | **1.52** | **-5.3 %** | **0.01** |
| 104 | 30 % | min-variance | 7.5 % | 5.3 % | 1.39 | -5.6 % | 0.01 |
| 52 | 40 % | min-variance | 6.4 % | 4.7 % | 1.31 | -6.3 % | 0.02 |
| 104 | 40 % | max-Sharpe | 7.3 % | 5.5 % | 1.29 | -5.5 % | 0.06 |
| 52 | 30 % | min-variance | 7.0 % | 5.3 % | 1.28 | -6.8 % | 0.02 |
| 104 | 30 % | max-Sharpe | 7.7 % | 6.1 % | 1.23 | -5.9 % | 0.05 |
| 26 | 40 % | min-variance | 5.9 % | 5.0 % | 1.17 | -7.2 % | 0.04 |
| 26 | 30 % | min-variance | 6.5 % | 5.5 % | 1.16 | -7.3 % | 0.03 |
| 52 | 30 % | max-Sharpe | 5.9 % | 6.9 % | 0.85 | -8.8 % | 0.10 |
| 26 | 30 % | max-Sharpe | 5.1 % | 7.3 % | 0.70 | -12.2 % | 0.15 |
| 52 | 40 % | max-Sharpe | 4.6 % | 6.8 % | 0.69 | -10.0 % | 0.13 |
| 26 | 40 % | max-Sharpe | 2.9 % | 7.5 % | 0.41 | -12.2 % | 0.21 |

Selected baseline: **minimum-variance, 104-week window, cap 40 %** (Sharpe 1.52).

## Baseline vs benchmarks

| strategy | ann. return | ann. vol | Sharpe | max DD | turnover |
|---|---|---|---|---|---|
| Baseline (min-variance, 104w, cap 40 %) | 7.2 % | 4.6 % | 1.52 | -5.3 % | 0.01 |
| Best max-Sharpe (104w, cap 40 %) | 7.3 % | 5.5 % | 1.29 | -5.5 % | 0.06 |
| Spec default (max-Sharpe, 52w, cap 40 %) | 4.6 % | 6.8 % | 0.69 | -10.0 % | 0.13 |
| 1/N | 6.8 % | 9.2 % | 0.75 | -10.7 % | 0.01 |
| SPY buy-and-hold | 12.9 % | 14.4 % | 0.91 | -17.0 % | 0 |

## Forecast metrics (ML level)

Benchmark of the out-of-sample R2: the **historical mean** (expanding window since 2005), as in Campbell and
Thompson (2008). Note: the week 1 report used the 52-week mean as benchmark; we switched to the historical
mean, which is the standard in the literature and does not favour one of the rolling windows.

| estimator | MSE of mu | sign hit rate | R2 oos vs historical mean | realized / predicted variance |
|---|---|---|---|---|
| historical mean (benchmark) | 4.27e-4 | 57.0 % | 0 (benchmark) | 0.68 |
| rolling 26w | 4.44e-4 | 53.3 % | -4.0 % | 0.90 |
| rolling 52w | 4.34e-4 | 54.3 % | -1.6 % | 0.87 |
| rolling 104w | 4.29e-4 | 55.6 % | -0.5 % | 0.70 |

## What we read in these numbers

1. **mu is the weak input.** No rolling window beats the historical mean (R2 < 0 for all three), and the
   sign hit rate (53-56 %) is below the share of positive weeks (57 %), which the historical mean, almost
   always positive, gets for free. Rolling means add noise, not information.
2. **So the best baseline ignores mu.** Minimum-variance (Sigma only) beats max-Sharpe for every window and
   cap; the selected baseline reaches a Sharpe of 1.52 with a turnover of 1 % per week. This is the
   estimation-error problem of the spec, and it sets the bar: the k-NN mu must beat the historical mean
   (R2 > 0) to be worth using in max-Sharpe, and the k-NN Sigma must beat the min-variance baseline.
3. **Longer windows win.** In both modes, 104 weeks beats 52 and 26: less noise, fewer trades, lower costs.
4. **Risk is over-predicted**, not under-predicted, on validation: realized variance is 70-90 % of the
   forecast (2010-2016 was calmer than the windows it was estimated on, which include 2008-2011).
5. **The baseline is not a straw man.** It beats 1/N (Sharpe 1.52 vs 0.75) and SPY (0.91) on a
   risk-adjusted basis, with a drawdown three times smaller than SPY's.
6. **Caveat.** 366 weeks is short: the Sharpe differences between the top configurations are within
   noise. The bootstrap interval (extension) would quantify it.

## Decisions for week 3

* Every model is compared with the **min-variance, 104-week, 40 % cap baseline**, and in both modes
  (max-Sharpe and min-variance): max-Sharpe tests the mu forecast, min-variance isolates the Sigma forecast.
* Every model plugs into the same engine as `estimate(past) -> (mu, Sigma)` and
  `markowitz_strategy(estimate, w_max, min_variance=...)`; R2 is always reported against
  `baseline.expanding_mean_estimate`.
