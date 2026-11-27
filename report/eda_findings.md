# Week 1 - Data quality and EDA findings

All the numbers below come from `python -m src.data` and `python -m src.eda` (outputs in `report/figures/`).
Anything we use to make a modelling decision is computed on the training weeks only (pool + validation,
2005-2016). We only look at the test period to describe it, never to choose anything.

## Dataset

* 1,142 weekly observations (Friday close), from 7 Jan 2005 to 20 Nov 2026. The snapshot was downloaded on 27 Nov 2026.
* Split: neighbour pool 2005-2009 (260 weeks), validation 2010-2016 (366 weeks), test 2017-2026 (516 weeks).
* Assets: SPY, EEM, IEF, LQD, GLD, VNQ. We use Yahoo's adjusted close, so returns include dividends.
  Indicators: ^VIX, ^TNX, ^IRX.
* 8 features (listed in `config.FEATURES`). The targets are the next-week returns of each ETF, and the
  risk-free rate is ^IRX / 100 / 52.

## Data quality

| Check | What we found | What we did |
|---|---|---|
| Last week (23-27 Nov) | not finished at download time | dropped, last full week = 20 Nov |
| Start dates | GLD starts on 18 Nov 2004, VNQ on 29 Sep 2004, the others earlier | dataset starts on 7 Jan 2005, once the 12-week windows are filled |
| Duplicated dates / non-positive prices | 0 / 0 | nothing to fix |
| Missing values | only from the rolling windows, the late listing of GLD and VNQ (all before 2005) and the last target | rows dropped; the last week is kept for inference |
| Yahoo re-adjusts past prices after each dividend | results would change from one run to the next | download frozen in `data/raw/yahoo_prices.csv` |

## What we learned from the data (2005-2016)

1. The stock-bond relation changes with the regime (`eda_regime_correlations.png`). The correlation of
   weekly SPY and IEF returns is -0.01 in calm weeks (VIX <= 14.0) but -0.39 in stressed weeks
   (VIX >= 19.6). Other pairs move a lot too: SPY-VNQ goes from 0.56 to 0.79 and SPY-EEM from 0.66 to 0.87.
   This is the main motivation of the project.
2. Returns are not Gaussian (`eda_return_stats.csv`). Jarque-Bera rejects normality for all 6 ETFs, and
   excess kurtosis goes from 0.9 (IEF) to 14 (LQD, mostly because of 2008). We keep these outliers because
   they are real market events. It's also why we shrink the covariance with Ledoit-Wolf and why we look at
   drawdowns, not just the Sharpe ratio.
3. Two features overlap: vix_level and spy_vol_12w have a correlation of 0.86 (`eda_feature_correlation.png`).
   In the Euclidean distance, volatility is basically counted twice, so the feature ablation will test
   removing one of them.
4. The features are persistent (`eda_feature_persistence.png`). At a 26-week lag, the autocorrelation is
   still 0.80 for the yield curve and 0.35 for the short-rate change. So if the k-NN can search all past
   weeks, it will mostly pick the last few weeks. That's why we skip the most recent g weeks.
5. Targets and prevalence (`eda_target_distribution.png`, `eda_prevalence.csv`). Between 51 % and 63 % of
   next-week returns are positive, depending on the ETF and the period. Stressed weeks (VIX >= 19.6) are
   44 % of the pool, 26 % of validation and 34 % of test, but really extreme weeks (VIX >= 30) are rare:
   15 %, 6 % and 6 %. Those are the weeks where the k-NN will have the fewest good neighbours.
6. The test period doesn't look like the past (`eda_covariate_shift.png`, `eda_stationarity.csv`). The yield
   curve is not stationary on 2005-2016 (ADF p = 0.41): its mean is 2.28 in validation but only 0.47 in test,
   and the SPY-IEF correlation averages -0.43 in validation against -0.06 in test. In practice, some test
   weeks may not have close neighbours at all, so we'll track the average neighbour distance and try
   dropping some features.
