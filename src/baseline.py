"""Step 2 - Baseline and benchmarks.

* Rolling-window Markowitz (the baseline): mu and Sigma estimated on the last
  `window` weeks of returns, Sigma shrunk with Ledoit-Wolf, then the same
  optimizer as every other model.
* Equal weight 1/N and SPY buy-and-hold (benchmarks, no estimation).
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from src import config as C
from src.backtest import RET_COLS


def rolling_estimate(window=52):
    """Estimator using the last `window` weekly returns known at t.

    Uses the columns ret_* (r_s for s <= t), which are all known at the close
    of week t: no target column is touched, so no look-ahead is possible.
    """
    def estimate(past):
        R = past[RET_COLS].iloc[-window:].to_numpy()
        mu = R.mean(axis=0)
        Sigma = LedoitWolf().fit(R).covariance_
        return mu, Sigma
    return estimate


def equal_weight(past):
    """1/N: the same weight in each ETF, rebalanced every week."""
    return np.full(len(C.ASSETS), 1 / len(C.ASSETS))


def spy_buy_and_hold(past):
    """100 % SPY."""
    w = np.zeros(len(C.ASSETS))
    w[C.ASSETS.index("SPY")] = 1.0
    return w


def expanding_mean_estimate(past):
    """Historical mean of all returns known at t (expanding window).

    This is the benchmark forecast of Campbell and Thompson (2008) for the
    out-of-sample R2: a model is useful only if it beats the plain average of
    everything observed so far.
    """
    R = past[RET_COLS].to_numpy()
    return R.mean(axis=0), LedoitWolf().fit(R).covariance_
