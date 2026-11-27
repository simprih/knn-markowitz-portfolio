"""Step 2 - Walk-forward backtest engine and Markowitz optimizer.

Every strategy is a function `weights_fn(past) -> w`:
  * `past` is the dataset up to the Friday close of week t (rows <= t),
  * `w` is the vector of portfolio weights held during week t+1.

The engine guarantees that a strategy cannot see the future: the next-week
returns (TARGETS) of the last row t are hidden, because r_{t+1} is not known at t.

Markowitz strategies are built from an estimator `estimate(past) -> (mu, Sigma)`
with `markowitz_strategy`, so the baseline and the k-NN only differ by their
estimator (same optimizer, same cap, same costs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src import config as C

RET_COLS = [f"ret_{a}" for a in C.ASSETS]   # r_t, known at the close of week t


# --------------------------------------------------------------- optimizer --
def max_sharpe_weights(mu, Sigma, rf, w_max):
    """Long-only, fully invested portfolio with the highest Sharpe ratio.

    max  w'(mu - rf) / sqrt(w' Sigma w)   s.t.  sum(w) = 1,  0 <= w_i <= w_max

    If no asset is expected to beat the risk-free rate, the maximum Sharpe
    portfolio is not meaningful, so we fall back to the minimum-variance one.
    """
    excess = np.asarray(mu) - rf
    if excess.max() <= 0:
        return min_variance_weights(Sigma, w_max)

    def neg_sharpe(w):
        return -(w @ excess) / np.sqrt(w @ Sigma @ w)

    return _solve(neg_sharpe, len(mu), w_max)


def min_variance_weights(Sigma, w_max):
    """Long-only, fully invested portfolio with the lowest variance.

    Weekly variances are tiny (~1e-4), below the optimizer's default stopping
    tolerance, so Sigma is rescaled by its average variance first. This does
    not change the solution, only the units.
    """
    S = Sigma / np.mean(np.diag(Sigma))
    return _solve(lambda w: w @ S @ w, len(Sigma), w_max)


def _solve(objective, n, w_max):
    w0 = np.full(n, 1 / n)                        # start from equal weights
    res = minimize(objective, w0, method="SLSQP",
                   bounds=[(0, w_max)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-12, "maxiter": 500})
    w = np.clip(res.x, 0, w_max)
    return w / w.sum()


def markowitz_strategy(estimate, w_max=0.40, min_variance=False):
    """Turn an estimator `estimate(past) -> (mu, Sigma)` into a strategy."""
    def weights_fn(past):
        mu, Sigma = estimate(past)
        if min_variance:
            return min_variance_weights(Sigma, w_max)
        rf = past["rf_weekly"].iloc[-1]          # risk-free rate known at t
        return max_sharpe_weights(mu, Sigma, rf, w_max)
    return weights_fn


# ------------------------------------------------------------- walk-forward --
def walk_forward(df, weights_fn, start, end, cost_bps=10):
    """Weekly walk-forward backtest between `start` and `end` (inclusive).

    For each week t: weights are decided with data <= t, executed at the
    Friday close of t, and earn the returns of week t+1. Costs are charged on
    turnover, i.e. the trades needed to go from the drifted old weights to
    the new ones.

    Returns a DataFrame indexed by t with the weights, gross and net return of
    week t+1, turnover and risk-free rate.
    """
    weeks = df.loc[start:end].index
    weeks = weeks[df.loc[weeks, "has_target"]]
    rows = []
    w_prev = None
    for t in weeks:
        past = df.loc[:t].copy()
        past.loc[t, C.TARGETS] = np.nan          # r_{t+1} is not known at t
        w = np.asarray(weights_fn(past), dtype=float)

        r_next = df.loc[t, C.TARGETS].to_numpy(dtype=float)
        turnover = np.abs(w).sum() if w_prev is None else np.abs(w - w_prev).sum()
        gross = w @ r_next
        net = gross - cost_bps / 1e4 * turnover
        rows.append({"week": t, "gross": gross, "net": net, "turnover": turnover,
                     "rf": df.loc[t, "rf_weekly"],
                     **{f"w_{a}": x for a, x in zip(C.ASSETS, w)}})
        # weights drift with returns during week t+1 before the next rebalance
        w_prev = w * (1 + r_next) / (1 + gross)
    return pd.DataFrame(rows).set_index("week")


def walk_forward_forecasts(df, estimate, start, end, w_max=0.40):
    """Forecasts of an estimator, week by week, for the ML-level metrics.

    Returns two DataFrames indexed by t:
      * mu_hat: the 6 forecasts of next-week returns, and the realized r_{t+1};
      * var: predicted vs realized variance of the minimum-variance portfolio
        built from Sigma_hat (realized variance = squared portfolio return).
    """
    weeks = df.loc[start:end].index
    weeks = weeks[df.loc[weeks, "has_target"]]
    mu_rows, var_rows = [], []
    for t in weeks:
        past = df.loc[:t].copy()
        past.loc[t, C.TARGETS] = np.nan
        mu, Sigma = estimate(past)
        r_next = df.loc[t, C.TARGETS].to_numpy(dtype=float)
        w = min_variance_weights(Sigma, w_max)
        mu_rows.append({"week": t, **{f"mu_{a}": m for a, m in zip(C.ASSETS, mu)},
                        **{f"r_{a}": r for a, r in zip(C.ASSETS, r_next)}})
        var_rows.append({"week": t, "pred_var": w @ Sigma @ w, "real_var": (w @ r_next) ** 2})
    return (pd.DataFrame(mu_rows).set_index("week"),
            pd.DataFrame(var_rows).set_index("week"))
