"""Step 2 - Evaluation metrics.

Portfolio metrics (finance level) on weekly returns, and forecast metrics
(ML level) on the mu_hat and Sigma_hat of an estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


# ------------------------------------------------------------ portfolio --
def portfolio_metrics(res: pd.DataFrame, col: str = "net") -> dict:
    """Annualized return and volatility, Sharpe ratio, max drawdown, turnover.

    `res` is the output of backtest.walk_forward. The Sharpe ratio uses
    returns in excess of the risk-free rate, annualized with sqrt(52).
    """
    r = res[col]
    excess = r - res["rf"]
    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return {
        "ann_return": wealth.iloc[-1] ** (WEEKS_PER_YEAR / len(r)) - 1,
        "ann_vol": r.std() * np.sqrt(WEEKS_PER_YEAR),
        "sharpe": excess.mean() / excess.std() * np.sqrt(WEEKS_PER_YEAR),
        "max_drawdown": drawdown.min(),
        "avg_turnover": res["turnover"].iloc[1:].mean(),   # week 1 = initial buy
    }


# ------------------------------------------------------------- forecasts --
def forecast_metrics(mu_df: pd.DataFrame, var_df: pd.DataFrame, assets, bench_mu=None) -> dict:
    """MSE and out-of-sample R2 of mu_hat, sign hit rate, variance forecast error.

    R2_oos = 1 - SSE(model) / SSE(benchmark), with the historical mean
    (expanding window, baseline.expanding_mean_estimate) as the benchmark
    forecast (Campbell and Thompson, 2008). R2 > 0 means the model
    forecasts next-week returns better than the benchmark.
    """
    mu = mu_df[[f"mu_{a}" for a in assets]].to_numpy()
    r = mu_df[[f"r_{a}" for a in assets]].to_numpy()
    out = {"mse_mu": np.mean((r - mu) ** 2),
           "hit_rate": np.mean(np.sign(mu) == np.sign(r))}
    if bench_mu is not None:
        b = bench_mu[[f"mu_{a}" for a in assets]].to_numpy()
        out["r2_oos"] = 1 - ((r - mu) ** 2).sum() / ((r - b) ** 2).sum()
    # variance forecast: mean squared error and ratio realized / predicted
    out["var_mse"] = np.mean((var_df["real_var"] - var_df["pred_var"]) ** 2)
    out["var_ratio"] = var_df["real_var"].mean() / var_df["pred_var"].mean()
    return out
