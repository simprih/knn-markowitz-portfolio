"""Checks on the walk-forward engine (run: python -m pytest tests -q)."""
import numpy as np
import pandas as pd

from src import config as C
from src.backtest import (markowitz_strategy, max_sharpe_weights, min_variance_weights,
                          walk_forward)
from src.baseline import equal_weight, rolling_estimate

DF = pd.read_csv(C.PROCESSED_DIR / "weekly_dataset.csv", index_col=0, parse_dates=True)
START, END = "2012-01-01", "2012-06-30"     # a short window keeps the tests fast


def test_strategy_never_sees_the_future():
    seen = []

    def spy(past):
        t = past.index[-1]
        assert past[C.TARGETS].iloc[-1].isna().all()           # r_{t+1} hidden
        assert past[C.TARGETS].iloc[:-1].notna().all().all()   # r_{s+1}, s < t, known
        assert past.index.max() == t and (past.index <= t).all()
        seen.append(t)
        return equal_weight(past)

    res = walk_forward(DF, spy, START, END)
    assert list(res.index) == seen


def test_weights_are_long_only_capped_and_fully_invested():
    res = walk_forward(DF, markowitz_strategy(rolling_estimate(52), w_max=0.3), START, END)
    W = res[[f"w_{a}" for a in C.ASSETS]].to_numpy()
    assert np.allclose(W.sum(axis=1), 1)
    assert (W >= -1e-9).all() and (W <= 0.3 + 1e-9).all()


def test_net_return_is_gross_minus_costs_on_turnover():
    res = walk_forward(DF, equal_weight, START, END, cost_bps=10)
    assert np.allclose(res["net"], res["gross"] - 10 / 1e4 * res["turnover"])
    assert np.isclose(res["turnover"].iloc[0], 1.0)              # initial purchase


def test_gross_return_is_the_return_of_week_t_plus_1():
    res = walk_forward(DF, equal_weight, START, END)
    # equal weights: gross = average of the 6 returns of the following week
    expected = DF[[f"ret_{a}" for a in C.ASSETS]].shift(-1).loc[res.index].mean(axis=1)
    assert np.allclose(res["gross"], expected)


def test_max_sharpe_falls_back_to_min_variance_when_nothing_beats_cash():
    Sigma = np.diag([0.01, 0.02, 0.03])
    w = max_sharpe_weights(np.array([-0.01, -0.02, 0.0]), Sigma, rf=0.001, w_max=1)
    assert np.allclose(w, min_variance_weights(Sigma, 1), atol=1e-6)


def test_optimizers_beat_a_random_search():
    # Weekly variances are ~1e-4: the optimizer must not stop at its starting point.
    mu, Sigma = rolling_estimate(52)(DF.loc[:"2012-06-29"])
    rng = np.random.default_rng(C.SEED)
    cands = [x for x in rng.dirichlet(np.full(len(mu), 0.5), 20000) if x.max() <= 0.4]
    w = min_variance_weights(Sigma, 0.4)
    assert w @ Sigma @ w <= min(x @ Sigma @ x for x in cands) + 1e-12
    sharpe = lambda x: (x @ mu) / np.sqrt(x @ Sigma @ x)
    assert sharpe(max_sharpe_weights(mu, Sigma, 0.0, 0.4)) >= max(sharpe(x) for x in cands) - 1e-9
