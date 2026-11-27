"""Step 1 - Data pipeline.

Builds one clean weekly dataset (Friday close) with:
  * weekly total returns of the 6 ETFs, and next-week returns (the targets),
  * the 8 market-state features, all known at the Friday close of week t,
  * the weekly risk-free rate (13-week T-bill) for excess returns.

Source: Yahoo Finance through yfinance. The first download is frozen in
data/raw/yahoo_prices.csv, because Yahoo re-adjusts past prices after every
new dividend: rerunning from the snapshot gives exactly the same numbers.

Run from the project root:  python -m src.data          (use --refresh to re-download)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import config as C


def download_prices(refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted closes of the ETFs and indicators, from the snapshot if present."""
    if C.RAW_PRICES.exists() and not refresh:
        return pd.read_csv(C.RAW_PRICES, index_col=0, parse_dates=True)
    import yfinance as yf
    px = yf.download(C.ASSETS + C.INDICATORS, start=C.DOWNLOAD_START,
                     auto_adjust=True, progress=False)["Close"]
    px = px[C.ASSETS + C.INDICATORS]
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    px.to_csv(C.RAW_PRICES)
    return px


def last_complete_friday(daily: pd.DataFrame) -> pd.Timestamp:
    """Week-end date of the last fully traded week.

    A download made during the week returns a partial week; it is dropped.
    """
    last = daily[C.ASSETS].dropna(how="all").index.max()
    return last if last.weekday() == 4 else last - pd.offsets.Week(weekday=4)


def build_dataset(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Return the weekly dataset and a dict of data-quality facts for the report."""
    report: dict = {}
    daily = download_prices(refresh).sort_index()
    daily = daily[~daily.index.duplicated()]
    report["download_last_day"] = str(daily.index.max().date())
    report["first_valid_day"] = {c: str(daily[c].first_valid_index().date()) for c in daily}
    report["non_positive_etf_prices"] = int((daily[C.ASSETS] <= 0).sum().sum())

    # Weekly sampling: last close of each week ending Friday (handles Friday holidays).
    cutoff = last_complete_friday(daily)
    weekly = daily.resample("W-FRI").last().loc[:cutoff]
    report["last_complete_week"] = str(cutoff.date())

    # Weekly total returns (adjusted prices include dividends).
    rets = weekly[C.ASSETS].pct_change(fill_method=None)
    rets.columns = [f"ret_{a}" for a in C.ASSETS]

    vix, tnx, irx = weekly["^VIX"], weekly["^TNX"], weekly["^IRX"]
    w = C.WINDOW
    feat = pd.DataFrame(index=weekly.index)
    feat["vix_level"] = vix
    feat["vix_change_4w"] = vix - vix.shift(C.VIX_CHANGE_WINDOW)
    feat["spy_mom_12w"] = weekly["SPY"] / weekly["SPY"].shift(w) - 1
    feat["spy_vol_12w"] = rets["ret_SPY"].rolling(w).std()
    feat["corr_spy_ief_12w"] = rets["ret_SPY"].rolling(w).corr(rets["ret_IEF"])
    feat["yield_curve"] = tnx - irx
    feat["short_rate_change_12w"] = irx - irx.shift(w)
    feat["credit_12w"] = (weekly["LQD"] / weekly["LQD"].shift(w)
                          - weekly["IEF"] / weekly["IEF"].shift(w))

    # Targets: next week's return of each ETF (row t holds r_{t+1}).
    targets = rets.shift(-1)
    targets.columns = C.TARGETS

    df = pd.concat([feat, rets, targets], axis=1)
    df["rf_weekly"] = irx / 100 / 52   # known at t, earned during week t+1
    df.index.name = "week_end"

    report["missing_before_dropna"] = {k: int(v) for k, v in
                                       df[C.FEATURES + list(rets)].isna().sum().items()}
    df = df.dropna(subset=C.FEATURES + list(rets.columns) + ["rf_weekly"])
    df = df.loc["2005-01-01":]
    # The last week has no next-week return yet: kept for inference, flagged.
    df["has_target"] = df[C.TARGETS].notna().all(axis=1)

    df["split"] = np.select([df.index <= C.POOL_END, df.index <= C.VALID_END],
                            ["pool", "validation"], default="test")
    report["weeks"] = len(df)
    report["first_week"] = str(df.index.min().date())
    report["last_week"] = str(df.index.max().date())
    report["split_counts"] = df["split"].value_counts().to_dict()
    return df, report


def main() -> None:
    df, report = build_dataset(refresh="--refresh" in sys.argv)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = C.PROCESSED_DIR / "weekly_dataset.csv"
    df.to_csv(out, float_format="%.8g")
    pd.Series(report, dtype=object).to_json(C.PROCESSED_DIR / "data_quality.json", indent=2)
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"saved {out} {df.shape}")


if __name__ == "__main__":
    main()
