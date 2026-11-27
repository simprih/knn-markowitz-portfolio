"""Leakage and integrity checks on the weekly dataset (run: python -m pytest tests -q)."""
import pandas as pd

from src import config as C

DF = pd.read_csv(C.PROCESSED_DIR / "weekly_dataset.csv", index_col=0, parse_dates=True)
PX = pd.read_csv(C.RAW_PRICES, index_col=0, parse_dates=True)


def test_target_is_next_week_return():
    # row t must hold r_{t+1}: the target of week t equals the return of week t+1
    for a in C.ASSETS:
        both = pd.concat([DF[f"ret_next_{a}"], DF[f"ret_{a}"].shift(-1)], axis=1).dropna()
        assert (both.iloc[:, 0] - both.iloc[:, 1]).abs().max() < 1e-9


def test_features_only_use_data_up_to_t():
    # Recompute spy_mom_12w at a few dates from daily prices truncated at t.
    for t in DF.index[[50, 400, 900]]:
        past = PX.loc[:t, "SPY"].resample("W-FRI").last()
        expected = past.iloc[-1] / past.iloc[-1 - C.WINDOW] - 1
        assert abs(DF.loc[t, "spy_mom_12w"] - expected) < 1e-6


def test_weeks_end_on_friday_and_are_sorted():
    assert (DF.index.weekday == 4).all()
    assert DF.index.is_monotonic_increasing and not DF.index.duplicated().any()


def test_no_missing_features_and_only_last_week_lacks_a_target():
    assert DF[C.FEATURES].notna().all().all()
    assert (~DF["has_target"]).sum() == 1 and not DF["has_target"].iloc[-1]


def test_splits_are_chronological():
    order = DF["split"].map({"pool": 0, "validation": 1, "test": 2})
    assert order.is_monotonic_increasing
