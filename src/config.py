"""Project-wide constants: asset universe, data sources, dates and paths.

Every other module imports from here, so a change of universe or period
is made in one place only.
"""
from pathlib import Path

# Paths are relative to the project root, so the archive runs anywhere.
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "report" / "figures"
RAW_PRICES = RAW_DIR / "yahoo_prices.csv"   # frozen snapshot of the Yahoo download

SEED = 42

# Portfolio universe: 6 US-listed ETFs (Yahoo adjusted close = total return).
ASSETS = ["SPY", "EEM", "IEF", "LQD", "GLD", "VNQ"]
# Market indicators (Yahoo): VIX, 10-year yield, 13-week T-bill yield (in %).
INDICATORS = ["^VIX", "^TNX", "^IRX"]

# GLD starts on 18 Nov 2004; extra history lets the 12-week windows fill before 2005.
DOWNLOAD_START = "2004-06-01"

# Chronological split (week-ending Fridays).
POOL_END = "2009-12-31"                          # 2005-2009: neighbour pool only (includes 2008)
VALID_START, VALID_END = "2010-01-01", "2016-12-31"   # choice of k and gap
TEST_START = "2017-01-01"                        # final evaluation, run once

WINDOW = 12            # weeks, for momentum, volatility, correlation, rate change, credit
VIX_CHANGE_WINDOW = 4

FEATURES = [
    "vix_level",
    "vix_change_4w",
    "spy_mom_12w",
    "spy_vol_12w",
    "corr_spy_ief_12w",
    "yield_curve",
    "short_rate_change_12w",
    "credit_12w",
]
TARGETS = [f"ret_next_{a}" for a in ASSETS]

# Stress episodes used in the EDA and in the evaluation (objective O3).
STRESS_PERIODS = {
    "GFC 2008": ("2008-09-01", "2009-03-31"),
    "Q4 2018": ("2018-10-01", "2018-12-31"),
    "Covid 2020": ("2020-02-15", "2020-04-15"),
    "Rates 2022": ("2022-01-01", "2022-12-31"),
}
