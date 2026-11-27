"""Step 1 - Exploratory data analysis of the weekly dataset.

Produces the report figures (report/figures/eda_*.png) and the tables /
numbers quoted in the report (report/figures/eda_summary.json, eda_*.csv).

Leakage rule: every statistic that drives a modelling decision (feature
redundancy, VIX thresholds, persistence -> gap g) is computed on the
training weeks only (pool + validation, 2005-2016). The test period is
only described (covariate shift, prevalence), never used to decide.

Run from the project root, after src.data:  python -m src.eda
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from src import config as C

# Palette: one ink for single series, a fixed categorical order, and a
# blue <-> red diverging map through a neutral gray for correlations.
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
BLUE, ORANGE, AQUA, RED = "#257de0", "#f47230", "#00b894", "#ec4647"
SHADE = "#f0f0f0"
DIVERGING = LinearSegmentedColormap.from_list("div", [BLUE, "#efefef", RED])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#cce3fd", "#519ced", "#1757ac", "#0e3d7e"])
SPLIT_COLORS = {"pool": AQUA, "validation": BLUE, "test": ORANGE}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
    "lines.linewidth": 1.2, "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def shade_stress(ax) -> None:
    """Grey bands on the stress episodes and dashed lines at the split dates."""
    for start, end in C.STRESS_PERIODS.values():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=SHADE, zorder=0, lw=0)
    for d in (C.VALID_START, C.TEST_START):
        ax.axvline(pd.Timestamp(d), color=INK2, lw=0.8, ls="--", zorder=3)


def load() -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED_DIR / "weekly_dataset.csv", index_col=0, parse_dates=True)


def train_part(df: pd.DataFrame) -> pd.DataFrame:
    """Weeks available for decisions: pool + validation (the test is never looked at)."""
    return df[df["split"] != "test"]


# ----------------------------------------------------------------- figures --
def fig_asset_growth(df: pd.DataFrame) -> None:
    growth = (1 + df[[f"ret_{a}" for a in C.ASSETS]]).cumprod()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True, sharey=True)
    for ax, a in zip(axes.ravel(), C.ASSETS):
        shade_stress(ax)
        ax.plot(growth.index, growth[f"ret_{a}"], color=BLUE)
        ax.set_yscale("log")
        ax.set_title(a, loc="left")
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(5))
    fig.suptitle("Growth of $1 (total return, log scale), 2005-2026. Grey = stress episodes, "
                 "dashed = validation / test start", x=0.01, ha="left", color=INK)
    fig.savefig(C.FIG_DIR / "eda_asset_growth.png")
    plt.close(fig)


def corr_heatmap(ax, corr: pd.DataFrame, title: str, vmin=-1, vmax=1, cmap=DIVERGING):
    im = ax.imshow(corr.values, cmap=cmap, vmin=vmin, vmax=vmax)
    labels = [c.replace("ret_", "") for c in corr.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            dark = cmap is SEQUENTIAL and v > 0.45
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if dark else INK)
    ax.set_title(title, loc="left")
    return im


def fig_regime_correlations(df: pd.DataFrame) -> dict:
    """Asset correlations in calm vs stressed weeks (VIX terciles, training weeks only)."""
    tr = train_part(df)
    rets = tr[[f"ret_{a}" for a in C.ASSETS]]
    lo, hi = tr["vix_level"].quantile([1 / 3, 2 / 3])
    calm = rets[tr["vix_level"] <= lo].corr()
    stress = rets[tr["vix_level"] >= hi].corr()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    corr_heatmap(axes[0], calm, f"Calm weeks 2005-2016 (VIX <= {lo:.1f})")
    im = corr_heatmap(axes[1], stress, f"Stressed weeks 2005-2016 (VIX >= {hi:.1f})")
    fig.colorbar(im, ax=axes, shrink=0.8, label="correlation of weekly returns")
    fig.savefig(C.FIG_DIR / "eda_regime_correlations.png")
    plt.close(fig)
    off = ~np.eye(len(C.ASSETS), dtype=bool)
    return {"vix_tercile_low": round(lo, 2), "vix_tercile_high": round(hi, 2),
            "mean_pairwise_corr_calm": round(calm.values[off].mean(), 3),
            "mean_pairwise_corr_stress": round(stress.values[off].mean(), 3),
            "calm": calm.round(3).to_dict(), "stress": stress.round(3).to_dict()}


def fig_features(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(C.FEATURES), 1, figsize=(10, 12), sharex=True,
                             gridspec_kw={"hspace": 0.45})
    for ax, f in zip(axes, C.FEATURES):
        shade_stress(ax)
        ax.plot(df.index, df[f], color=BLUE, lw=0.9)
        ax.set_title(f, loc="left", fontsize=9)
    axes[0].text(0, 1.35, "State features (weekly, Friday close). Grey = stress episodes, "
                 "dashed = validation / test start", transform=axes[0].transAxes, color=INK)
    fig.savefig(C.FIG_DIR / "eda_features.png")
    plt.close(fig)


def fig_feature_corr(df: pd.DataFrame) -> pd.DataFrame:
    corr = train_part(df)[C.FEATURES].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = corr_heatmap(ax, corr, "Correlation between state features (2005-2016)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(C.FIG_DIR / "eda_feature_correlation.png")
    plt.close(fig)
    return corr


def fig_persistence(df: pd.DataFrame) -> dict:
    """Autocorrelation of each feature at 1, 13, 26 and 52 weeks.

    High persistence means the closest past states are the most recent weeks,
    which is why the k-NN needs an exclusion gap g.
    """
    lags = [1, 13, 26, 52]
    tr = train_part(df)
    acf = pd.DataFrame({f: [tr[f].autocorr(l) for l in lags] for f in C.FEATURES}, index=lags).T
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(C.FEATURES))
    ax.barh(y, acf[26], color=BLUE, height=0.6)
    ax.set_yticks(y, C.FEATURES)
    ax.invert_yaxis()
    ax.set_xlim(-0.2, 1)
    for i, v in enumerate(acf[26]):
        ax.text(v + 0.02 if v >= 0 else 0.02, i, f"{v:.2f}", va="center", color=INK, fontsize=8)
    ax.set_xlabel("autocorrelation at a 26-week lag")
    ax.set_title("Autocorrelation of the state features at a 26-week lag (2005-2016)", loc="left")
    fig.savefig(C.FIG_DIR / "eda_feature_persistence.png")
    plt.close(fig)
    return acf.round(3).to_dict(orient="index")


def fig_covariate_shift(df: pd.DataFrame) -> dict:
    """Feature distributions per split: k-NN distances suffer if the test
    period sits in a region of the feature space never seen before."""
    order = ["pool", "validation", "test"]
    fig, axes = plt.subplots(1, len(C.FEATURES), figsize=(13, 3.4))
    for ax, f in zip(axes, C.FEATURES):
        data = [df.loc[df["split"] == s, f].values for s in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": INK})
        for patch, s in zip(bp["boxes"], order):
            patch.set_facecolor(SPLIT_COLORS[s])
            patch.set_edgecolor("white")
        ax.set_xticks([1, 2, 3], ["pool", "val", "test"], fontsize=7)
        ax.set_title(f, fontsize=8, loc="left")
    fig.suptitle("Feature distributions by period (whiskers 1.5 IQR, outliers hidden)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_covariate_shift.png")
    plt.close(fig)
    return df.groupby("split")[C.FEATURES].mean().round(3).to_dict(orient="index")


def fig_target_distribution(df: pd.DataFrame) -> None:
    """Histogram of the targets (next-week returns) of each ETF, training weeks only."""
    tr = train_part(df)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharey=True)
    for ax, a in zip(axes.ravel(), C.ASSETS):
        r = 100 * tr[f"ret_next_{a}"].dropna()
        ax.hist(r, bins=60, color=BLUE, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        ax.set_title(a, loc="left")
        ax.set_xlabel("next-week return (%)")
    fig.suptitle("Distribution of the targets (next-week returns), 2005-2016",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_target_distribution.png")
    plt.close(fig)


def prevalence(df: pd.DataFrame) -> pd.DataFrame:
    """Prevalence table by split.

    * share of weeks with a positive next-week return, per ETF;
    * share of stress weeks: VIX above the upper tercile of the training weeks,
      and VIX >= 30 (the regime with few neighbours);
    * share of weeks inside the named stress episodes.
    """
    hi = train_part(df)["vix_level"].quantile(2 / 3)
    in_episode = pd.Series(False, index=df.index)
    for start, end in C.STRESS_PERIODS.values():
        in_episode |= (df.index >= start) & (df.index <= end)
    rows = {}
    for s in ["pool", "validation", "test"]:
        m = df["split"] == s
        d = df[m & df["has_target"]]
        row = {f"pos_{a}_%": 100 * (d[f"ret_next_{a}"] > 0).mean() for a in C.ASSETS}
        row[f"vix_ge_{hi:.1f}_%"] = 100 * (df.loc[m, "vix_level"] >= hi).mean()
        row["vix_ge_30_%"] = 100 * (df.loc[m, "vix_level"] >= 30).mean()
        row["stress_episode_%"] = 100 * in_episode[m].mean()
        row["weeks"] = int(m.sum())
        rows[s] = row
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------ tables --
def return_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return statistics per ETF, training weeks only (2005-2016)."""
    rows = {}
    tr = train_part(df)
    for a in C.ASSETS:
        r = tr[f"ret_{a}"].dropna()
        z = (r - r.mean()) / r.std()
        jb = stats.jarque_bera(r)
        rows[a] = {
            "ann_return_%": 100 * ((1 + r).prod() ** (52 / len(r)) - 1),
            "ann_vol_%": 100 * r.std() * np.sqrt(52),
            "skew": stats.skew(r), "excess_kurtosis": stats.kurtosis(r),
            "jarque_bera_p": jb.pvalue, "worst_week_%": 100 * r.min(),
            "weeks_|z|>4": int((z.abs() > 4).sum()),
        }
    return pd.DataFrame(rows).T


def stationarity(df: pd.DataFrame) -> pd.DataFrame:
    """ADF test per feature (H0: unit root), training weeks only. Used to discuss covariate shift."""
    rows = {}
    tr = train_part(df)
    for f in C.FEATURES:
        stat, p, *_ = adfuller(tr[f].dropna().values, autolag="AIC", result_object=False)
        rows[f] = {"adf_stat": stat, "p_value": p}
    return pd.DataFrame(rows).T


def main() -> None:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    summary: dict = {}
    fig_asset_growth(df)
    fig_features(df)
    summary["regime_correlations"] = fig_regime_correlations(df)
    fcorr = fig_feature_corr(df)
    summary["max_abs_feature_corr"] = (
        fcorr.where(~np.eye(len(fcorr), dtype=bool)).abs().stack().idxmax(),
        round(fcorr.where(~np.eye(len(fcorr), dtype=bool)).abs().max().max(), 3))
    summary["feature_autocorr"] = fig_persistence(df)
    summary["feature_mean_by_split"] = fig_covariate_shift(df)
    fig_target_distribution(df)

    prev = prevalence(df)
    prev.round(1).to_csv(C.FIG_DIR / "eda_prevalence.csv")
    summary["prevalence"] = prev.round(1).to_dict(orient="index")

    rs = return_stats(df)
    rs.round(4).to_csv(C.FIG_DIR / "eda_return_stats.csv")
    st = stationarity(df)
    st.round(4).to_csv(C.FIG_DIR / "eda_stationarity.csv")
    summary["return_stats"] = rs.round(4).to_dict(orient="index")
    summary["adf"] = st.round(4).to_dict(orient="index")
    summary["n_weeks"] = len(df)
    summary["split_counts"] = df["split"].value_counts().to_dict()
    (C.FIG_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
