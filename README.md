# Regime-Aware Portfolio Allocation (k-NN + Markowitz)

ESILV ML Project 2026, Financial Engineering. Team: Simon Prihnenko, Sacha Audouy, Pierre Rançon.

The idea is simple: each week, we look for the k past weeks whose market conditions are the closest
to today's, and we use what happened right after them to estimate expected returns and covariances
for 6 ETFs (SPY, EEM, IEF, LQD, GLD, VNQ). These estimates go into a Markowitz optimizer, and we
compare the result with a classic rolling-window Markowitz, in a weekly backtest from 2005 to 2026.

## Layout

```
requirements.txt          pinned versions (Python 3.13)
src/config.py             universe, features, split dates, paths
src/data.py               step 1: Yahoo -> data/processed/weekly_dataset.csv
src/eda.py                step 1: EDA figures and tables -> report/figures/
tests/test_data.py        leakage and integrity checks
src/backtest.py           step 2: walk-forward engine, capped Markowitz (max-Sharpe / min-variance), costs
src/baseline.py           step 2: rolling Ledoit-Wolf baseline, historical mean, 1/N, SPY
src/metrics.py            step 2: portfolio metrics (Sharpe in excess of rf) and forecast metrics (R2 oos)
tests/test_backtest.py    no look-ahead, weights, costs, optimizer checks
notebooks/02_baseline.ipynb  week 2: baseline grid search and first validation results
data/raw/yahoo_prices.csv frozen download (Yahoo re-adjusts past prices after each dividend)
report/week1/             week 1 specification (LaTeX source + PDF)
report/eda_findings.md    data quality and EDA notes
report/week2_notes.md     week 2 results and decisions (validation 2010-2016)
ai_governance/            CONTRIBUTIONS.md, PROMPT_LOG.md, AI_AUDIT.md
```

## Run

```bash
pip install -r requirements.txt
python -m src.data        # reads the snapshot; add --refresh to download again from Yahoo
python -m src.eda
python -m pytest tests -q
jupyter nbconvert --to notebook --execute notebooks/02_baseline.ipynb   # week 2
```

Everything runs from the project root with relative paths, and the random seed is set in `config.SEED`.
