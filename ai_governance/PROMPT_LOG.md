# PROMPT_LOG - high-stakes AI interactions

Goal: 4 to 6 entries, at least one per student. We only log the interactions that shaped a real
decision (leakage, statistical tests, choice of model or metric), with the exact prompt, what the AI
gave back, and what we kept or changed. The idea is also to show how our prompts got more precise over time.

Template for each entry:

```
## #N - <decision> (<date>, <student>, <tool and model>)
Prompt (verbatim):
Constraints given: schema / leakage boundary / assumptions to check / fallback
AI output (summary):
What we verified and how:
Kept / changed:
```

---

## #1 - Week 1 data pipeline, EDA and spec review (2026-11-25 to 27, Pierre Rançon, Claude)

Prompts (verbatim, in French): "on fait la week 1 ?" (after sharing the course PDFs), then, with the
week 1 report attached, "voici le premier compte rendu qu'est ce que tu en penses ?" and "on garde
ce projet comme idée mais je suis ouvert au correction qu'il faut faire".

Constraints given: none, apart from the spec itself. This was a bare prompt: no data schema, no leakage
boundary, no assumptions to check. So the output needed the checks below, and the next prompts have to
be constrained (see #2, to write).

AI output (summary): a review of the report (8 issues), the corrected LaTeX report, `src/data.py`
(Yahoo download frozen as a snapshot, weekly Friday sampling, 8 features, next-week targets),
`src/eda.py` (6 figures, return statistics, ADF tests) and `tests/test_data.py` (5 tests).

What we verified and how: *(to fill by the reviewer: rerun everything, read every function, check the
tests and at least 3 numbers by hand against the Yahoo snapshot)*

Kept / changed: *(to fill)*

---

## #2 - Week 2 backtest engine and baseline (2026-11-27, Pierre Rançon, Claude)

Prompt (verbatim, in French): "oui je veux bien, je veux un code simple qui ressemble à ce qu'on a
fait aux différents labs, on suit toujours la même ligne", in answer to the proposal to build the
walk-forward engine described in the week 1 specification.

Constraints given: the specification (section 4 to 6: weights decided with data <= t, max-Sharpe in
excess of r_f, cap, 10 bps on turnover, rolling Ledoit-Wolf baseline, chronological validation).
Still not an explicit prompt: next time, write the leakage boundary and expected tests in the prompt.

AI output (summary): `src/backtest.py`, `src/baseline.py`, `src/metrics.py`,
`tests/test_backtest.py`, `notebooks/02_baseline.ipynb`, `report/week2_notes.md`.

What we verified and how: optimizer bug found and fixed (AI_AUDIT #1); *(to fill: read every
function, recompute one week of turnover and net return by hand)*

Kept / changed: after Simon's review (PROMPT_LOG #3), the R2 benchmark became the historical mean
(`baseline.expanding_mean_estimate`) and the min-variance mode was added to the baseline grid
(12 configurations); selected baseline: min-variance, 104 weeks, cap 40 % (validation Sharpe 1.52).

---

## #3 - Spec review, scope, EDA audit and week 2 code review (2026-11-23 to 27, Simon Prihnenko, Claude)

Prompts (verbatim, in French), from the vaguest to the most constrained:

1. "on a choisis un sujet avec mon équipe et travaillé un petit peu mais je veux continuer ici avec toi,
   regarde ce qu'on a et dis moi ce que tu en penses" (draft spec attached).
2. "je veux deja que le projet soit assez simple ppour qu'on puisse le faire nous meme, avec ce qu'on a
   fait en TD, est ce que c'est la cas la ?"
3. "je veux qu'on respecte tout ce qui est demandé pour les lab pour etre sur d'avoir tous les critères
   requis" (LAB 4, 5 and 6 attached).
4. After numbers showed up in the report: "d'ou viennent les résultats que tu donnes genre la corrélation
   etc ???? tu donnes des ciffres meme avant d'avoir fait quelqu'onque code etc ???? [...] tu les as
   calculés toi meme ?????"
5. With the project archive attached: "il faut aller chercher TOUS les points du barème ; [...] je t'ai
   mis les notebook etc que mon camarades m'a donné, revois ca aussi".

Constraints given: none at first (prompt 1). Then I added them one at a time: stay within the methods
seen in the labs (prompt 2), cover every requirement of LAB 4 to 6 (prompt 3), no number in the report
without the code that produced it (prompt 4), cover the whole rubric (prompt 5).

AI output (summary): a review of the draft spec (missing ML formulation, baseline and validation split),
a simpler version of the method, the week 1 report rewritten in LaTeX, a review of `src/data.py` and
`src/eda.py`, a new `src/eda.py` (decision statistics on 2005-2016 only, target distribution, prevalence
table), a review of Pierre's week 2 engine (R2 benchmark, missing min-variance baseline), and a rewrite of
the markdown notes (README, EDA findings, governance files) in a more natural style.

What I verified and how:
* I asked where every EDA number came from: they come from `src/eda.py`, run on the frozen snapshot.
  I reran `python -m src.data`, `python -m src.eda` and `pytest`: 1,142 weeks, same numbers, all tests pass.
* Pierre recomputed the 2005-2016 statistics on his side (rho = 0.86, SPY-IEF -0.01 / -0.39,
  autocorrelation 0.80, VIX thresholds 14.0 / 19.6, extreme weeks 15 / 6 / 6 %) and got the same values.
* Week 2: I reran the 11 tests and the baseline grid search, and got the same Sharpe ratios (1.29 / 0.69 / 1.31).
* I checked every claim of the AI against the course documents (LAB 4 to 6, guidelines, governance policy).

Kept / changed:
* Kept: the 2005-2016 EDA statistics, the target distribution and prevalence, the report structure,
  the R2 benchmark fix, and the min-variance mode in the baseline grid search.
* Rejected: the AI said no model outside the course was required. The guidelines (slide 18) say the
  opposite, so we kept the HMM. Logged in AI_AUDIT.
* Fixed after team review: the LQD excess kurtosis is 14 on 2005-2016 (because of 2008), not 41
  (full sample, because of 2020).
* What I take from this: my first prompt was much too vague. For the k-NN module, I'll state from the
  start the data format (`past` up to t, with the targets of the last row hidden), the leakage rules
  (neighbours with s + 1 <= t and t - s > g, scaler fitted on past rows only) and the test to run
  (every neighbour must be eligible).
