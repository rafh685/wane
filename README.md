# Wane: adaptive nicotine tapering

Every quit product tapers how often you vape. Wane tapers how much nicotine is in each puff, from measured behaviour, so the user never has to decide when to step down.

This repo is the **taper engine** and a simulation: synthetic vapers run through a traditional fixed ladder and through the Wane engine, side by side, all the way to zero nicotine.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas streamlit plotly pyreadstat
.venv/bin/streamlit run app.py
```

Numbers only: `.venv/bin/python simulate.py`. Refit the engine's weights: `.venv/bin/python fit.py`.

## What is in here

| File | What |
|---|---|
| `profiles.py` | One behavioural model, five named people. Hourly routines (work bans, evening-only, shift patterns), cues (coffee, alcohol, stress), compensation split into more puffs and longer puffs, acute withdrawal after a cut, chronic craving at low intake, probabilistic relapse. Output is puff timestamps per day, the same shape a counter app or a metering device produces. |
| `population.py` | Random people drawn from the same ranges, for fitting on one population and testing on another. |
| `engine.py` | `FixedTaper` (cut x % per step), `AdaptiveTaper` (hand-set weights) and `FittedTaper` (weights learned by `fit.py`). The JITAI loop: features from the last 7 days of puffs, risk of relapse next week, estimated craving, then cut / half cut / hold. Per-person rate that learns. Below 3 mg/ml, absolute steps so the last step to zero is no bigger than the others. The engine can only lower or hold, never raise. |
| `simulate.py` | Same people, both engines, N runs with noise. Decision period 7 / 3 / 2 / 1 days. Manual delivery (the user has to switch liquid, and sometimes does not) or automatic. Outcomes: relapsed, stalled on the ladder, off nicotine, weeks to zero. |
| `fit.py` | Fits a craving model (linear) and a relapse-next-week model (logistic) on population A, picks thresholds on A, reports on population B. Writes `engine_weights.json`. |
| `app.py` | Streamlit demo: play through the weeks, fixed vs Wane, per-person view with every decision and the measured features behind it, engine and delivery switches. |
| `experiments/` | Things we tried and kept as evidence, including the fast-signal experiment that did not earn its place and the speed-versus-safety chart. |
| `docs/model_assumptions.md` | Every formula, what it claims, and whether each constant is sourced, calibrated or a guess. |
| `data/lsbu/` | Real-world compensation data, see below. |

## Real data

The behavioural model is calibrated against the one public dataset of real vapers under a nicotine cut: Dawkins et al. (2018), *'Real-world' compensatory behaviour with low nicotine concentration e-liquid*, Addiction 113(10), open data CC BY 4.0, DOI [10.18744/LSBU.002952](https://doi.org/10.18744/LSBU.002952). Twenty experienced vapers, one week each at 18 and 6 mg/ml with every puff logged by the device. It fixed the size of compensation (total puffing x1.47 for a 67 % cut, split between more puffs and longer puffs), the size of the craving response, and the fact that a single cut on its own does not make people relapse. What it cannot tell us, night puffs, time to first puff, weekend patterns and months of tapering, remains synthetic until a pilot.

## What the simulation says so far

- A gentle ladder works for most people. Its main failure is not relapse but stalling: steps the user never applies. Removing the decision from the user, automatic delivery, is the single largest effect in the simulation.
- The people the ladder loses are predictable from their puff pattern a week ahead (night puffs and time to first puff carry most of the signal). Treating them differently is what the engine is for.
- Deciding daily is not better than weekly unless the signals are faster than the decisions. Small steps are not free, withdrawal accumulates.

## Honest status

The engine is evaluated on synthetic people who obey the assumptions it was written with. The calibration above anchors the size of compensation and craving; everything else is population ranges from the literature or stated guesses (all listed in `docs/model_assumptions.md`). A supervised pilot with adult volunteers logging puffs, adaptive arm versus fixed arm, is what tests the thesis.

Built for IE University's Tech Venture Bootcamp, Sept to Oct 2026.
