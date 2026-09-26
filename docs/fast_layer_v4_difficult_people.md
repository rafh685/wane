# Fast layer v4 on difficult people, and the draw-length fixes

Status: simulation, 27 Sept 2026, branch `claude/per-puff-nn`. Relative comparisons on synthetic people only.

## The brief (Rafael)
1. Draw length: long draws deliver more nicotine than the dose the device sets, and the daily budget did not
   count that. Try the fixes, keep whichever version gives the best results.
2. Judge the engine on difficult people with irregular habits, like Lucía, not on people with a stable routine.

## Difficult people (`experiments/hard_people.py`)
- Habits: every person's daily amount swings from day to day (profile noise x1.5), and compensation goes 57 %
  into longer draws and 43 % into more puffs. That is the LSBU split already in `profiles.py`; the earlier
  per-puff tests used 40 / 60, which undersold long draws.
- Selection: 200 synthetic people per set, the top quarter by expected relapse risk on the flat taper, measured
  with selection seeds that are never used for evaluation (so people who were merely unlucky cannot flatter the
  other controllers).
- Development set: 50 people (pool seed 41), flat risk 0.29 on average against a pool median of 0.12.
- Test set: 50 people (pool seed 51, flat risk 0.31) plus Lucía and Karim.
- Scenarios: `hard` (irregular days, 57 % long draws), `hard_long` (80 % long draws, stress test),
  `hard_change` (as hard, plus a new routine at taper week 6).

## Draw-length fixes (development set, `hard_people_eval.py fixes`)
| Version | Expected risk, hard | hard_long | vs v4 |
|---|---|---|---|
| v4 (budget counts dose settings) | 0.170 | 0.164 | |
| budget counts delivered nicotine | 0.173 | 0.170 | +0.005 [+0.001, +0.009], worse for 17, better for 4 |
| delivery capped at 1.5x the person's usual draw | 0.174 | 0.167 | +0.004 [+0.002, +0.006] |
| cap at 2x | 0.171 | 0.165 | +0.001 [+0.000, +0.003] |
| delivered + cap 1.5x | 0.177 | 0.174 | +0.009 [+0.005, +0.013] |
| delivered + cap 2x | 0.173 | 0.171 | +0.005 [+0.002, +0.010] |

**Every fix made difficult people worse, so v4 is kept as it was.** In the simulator a long draw is a sign of
withdrawal, so the few percent of extra nicotine it brings act as a safety valve; taking them away hits people
exactly when they struggle. What the simulator does not model is learning: a real person may learn that pulling
harder pays, and slowly draw longer. That risk cannot show up here, so both fixes stay in the code as switches
(`OnTheSpot(accounting="delivered")`, `OnTheSpot(cap_x=...)`, off by default) for the pilot to test.

## Settings retuned for difficult people (development set, `hard_people_eval.py retune`)
32 combinations of relief boost, relief gap threshold (2 or 3 h), pre-gap cut and pre-gap window (2 or 3 h).
Best against v4's settings: -0.0009 [-0.0030, +0.0011]. The rule fixed before running was to keep v4's settings
unless the gain was clear; it was not. **The final version is v4, unchanged.**

## Final test on difficult people (test set, once, `hard_people_eval.py test`)
52 people x 3 scenarios x 3 seeds. Expected relapse risk:

| | hard | hard_long | hard_change |
|---|---|---|---|
| Flat (weaker bottle) | 0.334 | 0.346 | 0.365 |
| v3 | 0.220 | 0.228 | 0.243 |
| **v4 (final)** | **0.179** | **0.185** | **0.196** |
| v4 without the network | 0.178 | 0.185 | 0.198 |

- **v4 vs flat**: -0.162 [-0.182, -0.143], about 46 % lower in every scenario, **better for all 52 people, worse
  for none**. Success (reached zero without relapse) +0.18 [+0.12, +0.24].
- **v4 vs v3**: -0.044 [-0.053, -0.035], better for 49 of 52; success +0.06 [+0.02, +0.10].
- **Network vs no network**: no difference overall (-0.0004); slightly better after a routine change (0.196 vs
  0.198), and no empty puffs (0.13 % without the network).
- **Lucía**: flat 0.48, v3 0.30, v4 0.24. **Karim**: 0.33, 0.20, 0.16.

v4's gain is larger for difficult people (-0.16) than for the general population (-0.06 to -0.07,
`docs/fast_layer_v4.md`), and the ordinary-population results in that document already cover v4 unchanged, so
no separate no-harm run was needed.

## What is left for difficult people
Their risk is still 0.18 to 0.20 with v4, twice the general population's. The fast layer's settings are
saturated (the retune found nothing), so the next lever is the slow layer: a taper speed per person, and
re-sizing the daily budget when a person's puff count changes for good (the P12-023 case).

## Limits
Same as `docs/fast_layer_v4.md`: our own withdrawal model, no learning or habit reinforcement in the simulated
person, synthetic people only.
