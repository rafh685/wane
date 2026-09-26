# Fast layer v3: the dose decided at every puff

Status: simulation prototype, 26 Sept 2026, branch `claude/per-puff-nn`. Every number below comes from synthetic
people in `puffsim.py`. They compare controllers with each other; they say nothing about real users.

## The idea (Rafael)
A neural network with **base weights that stay frozen** (what all vapers have in common, learned offline) and an
**unfrozen part that changes as the user vapes**, so the dose is **regulated on the spot**, at every puff.
Time to first puff is not available on the device and is not used.

## How it works
Two layers.

| Layer | Timescale | Decides | Code |
|---|---|---|---|
| Slow | weeks | today's target level `u` and daily budget `u x baseline puffs for this day type` | `puffsim.SlowSchedule` (12 %/week, the same for every fast layer in the tests) |
| Fast | each puff | how today's budget is spent, puff by puff | `puff_nn.OnTheSpot` |

At every puff, **before** it happens (so a harder pull never buys nicotine):

1. **Device state**: the device's own nicotine estimate (the doses it gave, measured puff length, population
   2 h half-life), a tolerance proxy (3-day average of that estimate), minutes since the last puff, puffs in the
   last 10 min, the last hour and today against this person's own baseline habit for that hour, puff length trend,
   clock hour, weekend, budget used, longest gap in 24 h against baseline. 16 features.
2. **Network**: frozen base, 16 -> 32 -> 16 (tanh), trained on synthetic population A. Unfrozen head, 16 -> 1,
   predicts how many puffs are left today.
3. **Optimiser**: what is left of the budget, divided by the predicted remaining puffs, times a relief factor:
   more for puffs where the nicotine estimate sits far below tolerance (the first puffs after a gap, withdrawal),
   less for chain puffs (habit). The factor is renormalised so it reshapes the day without shrinking it.
4. **Safety layer**, fixed, the network cannot override it: never above the starting per-puff level, never above
   2x today's target, never above what is left of today's budget, at most +0.5x target per puff inside a bout,
   fall back to the baseline habit if the network output is not finite. The daily budget never rises.

**Learning**: every night the head is updated with the real answer for each sampled puff (how many puffs followed
it), by recursive least squares starting from the population head, with forgetting so it follows a changing
person. Uncertainty is capped at the prior's and drift from the population head is bounded, so it personalises
but cannot run away. The base weights are read-only after training (tested).

## The simulator
`puffsim.py` closes the loop that `simulate.py` leaves open: each puff's dose raises a hidden nicotine level,
tolerance follows that level over days, withdrawal (level below tolerance) raises bout rate and puff length
(compensation) and daily relapse risk. Puff timing comes from each person's routine and cues (`profiles.py`,
`population.py`); in-bout puff spacing (9.8 s) and puff duration (2.1 s) come from Robinson 2016. Everything
marked "assumption" in the code is a guess.

## Results (SIMULATION ONLY)
65 people (population B, never trained on, plus the five named profiles) x 3 seeds, 26 weeks of taper after 3
baseline weeks, then 4 weeks at zero. Success = reached zero without relapse. Differences are paired by person,
95 % bootstrap interval. Full numbers: `experiments/per-puff-v3-summary.json`.

| Fast layer | Stable | Routine changes at week 6 | Controller's nicotine model wrong |
|---|---|---|---|
| Flat (weaker bottle) | 0.85 | 0.69 | 0.83 |
| Codex budget rules | 0.73 | 0.70 | 0.70 |
| On the spot, habit only (no NN) | 0.92 | 0.79 | 0.90 |
| On the spot, NN fully frozen | 0.90 | 0.77 | 0.90 |
| **On the spot, NN frozen base + adaptive head** | 0.90 | **0.81** | 0.89 |

What this shows, inside the simulator:

- **Spending the day's nicotine where withdrawal is, instead of evenly, is the main gain**: +5 to +12 points of
  success over the flat taper, with about 13 % less nicotine per day, and it survives a deliberately wrong
  nicotine model in the controller (3 h half-life and linear puff length in the person, 2 h and 0.7 in the
  controller).
- **The adaptive head does its job on prediction**: when the person's routine changes, the frozen network's error
  doubles (0.38 -> 0.73 log units) and stays high; the adaptive head brings it back to 0.51. Success moves from
  0.77 to 0.81 (difference +0.04, interval -0.01 to +0.08: not yet distinguishable from zero).
- **When nothing changes, the network adds nothing over the person's own baseline habit**; the habit-only version
  is as good or slightly better. The NN earns its place only where the person drifts away from their baseline.
- **Codex's rules under-deliver**: they spend 27 % of the day's budget, because the exposure penalty grows with
  puffs per hour, so heavy vapers get a fraction of the plan and relapse more (-12 points against flat). Logged
  for Codex in `AGENTS.md`.

## Limits
- The simulator and the controllers share a structure (a decaying nicotine level against a slow tolerance). The
  mismatch scenario tests wrong parameters, not a wrong structure. Real withdrawal may not work this way.
- The network was trained on the same simulator family it is tested in (different people, same physics).
- 65 synthetic people, 3 seeds: the adaptive-head gain is within noise.
- Dose units are fractions of the starting per-puff level. Mapping them to hardware needs a bench calibration;
  mechanism details stay out of this repo.
- Nothing here has met a real user. The pilot data decides.

## Reproduce
```
python3 train_puff_nn.py                                   # about 10 s, writes puff_nn_base.json
WANE_MAIN=/path/to/main/checkout python3 experiments/per_puff_comparison.py   # about 7 min on 10 cores
python3 -m unittest test_puff_nn
```
`WANE_MAIN` is only needed while Codex's `puff_controller.py` is not yet committed.
