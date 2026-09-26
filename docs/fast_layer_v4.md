# Fast layer v4: what a human planner did, learned by the device

Status: simulation prototype, 26 Sept 2026, branch `claude/per-puff-nn`. Every number comes from synthetic people
in `puffsim.py` and compares controllers with each other. None of it is evidence about real users.
v3 is described in `docs/fast_layer_v3.md`.

## Where v4 comes from
In `experiments/human_vs_algorithm.py`, Claude wrote a dosing plan by hand from one person's device data and
beat v3 on withdrawal. The plan did three things v3 could not. v4 learns them per person, from device data only:

| Rule | What it does | Learned from |
|---|---|---|
| Relief bout | the first 5 puffs of the first bout after >= 3 h without a puff get weight 2.0 | puff timestamps |
| Front-loading | the first ~30 % of each bout get up to 1.5, the rest 0.8 | the person's usual bout length (baseline weeks) |
| Pre-gap cut | puffs in the 2 h before this person's usual long daily gap (and any later) get weight 0.2 | when the longest gap of each day starts, relearned daily from the last 5 days of the same day type |

v3's own shaping (more nicotine where the device's nicotine estimate sits below its tolerance proxy) is switched
off in v4: once the three rules are in, it made things worse. Everything else is unchanged: the network (frozen
base + adaptive head) predicts the puffs left today, the budget is spread over them, the safety layer caps every
dose (<= starting level, <= 2x today's target, <= what is left today, smooth inside a bout).

## Method (so the result is not tuned to the test)
1. **Tuning** on a development population (40 people, seed 31), which is neither the network's training people
   (seed 21) nor the test people (seed 12). 54 settings, then a finer 18 around the edge where the best sat
   (`experiments/tune_fast_layer.py`). Objective: expected relapse risk, `1 - exp(-sum of daily hazards)` along
   each person's full path, much less noisy than counting relapses.
2. **Choice**, fixed before testing: relief boost 2.0 (2.5 tied; the milder kept), front-loading 0.5, pre-gap cut
   0.2 (the grid's lowest; kept as a product floor, since lower would make the last puffs before bed nearly empty,
   which a real user would notice).
3. **Test** once on population B (60 + the 5 named, 3 seeds, 3 scenarios), `experiments/fast_layer_v4_eval.py`,
   analysis in `experiments/fast_layer_v4_analysis.py`, numbers in `experiments/fast-layer-v4-summary.json`.
4. **Controls**: a flat taper at v4's nicotine share (0.87, fixed from the development run), to separate timing
   from simply giving less nicotine; v4 with a fully frozen network; v4 with no network (habit only).

## Results (SIMULATION ONLY)
Expected relapse risk, mean over 65 people (lower is better):

| | Stable | Routine changes at week 6 | Controller's nicotine model wrong |
|---|---|---|---|
| Flat (weaker bottle) | 0.151 | 0.268 | 0.174 |
| Flat at v4's nicotine (0.87) | 0.145 | 0.252 | 0.171 |
| Codex budget rules | 0.256 | 0.264 | 0.298 |
| v3, network | 0.113 | 0.220 | 0.127 |
| v4, habit only (no network) | 0.094 | 0.204 | 0.099 |
| v4, network fully frozen | 0.093 | 0.201 | 0.100 |
| **v4, network frozen base + adaptive head** | **0.094** | **0.200** | **0.101** |

Paired over people, 95 % intervals:
- **v4 vs flat**: risk -0.057 [-0.071, -0.043] stable, -0.069 [-0.085, -0.053] routine change, -0.073 [-0.092, -0.055]
  model wrong. About 26 to 42 % lower. Better for 46 to 48 of 65 people, worse for 0 to 3.
- **v4 vs flat at the same nicotine**: -0.050, -0.053, -0.069. Giving less nicotine explains only 5 to 24 % of v4's
  gain; the rest is timing.
- **v4 vs v3**: -0.018 [-0.024, -0.013], -0.020 [-0.026, -0.015], -0.026 [-0.034, -0.019]. Better for 40 to 43
  people, worse for at most 1.
- **Network vs no network, within v4**: no difference when routines are stable (0.000) or the model is wrong
  (+0.002, borderline); a small gain when routines change (-0.004 [-0.009, -0.001]). The habit-only version gives
  3.5 % empty puffs after a routine change (its habit is stale, the budget runs out); the network gives none.
- **Adaptive head vs frozen head**, within v4: no measurable difference in any scenario.
- **Success** (reached zero, no relapse) moves the same way but its intervals are wide: v4 vs flat +0.08
  [+0.03, +0.13] stable, +0.11 [+0.04, +0.19] routine change, +0.06 [-0.01, +0.12] model wrong.

**Who gains** (v4 minus flat, averaged over scenarios): the gain concentrates where risk is highest.
People with a low relapse threshold -0.112, slow tolerance adaptation -0.103, work-ban routines -0.111, light
vapers -0.086. Evening-only vapers gain least (-0.008), but their risk on flat is already low (0.075).

**Harm check** (v4 raises risk by > 0.02 against flat or v3, any person, any scenario): one case. P12-023, a
429 puff/day all-day vaper switched to an evening-only routine: 0.852 vs flat 0.801. Every budgeted version
does worse than flat for this person, v3 included (0.858); v4's rules are not the cause (removing the pre-gap
cut makes it 0.861). Cause: the daily budget is still sized on the old puff count, so when the person suddenly
puffs 24 % less, the fast layer packs more nicotine into each puff (97 % of budget used vs flat's 76 %).
**Open for the slow layer:** re-size the budget when a lasting change in puff count is detected.

**Safety**: no dose above the starting level in 4,680 runs; the 2x-target ceiling is reached but never passed;
the daily cap holds (unit-tested). Empty puffs (dose 0 because the day's budget ran out): v4 with the adaptive
network, none in 585 runs; the habit-only v4 averaged 1.2 %, and up to 39 % for one person after a routine
change, because its habit is never relearned. This is the clearest practical reason to keep the network.

**Named profiles** (stable, expected risk flat -> v3 -> v4): Marta 0.150 -> 0.106 -> 0.089, Diego 0.183 -> 0.134
-> 0.105, Lucía 0.266 -> 0.159 -> 0.113, Karim 0.381 -> 0.248 -> 0.155, Ana 0.097 -> 0.077 -> 0.074.

## Rematch: hand plans against v4 (`experiments/human_vs_algorithm_v4.py`, output in `human-vs-algorithm-v4-output.txt`)
Plans written from device data only and frozen with a prediction before running. 20 seeds each.

| Person | | Flat | Flat, same nicotine | Hand plan | v3 | v4 |
|---|---|---|---|---|---|---|
| P12-030, evening heavy vaper | withdrawal weeks 1 to 8 | -0.025 | -0.035 | -0.042 | -0.033 | **-0.045** |
| | expected risk | 0.061 | 0.061 | 0.061 | 0.061 | 0.061 |
| P12-027, the hardest person | withdrawal weeks 1 to 8 | +0.015 | +0.012 | -0.014 | -0.007 | **-0.022** |
| | expected risk | 0.392 | 0.363 | 0.261 | 0.288 | **0.238** |

v4 now beats both hand plans (P12-027: -0.023 [-0.031, -0.016] risk vs the hand plan). The hand plan's
prediction held (clearly better than flat, within 0.03 of v4). P12-030 never approaches relapse under any
controller, so only withdrawal separates them there.

## What to take from this
1. Timing inside the day matters more than anything else tested: relief after long gaps, and not spending
   nicotine right before the longest gap. Both are learnable from puff timestamps alone.
2. The network is not what makes v4 better. Its value is robustness when a person's routine changes (no empty
   puffs, slightly lower risk). Keep it, with the habit fallback it already has.
3. The adaptive head shows no measurable benefit here; the simulated people change less than real ones do.

## Limits
- The simulator's withdrawal model is ours. In it, slightly less nicotine lowers risk a little (flat at 0.87
  beats flat at 1.0); real short-term withdrawal may behave differently. The nicotine-matched control keeps this
  from inflating v4's result, but the pilot has to check the direction.
- The pre-gap cut and the relief boost sat at the edge of the tuning grid; the values kept are product choices.
- 65 synthetic people, 3 seeds, one slow layer (12 %/week). The budget re-sizing gap above is unsolved.
- Dose units are fractions of the starting per-puff level; hardware mapping needs a bench calibration, and
  mechanism details stay out of this repo.

## Reproduce
```
python3 experiments/tune_fast_layer.py && python3 experiments/tune_fast_layer.py --refine   # about 11 min
FLAT_MATCH=0.87 WANE_MAIN=/path/to/main/checkout python3 experiments/fast_layer_v4_eval.py   # about 25 min
python3 experiments/fast_layer_v4_analysis.py
python3 experiments/human_vs_algorithm_v4.py
python3 -m unittest test_puff_nn
```
