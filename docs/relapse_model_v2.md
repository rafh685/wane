# Relapse model v2, and the interview judged on it

## Update, 27 Sept evening: new data from Codex (PATH, Hughes 2004, Ussher 2013)
Codex downloaded PATH Waves 1, 2 and 4 and two papers (`docs/path_data_handoff_for_claude.md` in the main
checkout). What changed, and what it showed:

**Slow craving clock.** Ussher et al. 2013 (452 quitters with behavioural support, no medication, CO-verified,
52 weeks): among abstainers, strong urges fall from 66 % (week 1) to 45 % (week 4), 13 % (week 26) and 0 % (week
52); urge strength falls from about 2.8 to 1.0 to 0.45 on a 0 to 5 scale, an exponential with a half-life of
about 19 weeks. The slow clock was 30 days (from receptor imaging); it is now about 190 days per person (100 to
400). Survival in that study: 44 % week 1, 29 % week 4, 13 % week 26, 8 % week 52.

**Recalibration** (`calibrate_relapse.py`, abrupt unaided quit, 60 people):

| Check | Published | Model |
|---|---|---|
| Survival 14 days (target, Herd 2009 fit) | 0.36 | 0.34 |
| Survival 30 days (target) | 0.28 | 0.24 |
| Survival 180 days (target) | 0.16 | 0.19 |
| Urge at 26 weeks / week 1 (target, Ussher 2013) | 0.36 | 0.36 |
| Lapses before relapse (check, Kirchner 2012) | about 5 | 5.0 |
| Survival 90 days (check, Hughes 2004 range) | 0.10 to 0.20 | 0.20 |
| **Survival 7 days (check, Hughes 2004 range)** | **0.24 to 0.51** | **0.64, fails** |

The 7-day miss is structural: relapse is defined as 5 lapse days within 14, so nobody can relapse before day 5,
while most real quitters relapse within 8 days. It matters for abrupt quits, little for a gradual taper.
Calibrated: baseline lapse logit -4.1, one standardised unit of craving = 6 times the person's 3-day baseline
SD, slow-craving weight 0.5.

**PATH, real US adult vapers 2013 to 2015** (`experiments/path_vapers.py`, Wave 1 weights with 100 Fay replicates;
transitions use Wave 1 weights among people re-interviewed, so attrition is not corrected):
- time to first vape, daily vapers: within 5 min 19.8 % [16.3, 23.4], 6 to 30 min 37.2 %, 31 to 60 min 22.8 %,
  after 60 min 20.2 % (n = 627). The interview simulation now uses these shares (was 15 / 50 / 20 / 15).
- stopped a year later: all current vapers 31.4 % [28.1, 34.7] (n = 1,259); daily 18.2 % [13.8, 22.5]; some days
  41.6 %; exclusive vapers 25.7 %; dual users 34.9 %.
- **time to first vape does not clearly predict stopping** among daily vapers: within 5 min 23.3 %, 6 to 30 min
  10.9 %, 31 to 60 min 15.7 %, after 60 min 25.7 %; odds of stopping, after vs within 30 min, 1.43 [0.87, 2.34].
  The interview assumed a strong link (correlation 0.55, borrowed from smokers). For vapers that is not supported.
- former vapers back a year later: 26.8 % [22.7, 30.9] (n = 505); by time since quitting: under 1 month 38 % (n =
  27), 1 to 6 months 31.5 %, 6 to 12 months 25.9 %, over 1 year 22.5 %. Smokers (Herd 2009): 42 % for 1 to 6
  months, 22 % for 6 to 12 months, 5 to 17 % beyond a year. Vapers relapse less early, more late (PATH counts any
  past-30-day use as back).
- Wave 2 to Wave 4 (about 2 years, unweighted): 47.2 % of current vapers stopped, 21.2 % of former vapers back.

**Validation rerun on the recalibrated model** (`experiments/relapse_v2_eval.py`, 117 untouched people x 3 seeds):

| | Difficult | General |
|---|---|---|
| Abrupt quit, no support | 0.795 | 0.756 |
| Weaker bottle, 12 % a week | 0.651 | 0.520 |
| v3 | 0.645 | 0.513 |
| v4 | 0.642 | 0.508 |

- v4 vs weaker bottle: -0.0091 [-0.0115, -0.0071] difficult (better for 30 of 52, worse for none), -0.0127
  [-0.0158, -0.0100] general (better for 49 of 65, worse for none).
- **New mismatch:** a plain taper now succeeds for 48 % of the general group and 35 % of difficult people. Trials
  find gradual reduction no better than abrupt quitting with support (RR 1.01, 22 trials), with about 15 to 22 %
  abstinent at 6 months. The model's taper phase is too kind. Next calibration step: fit the taper-phase lapse
  parameters (contagion with nicotine available) to those trials and to the gradual arm of Hatsukami 2018.
- The interview results below were produced before this recalibration and before the PATH finding; they need a
  rerun with a weaker time-to-first-vape link.

## Earlier results (first calibration)

Status: simulation, 27 Sept 2026, branch `claude/relapse-model` (from `claude/interview-priors`). Synthetic people
only; every number compares versions with each other.

## Results in brief
1. **The rebuilt relapse model matches the published curve.** After an abrupt quit, 37 % of simulated people are
   still abstinent at 2 weeks (published 36 %), 24 % at 1 month (28 %) and 17 % at 6 months (16 %). The median
   number of lapses before relapse is 5.0, the published figure (Kirchner 2012), which was not fitted.
2. **On it, v4's per-puff timing helps only a little:** -0.004 relapse risk for difficult people and -0.010 for
   the general group against the weaker bottle (it was -0.16 on the old model). Its old advantage came mostly
   from smoothing the old model's random bad days.
3. **The taper's speed now matters, and the interview version beats v4 at 12 % for difficult people:**
   relapse risk 0.858 vs 0.874, a difference of -0.015 [-0.029, -0.003] on 52 untouched difficult people.
4. **But most of that gain comes from tapering more slowly on average** (26 weeks instead of 23). Against a
   uniform taper of the same average length, the interview's edge is -0.010 [-0.024, +0.003] for difficult
   people (not proven) and +0.008 [-0.006, +0.021] for the general group (slightly worse, not proven).
5. **Knowing each person's real difficulty would help** (-0.031 [-0.047, -0.017] vs the interview, difficult
   people). The interview's answers, at the assumed accuracy, are not informative enough to capture it.

## What was asked (Rafael)
Rebuild the simulator's relapse model from the research, so the interview (and everything else) can be judged.

## What changed in the simulator (`puffsim.py`, `relapse.py`)
| Part | Old (v1) | New (v2) | Source |
|---|---|---|---|
| Start | recording began with tolerance still settling (start-up spike) | 28 unrecorded burn-in days | diagnosis, 27 Sept |
| Craving | withdrawal only (one clock, days) | withdrawal + slow receptor-level craving (second clock, about 30 days, range 7 to 50) | PET receptor studies: Cosgrove 2009, Mamede 2007 |
| Scale | absolute excess withdrawal | 3-day craving against the person's own baseline, per weekday | Allen 2008: no absolute threshold |
| Link to lapse | near step: odds x7.4 per 0.03 above a threshold | smooth: odds x1.87 per standardised unit | Vafaie & Kober 2022, 237 studies |
| Cues | none in the hazard | alcohol days x3, stress days x1.5 | research report (x4 on drinking days; stress assumption) |
| Lapse to relapse | one draw | lapses cluster (contagion), weaker while nicotine is available; relapse = 5 lapse days in 14 | Kirchner 2012; Shiffman 2006 |
| Calibration | tuned by hand | baseline propensity, craving scale and slow-craving weight fitted to the abrupt-quit relapse curve | Herd 2009 fit, Garvey 1992 |

Calibrated values (`relapse_calibration.json`): baseline daily lapse odds -3.1 logits (about 4 % a day at average
craving), one standardised unit of craving = 12 times the person's own 3-day baseline SD, slow craving weight 0.5.
Assumptions still in the model: the contagion sizes, the stress multiplier, the spread of baseline propensity.

## Check against published outcomes (`experiments/relapse_v2_eval.py`, 117 untouched people x 3 seeds)
Relapse risk over 24 weeks of taper plus 12 weeks at zero:

| | Difficult | General |
|---|---|---|
| Abrupt quit, no support | 0.909 | 0.825 |
| Weaker bottle, 12 % a week | 0.825 | 0.641 |
| v3 | 0.822 | 0.636 |
| v4 | 0.820 | 0.630 |

- Tapering vs quitting cold without support: general success 36 % vs 17 %. Published: nicotine support roughly
  doubles to triples success over going unaided (unaided 3 to 5 % at 6 to 12 months). Consistent in direction
  and size.
- Success on a taper: 36 % (general) and 17 % (difficult), against published 13 to 31 % with support. The general
  group is at the optimistic end.
- Relapsed by week 26 on the weaker bottle (general): 54 %, against about 43 % not complying at the same point in
  the gradual arm of Hatsukami 2018. Slightly pessimistic.
- v4 vs weaker bottle: -0.0043 [-0.0056, -0.0031] difficult (better for 18 of 52, worse for none), -0.0104
  [-0.0132, -0.0081] general (better for 40 of 65, worse for none). v4 vs v3: -0.002 and -0.005.
- Lucía: abrupt 1.000, weaker bottle 0.945, v4 0.938. Karim: 0.968, 0.857, 0.853.

## The interview on the new model (`experiments/interview_eval.py`, RELAPSE=v2)
**Tuning** (90 development people, habit-only demand, rule fixed before running: best score among settings
averaging at most 26 weeks). First search: 47 settings; refinement: 25 settings around the best.

| Setting | Score | Risk, difficult | Risk, general | Weeks |
|---|---|---|---|---|
| Uniform 8 % | 1.192 | 0.864 | 0.655 | 35 |
| Uniform 10 % | 1.210 | 0.875 | 0.671 | 29 |
| Uniform 12 % (v4) | 1.227 | 0.883 | 0.688 | 23 |
| Hidden difficulty known, k = 3 | 1.186 | 0.842 | 0.688 | 34 |
| **Chosen interview setting (n15)** | **1.188** | 0.857 | 0.663 | 26 |

Slower is always better in this model, because giving up on a long taper is not simulated; that is why the
26-week limit was fixed in advance.

**What each answer does in the chosen setting** (`interview.chosen_policy()`):
- quit history: the strongest pace signal (weight 0.74)
- time to first vape: pace (0.47) and 25 % stronger relief on the first bout after the night for early vapers
- mood: a small pace signal (0.18)
- drinking: on Friday and Saturday nights the pre-bed cut is 0.8 instead of 0.2, so evening nicotine is kept
- night waking: weight 0, night relief off. It did not help in the simulation, so this answer changes nothing
Pace formula: weekly cut = 0.122 x exp(-3.49 x (score - 0.5)), between 6 % and 20 %.

**Test, once, on untouched people** (52 difficult, 65 general, 3 seeds, Felipe's network):

| Version | Risk, difficult | Risk, general | Weeks |
|---|---|---|---|
| v4, 12 % | 0.874 | 0.700 | 23 |
| v4, same average length (10.8 %) | 0.869 | 0.691 | 26 |
| **Interview** | **0.858** | 0.699 | 26 |
| Interview, weak and careless answers | 0.859 | 0.698 | 25 |
| Hidden difficulty known (upper bound) | 0.828 | 0.678 | 35 |

- Interview vs v4 12 %: -0.015 [-0.029, -0.003] difficult (better for 16, worse for 11); -0.001 [-0.015, +0.011]
  general.
- Interview vs same-length uniform: -0.010 [-0.024, +0.003] difficult; +0.008 [-0.006, +0.021] general (worse for
  36 of 65).
- Weak answers do as well as normal ones, which confirms the answers' information is not what drives the gain.
- Lucía: v4 0.982, interview 0.921 (slowed to 9.8 % a week). Karim: v4 0.939, interview 0.961 (his answers
  sped him up to 13 %).

## What this means
- **Keep relapse model v2** as the simulator's default for all future comparisons; v1 stays for reproducing old
  results.
- **Keep the interview version** as the best-scoring version, but say it plainly: most of its gain over v4 comes
  from a slower average pace, and its targeting edge is not proven. A slower default pace (about 11 % a week) is
  the robust part.
- **The bigger lever is learning the pace from the device over the first weeks**, since perfect knowledge of
  difficulty is worth -0.03 and a 0.4-accurate interview captures little of it. Next step: an adaptive weekly
  layer that slows or speeds the taper from observed lapses, puffing and craving proxies.
- The pilot should record the interview answers next to device data, so their real predictive power replaces the
  assumed one (0.4 correlation).
- Still assumed: contagion sizes, stress effect, spread of baseline propensity, the answers' accuracy, and that
  people do not drop out of long tapers.
