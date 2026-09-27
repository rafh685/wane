# Onboarding interview answers in the algorithm

Status: simulation, 27 Sept 2026, branch `claude/interview-priors` (from `claude/per-puff-nn`). Synthetic people only.

## Result in one paragraph
Every interview answer was wired to change the algorithm, and the settings were searched on 90 development
people. **No setting beat plain v4 by more than noise, so the kept version is v4 unchanged.** The reason is in
the simulator, not the interview: even perfect knowledge of each person's hidden difficulty did worse than a
uniform pace, and a person whose nicotine is never cut at all builds up nearly as much relapse risk as one on a
taper. The simulator's relapse is driven mostly by random day-to-day swings in withdrawal, which no starting plan
can fix. The research report said the same thing (relapse curve far too steep). The interview code stays in as
off-by-default options, ready to re-test once the relapse model is rebuilt from the research.

## What was asked (Rafael)
Create a branch where the answers to the new onboarding interview change the algorithm's behaviour, each answer
changing something, and tune the parameters for the best results.

## What each answer changes (`interview.py`)
| Question | Answer levels | What it changes |
|---|---|---|
| Time to first vape on waking | within 5 min, 6 to 30, 31 to 60, after 60 | taper pace; stronger relief for the first bout after the night (`morning_x`) |
| Wakes at night to vape | never, rarely, some nights, most nights | taper pace; night puffs spared from the pre-gap cut (`night_relief`) |
| Past quit attempts | never, lasted, back within a week, several times | taper pace |
| Optional: often anxious or low | no, yes, skip | taper pace |
| Optional: drinks a lot on nights out | no, yes, skip | softer pre-gap cut on Friday and Saturday (`pregap_social`) |
| Age, device and strength, cigarettes, pregnancy, goal | | routing and starting level, not simulated |

Pace: weekly cut = base x exp(-k x (score - 0.5)), score = weighted mean of the answer levels, clipped to 6 to
20 % a week. The 20 % cap is the fastest average tolerated in the gradual arm of Hatsukami 2018 (research report).

**How synthetic people answer:** noisy readings of their hidden difficulty (low relapse threshold, slow tolerance
adaptation). Correlations: time to first vape 0.55, night waking 0.4, quit history 0.4, mood 0.2 (assumptions
guided by the research ranking). Answer shares follow PATH for time to first vape and 7 to 10 % for night waking;
15 % careless answers, 15 % optimistic answers, 20 % skip the optional questions. The drinking answer is read from
the person's real alcohol cues. Across 400 people the interview score correlates 0.40 with hidden difficulty.

## Search (`experiments/interview_eval.py tune`, development people, habit-only demand, 1 seed)
90 people: 50 difficult (irregular habits) + 40 general. Horizon: up to 52 weeks of taper plus 4. Score J = expected
relapse risk (difficult) + 0.5 x expected risk (general); rule fixed before running: best J among settings whose
average taper lasts at most 26 weeks (v4 at 12 % lasts 23).

| Setting | J | Risk, difficult | Risk, general | Average weeks to zero |
|---|---|---|---|---|
| Uniform 10 % a week | 0.2779 | 0.213 | 0.130 | 29 |
| **Best interview setting (r25)** | **0.2788** | 0.214 | 0.130 | 19 |
| Uniform 12 % a week (v4) | 0.2790 | 0.214 | 0.130 | 23 |
| Uniform 15 % a week | 0.2795 | 0.215 | 0.130 | 19 |
| Uniform 8 % a week | 0.2851 | 0.220 | 0.130 | 35 |
| Hidden difficulty known exactly, k = 1 | 0.2846 | 0.220 | 0.130 | 27 |
| Hidden difficulty known exactly, k = 2 | 0.2886 | 0.223 | 0.131 | 32 |
| Median of 40 random interview settings | 0.288 | 0.222 | 0.131 | 19 (range 15 to 27) |

The best interview setting (J 0.2788) differs from plain v4 (0.2790) by less than the gap between neighbouring
uniform paces. Full table: `experiments/interview-tune-output.txt`.

## The non-pace levers alone (`experiments/interview_levers.py`, 90 people, 3 seeds, pace 12 %)
Expected risk against v4, 95 % intervals, and on the people each lever applies to:

| Lever | Difficult | People it applies to |
|---|---|---|
| Stronger morning relief (early vapers) | +0.0008 [-0.0001, +0.0018] | +0.0008 [+0.0000, +0.0017], n = 55 |
| Night puffs spared (night wakers) | +0.0007 [-0.0002, +0.0024] | +0.0038 [-0.0007, +0.0112], n = 11 |
| Softer cut on drinking nights | +0.0019 [+0.0006, +0.0036] | +0.0028 [+0.0010, +0.0051], n = 39 |
| All three | +0.0027 [+0.0008, +0.0052] | +0.0016 [+0.0005, +0.0032], n = 90 |

None helps; softening the cut on drinking nights is measurably worse.

## Why: where the simulator's relapse risk comes from (`experiments/risk_by_week.py`)
On the 50 difficult development people:
- **70 to 82 % of the excess relapse risk builds up while the dose is still above half** (the first 6 to 8 weeks);
  the tail below 0.15 and the time at zero add 0 to 2 %.
- **With no cut at all**, the flat taper still accumulates 0.023 to 0.034 excess hazard a week, about the same
  as with a 12 % taper (0.014 to 0.043). Taper week 1, before any cut, is the worst week (0.043 flat, 0.049 v4).
- The day-to-day SD of awake withdrawal is 0.116, while the relapse threshold is 0.08 to 0.17. Ordinary bad days
  cross the threshold on their own.

So in this simulator relapse is mostly random bad days plus a start-up artefact. A starting plan cannot change
either, and giving difficult people a slower pace only adds weeks exposed to the same noise.

## What this means
1. **Kept: v4 unchanged** (no interview-driven changes). The interview options stay in `interview.py` and
   `puff_nn.py` (`morning_x`, `night_relief`, `social_dows` / `pregap_social`), all off by default.
2. **The interview's value cannot be judged in this simulator.** The research gives the fix: a shallow logistic
   link from craving to lapse (odds about 1.9 per within-person SD), craving standardised against each person's own
   swings, relapse as a ramp of lapses over days, and a relapse curve that falls roughly as 1/t. Rebuild that, then
   rerun `experiments/interview_eval.py tune` and `test` unchanged.
3. The test on untouched people was not run: with even the upper bound (perfect knowledge) losing on development
   people, it would only measure noise.
4. In real life the interview answers may still matter a lot (time to first cigarette is the strongest relapse
   predictor in smokers). The pilot should record them next to the device data so their link to outcomes is
   measured, not assumed.
