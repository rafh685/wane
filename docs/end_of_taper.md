# Protecting the end of the taper: first fix

SIMULATION ONLY. Synthetic vapers, relapse model v2, realistic vapers, v4 per-puff layer (27 Sept 2026, night).

## What was asked (Rafael)

The roadmap's next step: about half of the people who reach zero on Wane relapse in the following 12 weeks.
"try the fixes and keep the version with the best results".

## Why the end is dangerous in the simulator

Tolerance moves on two clocks. The fast one (S, tau 3 to 14 days) follows the nicotine level within days; the slow
one (S2, tau about 190 days, from the 19-week half-life of urges in Ussher 2013) takes months. Slow craving is
S2 - S. At zero, S falls within days while S2 is still high, so slow craving peaks just after zero, on the same day
the "nicotine still in the plan" protection of the slip model ends.

## Candidates (`experiments/end_of_taper.py`, only Wane's weekly plan changes)

| Plan | Weekly plan | Zero in week | Mean dose over the plan |
|---|---|---:|---:|
| A current | 12 % a week to 15 % of the start (week 15), then 8 equal steps | 23 | 0.339 |
| B front-loaded | 16 % a week to 15 % (week 11), then 12 equal steps | 23 | 0.274 |
| C front-loaded+ | 20 % a week to 15 % (week 9), then 14 equal steps | 23 | 0.237 |
| E +4 weeks | as A, 12 equal steps at the bottom | 27 | 0.299 |
| F +8 weeks | as A, 16 equal steps at the bottom | 31 | 0.270 |
| D relative | 12 % a week down to 2 %, then zero | 31 | 0.264 |

Outcome, fixed before running: nicotine-free 12 weeks after each person's zero day. Rule, fixed before running:
adopt the best same-end-date plan if it beats A by at least 0.5 per 100 on general development people without
costing difficult people more than 0.5; longer plans are reported, not adopted, because the simulator has no dropout.

## Development (difficult dev 50 + population seed 31, 40; 2 seeds), nicotine-free per 100

| Plan | Typical free | vs A | Difficult vs A |
|---|---:|---:|---:|
| A current | 32.6 | | |
| B front-loaded | 33.4 | +0.75 [+0.48, +1.05] | +0.01 |
| C front-loaded+ | 33.8 | +1.14 [+0.75, +1.57] | +0.03 |
| E +4 weeks | 34.3 | +1.69 [+1.14, +2.27] | +0.01 |
| F +8 weeks | 35.6 | +2.94 [+1.96, +3.96] | +0.03 |
| D relative | 35.7 | +3.07 [+2.05, +4.16] | +0.03 |

Chosen by the rule: **C front-loaded+**, now `puffsim.END_PLAN` (`SlowSchedule(weekly_cut=0.20, bottom_steps=14)`).

## Test, once (difficult test 52 + population B 65, 3 seeds)

| Plan | Typical free | Relapse during taper | Relapse after zero | vs A | Difficult vs A |
|---|---:|---:|---:|---:|---:|
| A current | 29.1 | 42.3 | 28.6 | | |
| C front-loaded+ (adopted) | 30.4 | 44.0 | 25.6 | +1.28 [+0.94, +1.65], better 34, worse 0 of 65 | +0.17 [+0.04, +0.34] |
| E +4 weeks | 31.0 | 44.1 | 24.8 | +1.92 [+1.38, +2.49] | +0.18 |
| F +8 weeks | 32.6 | 45.4 | 22.0 | +3.52 [+2.57, +4.52] | +0.35 |

In the 9-month comparison with the other methods (`experiments/wane_end_plan_eval.py`, same windows as
`traditional_taper_eval.py`): Wane with the new plan leaves 29.7 in 100 typical vapers nicotine-free (44.0 relapse
during the taper, 26.3 after zero), +1.27 [+0.93, +1.65] over the old plan and +21.5 [+15.2, +28.3] over the hand
taper; difficult people 1.4.

## What it means

- Cutting faster early, while people still have plenty of nicotine, lets the slow clock start falling sooner, so
  there is less slow craving at zero. It costs a few more slips early (44 against 42) and saves more after zero
  (26 against 29). Same end date, so no extra plan length.
- The gain depends on the slow craving clock, which comes from one study of smokers (Ussher 2013); it should be
  checked against real vapers.
- Longer plans help more in the simulator (+1.9 for 4 more weeks, +3.5 for 8), but the simulator cannot count people
  who give up on a long plan, so they are not adopted. Real users must tell us how long is too long.
- About 26 in 100 still relapse after zero. Next: support in the weeks around zero.
