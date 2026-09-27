# Realistic vapers and a taper calibrated to trials

Status: simulation, 27 Sept 2026, branch `claude/realistic-vapers` (from `claude/relapse-model`). Synthetic people;
every number compares versions with each other, and none is evidence about real users.

## What was asked (Rafael)
Apply the rest of the research to the simulator (puff length, gaps, bouts, uneven days, weekends, night waking as a
trait, spread in nicotine clearance, delivery vs puff length, lasting compensation) plus the parts of Codex's 40
findings not already covered; then calibrate the taper phase to the gradual-vs-abrupt trials and to PATH.

## What changed in the simulator (`puffsim.py`, `population.py`; `REALISM = False` reproduces earlier results)
| Part | Before | Now | Source |
|---|---|---|---|
| Puffs per day | log-uniform 50 to 450 | log-normal, median 200 | device logs, research report |
| Puff length | 2.1 s for everyone | pods 2.2 s, tanks 3.2 s, puff-to-puff CV 0.5 | Dowd 2023, Dautzenberg 2015 |
| Gap inside a bout | 9.8 s | 13 s median | Dautzenberg 2015 |
| Bouts | 8 puffs, never single | 4 puffs, 42 % single; bouts grouped in sessions of about 10 | Dautzenberg, Dowd, Kosmider 2018 |
| Uneven days | only for difficult people | everyone, CV about 0.33 (0.16 to 0.5) | Dautzenberg, Gao 2023 |
| Weekends | 1.0 to 2.2 x more | about 0.9 x | Dautzenberg; Lee 2018 (Codex) |
| Night waking | everyone, a little | a trait of about 8 %, about 4 nights a week | Du 2019 |
| Clearance spread | half-life SD 20 % | 30 % (about threefold span) | St Helen 2016, 2020; Benowitz 2006 |
| Nicotine vs puff length | exponent 0.7 | 1.2 (sim and device assumption) | Talih 2015 (our fit) |
| Compensation | driven by withdrawal, fades | power law of the dose, builds over 2 days, persists | Dawkins 2018, Cox 2021, Etter 2016 |

Checks against published values (`experiments/realism_checks.py`, 60 people): puffs per bout 3.9 (4.0), single
bouts 0.41 (0.37 to 0.47), gap 12.9 s (13), puff length tank 3.28 s (3.1 to 3.4) and pod 2.27 s (2.2), CV 0.49
(0.5), day-to-day CV 0.39 (0.2 to 0.6), weekend 0.95 (0.75 to 0.95), night wakers 7.4 % over 2,000 people (7 to 10),
after a 3x cut: puffs x1.16 (x1.18 to 1.21), puff length x1.25 (x1.24 to 1.26), total puff time x1.45 (x1.47 to 1.50).

## Joint calibration of the slip model (`calibrate_relapse.py`, `relapse_calibration.json`)
Five settings fitted together on 60 calibration people: baseline slip tendency, craving scale, slow-craving weight,
and two new taper-phase settings (slip tendency while nicotine is still in the plan, and how much slips snowball
then).

| Target | Published | Model |
|---|---|---|
| Abrupt quit, still off at 2 weeks / 1 month / 6 months | 0.36 / 0.28 / 0.16 | 0.43 / 0.27 / 0.15 |
| Craving at 26 weeks / week 1 | 0.36 | 0.48 |
| Plain 12 % taper, still off 12 weeks after reaching zero | about 0.22 (0.15 to 0.30) | 0.27 (was 0.48) |
| Plain taper, still on plan at week 20 | 0.57 | 0.58 |
| Check: slips before relapse | about 5 | 5 |
| Check: abrupt, still off at 3 months | 0.10 to 0.20 | 0.19 |
| Check: abrupt, still off at 1 week | 0.24 to 0.51 | 0.80, fails (5-slip rule) |
| Check: PATH former vapers back within a year | 0.31 to 0.38 | 0.70 within 12 weeks of reaching zero, much harsher |

The fit says: while nicotine is still in the plan people slip much less (odds x0.26), but slips snowball as much;
once the plan reaches zero that protection ends, and relapse comes fast, like the post-quit-day relapse in trials
(the per-person average of relapse within 12 weeks of reaching zero is 0.70 in the calibration group). PATH's natural vape quitters relapse far less; they are self-selected, so this tension stays open.

Two bugs found and fixed on the way:
- A person's slip tendency was drawn from the run's seed, not the person, so a batch sharing one seed gave everyone
  the same random offset. It is now a fixed personal trait.
- After retraining on realistic vapers, the network made a long pull raise the next dose. Puff length is no
  longer shown to the network (held-out fit 0.703 vs 0.705), and the safety test passes again.

## Results on untouched people (`experiments/relapse_v2_eval.py`, 3 seeds)
Relapse out of 100 over the taper plus 12 weeks at zero (difficult = hardest quarter by relapse during the taper):

| | General (65): relapse | of which during the taper | after reaching zero | still off | Difficult (52): relapse |
|---|---|---|---|---|---|
| Stop at once, no help | 86.8 | | 86.8 | 13.2 | 99.9 |
| Weaker bottle, 12 % a week | 73.3 | 42.7 | 30.6 | 26.7 | 99.1 |
| v3 | 72.1 | 42.9 | 29.2 | 27.9 | 98.9 |
| v4 | 71.6 | 42.3 | 29.3 | 28.4 | 98.8 |

As a funnel with v4: of 100 people, 42 relapse during the taper, 58 reach zero, 29 of those relapse within 12
weeks (about half), 28 are still off.

- v4 vs weaker bottle: -1.66 per 100 [-2.12, -1.22] general, better for 35 of 65, worse for none; difficult -0.32.
- Taper vs stopping at once: -13.5 per 100 general.
- Lucía 98.4 with v4 (98.9 weaker bottle); Karim 77.3 (80.4).

## What it means
1. The largest risk sits at the end: about half of the people who reach zero relapse within three months. The
   next engine work should target the last steps and the weeks after zero (slower tail, a long low-dose phase,
   support around zero), not per-puff shaping, which adds about 1.7 in 100.
2. The hardest quarter relapses almost regardless of dose timing in this model. Whether that is real or an effect
   of the assumed spread in slip tendency (0.8 logit SD) is unknown; the pilot has to measure it.
3. The interview results were produced on earlier models and need a rerun.
