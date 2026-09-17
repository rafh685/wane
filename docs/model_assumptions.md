# Where every number comes from

The behavioural model in `profiles.py` and the toy version of it are built from a handful of formulas.
This file says, for each one, what it claims, where the shape comes from, and whether the constant
is sourced, calibrated, or a guess. Say this out loud when presenting: the shapes are from the
literature, the constants are tuned to reproduce published outcome ranges, and the pilot exists to
measure them per person.

Legend: **S** sourced from literature · **C** calibrated to reproduce a published range · **G** guess, order of magnitude only

## 1. Baseline puffing

`puffs = 100 × gauss(1, 0.10)` (toy) · `puffs_per_day` per profile, 40 to 250 (repo)

- **What it claims.** A person at a stable dose takes roughly the same number of puffs each day, with random variation.
- **Shape.** Multiplicative Gaussian noise: variation scales with the person's level.
- **100 puffs/day, range 40 to 250.** S. Puff-topography studies of e-cigarette users report daily counts from a few dozen to several hundred depending on device and user.
- **10% day-to-day spread.** G. Real coefficients of variation are about 10 to 30%. The repo uses 0.10 to 0.20 per profile.

## 2. Relative drop

`drop = (last_dose − dose) / last_dose`

- **What it claims.** What matters is the size of the cut relative to what the body is used to. 20 to 18 feels like 10 to 9.
- **Shape.** S. This is the assumption behind every percentage taper ladder and behind receptor-adaptation models of dependence.
- **Known limit.** It breaks near zero: 0.5 to 0 is a "100% cut". The last step needs an absolute-drop rule. Not implemented yet.

## 3. Withdrawal from a cut

`craving += sensitivity × drop × 4` (toy) · `pending_withdrawal += sensitivity × drop × 4`, released 35% per day (repo)

- **What it claims.** A cut adds withdrawal in proportion to its size, and it arrives over a few days, not at once.
- **Linear in the drop.** S for direction, G for linearity. Withdrawal severity rises with the size of the reduction; no study gives a precise curve, so linear is the honest default.
- **`sensitivity`, per person, 2.5 to 7.** S. Between-person variability in withdrawal severity is the best documented fact in the field. The range is chosen so the population spans "barely notices" to "cannot tolerate a standard ladder".
- **× 4.** C. Scale factor tuned so that a 12% weekly ladder breaks the high-sensitivity profile within a few weeks and spares the low-sensitivity one, giving a fixed-taper relapse rate near the fifth to a third reported in reduction trials.
- **35% released per day** (repo only). C to match S. Cumulative arrival 35% day 1, 58% day 2, 73% day 3, 82% day 4. Matches the finding that withdrawal symptoms peak within the first 3 days (Hughes 2007, review of withdrawal time course).
- **`+=` not `=`.** Cuts close together must stack. Overwriting silently discarded unreleased withdrawal and made frequent small cuts look free. Fixed 16 Sept 2026.

## 4. Decay of craving

`craving −= decay × (craving − 2)`, decay 0.12 (toy), 0.09 to 0.22 per profile (repo)

- **What it claims.** Craving fades back toward a resting level, faster when it is higher.
- **Shape.** S. Exponential return to baseline is the standard description of withdrawal time course: peak in days, back toward baseline over 1 to 4 weeks (Hughes 2007). Craving specifically fades faster than mood symptoms.
- **Decay constants.** C. 0.12/day gives a half-life of about 5.5 days; 0.09 about 8 days; 0.22 about 3 days. All inside the published range.
- **Resting level 2 on a 0 to 10 scale.** G. A vaper at a stable dose is not craving-free, just low.

## 5. Compensation

`puffs × (1 + 0.06 × (craving − 2))` (toy) · plus `elasticity × (1/dose_ratio − 1) × 0.5` (repo)

- **What it claims.** People puff more when they get less nicotine and when craving is up.
- **Shape and range.** S. Light-cigarette and reduced-nicotine studies find 30 to 70% of the lost nicotine recovered by puffing harder, longer or more often. `elasticity` 0.15 to 0.70 per profile spans that range.
- **0.06 per craving point.** C. Tuned so that craving 6 gives about 24% more puffs, inside the compensation range.

## 6. Relapse

`if craving > threshold and random() < p` · toy: threshold 6.5, p 0.05 · repo: threshold 6 to 8 per profile, p = 0.04 + 0.06 × (craving − threshold), × 1.5 at weekends

- **What it claims.** Relapse is probabilistic, not a switch. Two people at the same craving do not both fail. Risk rises with how far above threshold you are, and weekends add temptation.
- **Probabilistic form.** S in spirit: lapse studies model relapse as a hazard that rises with craving and with cue exposure (alcohol, social settings).
- **Threshold and daily probability.** G, then C. Chosen so that runs fail over weeks, not days, and so the fixed-taper relapse rate lands near published ranges.
- **Weekend × 1.5.** G. Direction is sourced (social and alcohol cues raise lapse risk), the size is not.

## 7. Signals the engine can measure (repo only)

`night_puffs = puffs × 0.02 × (1 + max(0, craving − 4))` · `ttfc_min = 30 × exp(−0.25 × (craving − 2))`

- **What it claims.** Withdrawal leaves traces in the timestamps: puffs at night and a shorter time to first puff after waking. These replace self-report.
- **Time to first puff.** S. It is the single strongest item in the Fagerström and Heaviness of Smoking indices; shorter means more dependent.
- **Night puffs as a withdrawal sign.** S for direction, night waking to use is a documented withdrawal symptom. The 2% baseline and the slope are G.
- **Circularity warning.** The model generates these signals from craving, and the engine reads them to infer craving. The engine is therefore evaluated on data that obeys the assumptions it was written with. This is the main honest limitation of the simulation and the reason the pilot needs real puff logs.

## 8. The fixed taper

`dose × 0.88` every 7 days

- **12% per week.** S as an order of magnitude. Commercial ladders (18 to 12 to 6 to 3 mg/ml over a few months) average out to roughly 10 to 15% a week.
- **Weekly.** S. It is how often a person refills. With hardware the same engine can decide daily; see `period` in `simulate.py`.

## 9. The adaptive rule (engine.py)

- **Decision points, tailoring variables, decision rule, intervention.** S. This is the JITAI framework (Nahum-Shani et al. 2018).
- **Hold / half cut / cut thresholds** (risk 0.6 and 0.3, derived craving 7 and 5). G. Hand-set. The plan is to replace them with weights fitted on a synthetic population, then on pilot data.
- **Personal rate learning** (× 1.05 when tolerated, × 0.75 when struggling, bounds 3% to 15% per week). G. Asymmetric on purpose: slow down fast, speed up slowly, because the cost of a relapse is larger than the cost of a slow week.
- **Never raise the dose.** Design rule, not a parameter. Raising would reward the behaviour the product is trying to end.


## 10. Step adherence (manual products)

`step applied with probability adherence × exp(−0.25 × max(0, craving − 3)) × 0.99^week` · automatic delivery: always

- **What it claims.** In a manual product every step requires the user to mix or buy a weaker liquid and switch. Some steps are skipped, more often when craving is up, and the habit of switching fades over months. Eight skipped steps in a row is counted as stalled: the person is parked on the ladder, not relapsed and not off.
- **Direction.** S. Adherence is the weakest link of nicotine replacement: roughly a third to a half of users complete a full course, and e-liquid reducers commonly stay at 6 or 3 mg/ml for years.
- **0.85 and 0.70 per step, the craving penalty, the 1 % weekly drift.** G. No study measures per-step adherence for e-liquid ladders. These are chosen so that 40-week completion under the ladder lands between a quarter and two thirds, the range implied by course-completion figures.
- **Why it matters.** With adherence in the model, the ladder's main failure mode is stalling, not relapse, and automatic execution (hardware) becomes the largest single effect in the simulation, larger than any change to the decision rule. Measuring real per-step adherence is a pilot objective.

## 11. Catching the minority a blind schedule loses (experiments/minority.py)

Once steps are automatic, a blind 12 %/week schedule already gets about four people in five off nicotine. The engine's
measurable job is the fifth person. Tried on the fitted engine, five profiles x 30 runs and an unseen population, 40 weeks:

- **Slow down harder after trouble** (rate x0.5 instead of x0.75, floor 1 %/week): no effect.
- **Hold at half the fitted risk threshold**: Lucía (hard compensator) 57 % -> 47 % relapse, small cost in speed.
- **Weekend guard** (never more than half a cut when `weekend_low`, heavy weekends at a low dose, is above 0.15):
  Karim (weekend drinker) 40 % -> 27 %. Cost: ordinary people with mild weekend patterns slow from 19 to 33 weeks to zero.
- **Both**: five-profile relapse 21 % -> 15 %, Marta 10 % -> 0 %, Lucía 47 %, Karim 27 %.

Status: the pattern is detectable (the `weekend_low` feature gets weight +0.32 in the risk model) and about a quarter to a
third of the minority's failures are preventable in simulation. The rest need the person's own response to their first
cuts, which only a pilot measures. The guard and the lower threshold are not in the default engine; they are a product
decision about how much speed to trade for the fragile few, and are kept as evidence.

## Calibration against real data (17 Sept 2026)

Source: Dawkins et al. 2018, *Addiction*, "Real-world compensatory behaviour with low nicotine concentration e-liquid",
open data CC BY 4.0, DOI 10.18744/LSBU.002952, in `data/lsbu/`. 20 experienced UK vapers, eVic device logging every
puff, one week each at 18 mg/ml and 6 mg/ml with fixed power, plus craving (urge to vape, 1-6) and the Mood and
Physical Symptoms Scale (MPSS, 0-24). Per-condition summaries only, not timestamps.

What the data says, 18 -> 6 mg (a 67 % cut), per person, medians:
- puffs per day x1.18 (IQR 1.10-1.42), puff duration x1.26, total puffing time x1.47 (IQR 1.34-1.68, max 2.15)
- e-liquid consumed x1.23
- strength of urges +0.75 on a 1-6 scale (2.15 -> 2.90); MPSS +1.3 on 0-24; only 55 % of people had urges rise at all
- nobody stopped or relapsed in the week
- implied elasticity in our formula: median 0.47, IQR 0.34-0.68, range -0.07 to 1.15
- device-measured puffs per day at 18 mg: median 292, range 114-585

What changed in the model because of it:
1. **Withdrawal scale x4 -> x0.6** (section 3). The old value gave a craving rise of about 3 points for this cut and
   relapses within the week; the data shows roughly +1.5 on a 0-10 scale and none.
2. **Withdrawal follows intake, not liquid strength.** Compensation gives part of the nicotine back, so the drop that
   drives craving is dose x sqrt(compensation), not dose alone.
3. **Compensation splits into more puffs (43 %) and longer puffs (57 %)**, matching the count/duration split.
   Puff duration is now simulated and is a new engine feature (`dur_trend`): the device logs it, and it is the larger
   compensation channel.
4. **Tonic craving at low intake** (new). With honest withdrawal numbers, a gentle 12 %/week taper never broke anyone:
   fixed-taper relapse fell to 1 %, against 20-33 % in reduction trials. The missing piece is chronic: at very low
   nicotine, resting craving stays elevated for weeks (reduced-nicotine cigarette trials, Donny et al. 2015, NEJM).
   Resting craving is now 2 + tonic_gain x (1 - intake ratio), tonic_gain 0.9-2.7 per profile, chosen so that the
   one-week replay still matches the data AND a full 12 %/week taper to zero fails about 22 % of people.
5. A bug: the craving effect on puffing was applied twice (hourly shape and daily total). Fixed.
6. Baseline puffs per day raised toward the measured range; population elasticity now drawn around 0.47.

Replay of the study in the calibrated model (five profiles x 15 seeds): total puffing x1.74, craving +1.44, relapse 1 %.
Still a little heavier than the data on puffing. Left as is: the profiles are chosen to be harder than the average
LSBU volunteer, who was not trying to quit.

What this data cannot tell us: night puffs, time to first puff, weekend patterns, or what happens over months.
Those parts of the model remain unsourced until the pilot.

## What is missing

- An end rule so runs can reach 0 mg/ml, and an absolute-drop term for the last step.
- A "weeks to zero" outcome next to relapse rate, so speed and safety are judged together.
- A fast feature (last 2 days vs previous 5) so that daily decisions can beat weekly ones.
- Fitted weights instead of hand-set thresholds.
- Real puff logs from adult volunteers to check sections 1, 5 and 7.

## Reality check for any outcome number

The simulation follows motivated people who keep using the product for the whole run. Real trials include early
dropout, so every absolute number here is optimistic and only differences between rows are meaningful. For scale, the
nicotine patch (the same ladder in another form, 21 -> 14 -> 7 mg, user buys the next box) gets about one person in six
nicotine-free at six months in supported trials (Hartmann-Boyce et al. 2018, Cochrane: ~17 % vs ~10 % control), about
one in fourteen over the counter, and only a third to a half of users complete the course. In Hajek et al. 2019 (NEJM),
e-cigarettes beat patches for quitting smoking (18 % vs 9.9 % at one year) but about 80 % of the e-cigarette quitters
were still vaping a year later: the "stuck" outcome, measured.

## References to name

- Hughes JR (2007). Effects of abstinence from tobacco: valid symptoms and time course. Nicotine & Tobacco Research.
- Heatherton et al. (1991). The Fagerström Test for Nicotine Dependence. British Journal of Addiction. (time to first cigarette)
- Nahum-Shani I et al. (2018). Just-in-Time Adaptive Interventions (JITAIs) in mobile health. Annals of Behavioral Medicine.
- Benowitz NL (2010). Nicotine addiction. New England Journal of Medicine. (compensation, dependence)
- Donny EC et al. (2015). Randomized trial of reduced-nicotine standards for cigarettes. NEJM. (chronic craving at low nicotine)
- Hartmann-Boyce J et al. (2018). Nicotine replacement therapy versus control for smoking cessation. Cochrane Database of Systematic Reviews. (patch outcomes, course completion)
- Hajek P et al. (2019). A randomized trial of e-cigarettes versus nicotine-replacement therapy. NEJM. (quitters still vaping at one year)
- Dawkins LE et al. (2018). 'Real-world' compensatory behaviour with low nicotine concentration e-liquid. Addiction 113(10). Open data DOI 10.18744/LSBU.002952. (calibration)
