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

## What is missing

- An end rule so runs can reach 0 mg/ml, and an absolute-drop term for the last step.
- A "weeks to zero" outcome next to relapse rate, so speed and safety are judged together.
- A fast feature (last 2 days vs previous 5) so that daily decisions can beat weekly ones.
- Fitted weights instead of hand-set thresholds.
- Real puff logs from adult volunteers to check sections 1, 5 and 7.

## References to name

- Hughes JR (2007). Effects of abstinence from tobacco: valid symptoms and time course. Nicotine & Tobacco Research.
- Heatherton et al. (1991). The Fagerström Test for Nicotine Dependence. British Journal of Addiction. (time to first cigarette)
- Nahum-Shani I et al. (2018). Just-in-Time Adaptive Interventions (JITAIs) in mobile health. Annals of Behavioral Medicine.
- Benowitz NL (2010). Nicotine addiction. New England Journal of Medicine. (compensation, dependence)
