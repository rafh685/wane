"""Five synthetic vapers. Each is the SAME behavioural model with different parameters.

The model has three sourced components:
  1. baseline topography  - puffs/day, time-of-day pattern (puff-topography studies)
  2. compensation         - when the dose drops x %, puffs rise by elasticity * x
                            (light-cigarette literature: 30-70 % of lost nicotine recovered)
  3. craving dynamics     - withdrawal rises after a cut, decays over ~1-2 weeks;
                            above a threshold for several days -> relapse

Nothing here is scripted per week. Behaviour EMERGES from parameters + the dose the
engine sets + random noise. The pilot replaces these population parameters with
measured individuals.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Profile:
    name: str
    story: str
    puffs_per_day: float       # baseline at 20 mg/ml
    elasticity: float          # puff increase per unit of RELATIVE dose drop (0 = none, 1 = full compensation)
    craving_sensitivity: float # how hard a cut hits craving (0-10 scale per unit relative drop)
    craving_decay: float       # fraction of excess craving that fades each day
    relapse_threshold: float   # craving level above which relapse risk becomes real
    weekend_factor: float      # multiplier on Sat/Sun puffs
    evening_share: float       # share of puffs after 19:00 (0.3 = spread, 0.9 = evening only)
    noise: float               # day-to-day randomness (coefficient of variation)


PROFILES = [
    Profile("Marta",  "27, office job. Heavy all-day vaper; the pod is never out of reach.",
            250, 0.35, 4.0, 0.15, 7.0, 1.0, 0.35, 0.10),
    Profile("Diego",  "22, student. Nothing before 19:00, then a dense evening block.",
            40,  0.30, 3.5, 0.15, 7.0, 1.2, 0.90, 0.20),
    Profile("Lucía",  "31, quit twice on the ladder and relapsed both times. Compensates hard.",
            150, 0.70, 6.0, 0.10, 6.5, 1.0, 0.40, 0.12),
    Profile("Karim",  "25, fine Monday to Friday, doubles at the weekend with friends and alcohol.",
            120, 0.40, 4.5, 0.15, 7.0, 2.0, 0.50, 0.15),
    Profile("Ana",    "34, motivated, low cravings, stable routine. Could go faster than a standard taper.",
            100, 0.15, 2.0, 0.25, 8.0, 1.0, 0.35, 0.10),
]

START_MG = 20.0


class Vaper:
    """One simulated person. Call .day(dose) once per day; it returns what they did."""

    def __init__(self, p: Profile, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.craving = 2.0          # resting craving, 0-10
        self.pending_withdrawal = 0.0
        self.last_dose = START_MG
        self.relapsed = False
        self.t = 0                  # day index

    def day(self, dose_mg):
        p = self.p
        # --- craving dynamics: a cut adds withdrawal; it decays daily ---
        rel_drop = max(0.0, (self.last_dose - dose_mg) / self.last_dose)
        if rel_drop > 0:
            self.pending_withdrawal = p.craving_sensitivity * rel_drop * 4   # spread over the coming days
        release = self.pending_withdrawal * 0.35                             # ~a third arrives each day
        self.pending_withdrawal -= release
        self.craving += release
        self.craving -= p.craving_decay * (self.craving - 2.0)     # decay toward resting 2.0
        self.craving = float(np.clip(self.craving + self.rng.normal(0, 0.3), 0, 10))
        self.last_dose = dose_mg

        # --- compensation: lower dose -> more puffs, scaled by elasticity ---
        dose_ratio = dose_mg / START_MG
        comp = 1 + p.elasticity * (1 / max(dose_ratio, 0.05) - 1) * 0.5
        weekend = p.weekend_factor if (self.t % 7) in (5, 6) else 1.0
        puffs = p.puffs_per_day * comp * weekend * (1 + 0.06 * (self.craving - 2))
        puffs = max(0, self.rng.normal(puffs, puffs * p.noise))

        # --- relapse: sustained craving above threshold -> chance of going back to 20 mg ---
        if self.craving > p.relapse_threshold and self.rng.random() < 0.05:
            self.relapsed = True

        # --- derived signals the device/sleeve could measure (no self-report needed) ---
        night_puffs = puffs * 0.02 * (1 + max(0, self.craving - 4))          # withdrawal shows at night
        ttfc_min = max(1, 30 * np.exp(-0.25 * (self.craving - 2)))            # time to first puff after waking
        evening = p.evening_share

        self.t += 1
        return dict(day=self.t, dose=dose_mg, puffs=puffs, craving=self.craving,
                    night_puffs=night_puffs, ttfc_min=ttfc_min, evening_share=evening,
                    relapsed=self.relapsed)
