"""Five synthetic vapers, v2. Each is the SAME behavioural model with different parameters.

What a person carries
  1. baseline topography  - puffs/day and an HOURLY routine for weekdays and weekends
                            (wake, work with or without access, lunch, evening, bedtime)
  2. cues                 - hours in the week when craving is up for reasons that are NOT withdrawal:
                            coffee, alcohol, stress, social. They raise puffing and lapse risk on those days.
                            To an engine that only counts puffs, a cue looks like withdrawal. That is the confound
                            the engine must learn to ignore.
  3. compensation         - when the dose drops x %, puffs rise by elasticity * x (30-70 % recovered, light-cigarette literature)
  4. craving dynamics     - withdrawal rises after a cut, decays over ~1-2 weeks; above a threshold -> relapse risk

Output per day: puff TIMESTAMPS (hours, 0-24), plus the daily summaries derived from them. This is the same shape
of data a counter app or a metering device produces, so the engine's feature code is the real feature code.

Nothing is scripted per week. Behaviour EMERGES from parameters + the dose the engine sets + random noise.
The pilot replaces these population parameters with measured individuals.
"""
from dataclasses import dataclass, field
import numpy as np

# day index t (0-based) -> weekday; 0 = Monday ... 5 = Saturday, 6 = Sunday
WEEKEND = (5, 6)


def _slots(*spans):
    """Build a 24-hour weight vector from (start, end, weight) spans. Hours outside all spans get 0."""
    w = np.zeros(24)
    for a, b, k in spans:
        for h in range(a, b):
            w[h % 24] += k
    return w


@dataclass
class Cue:
    days: tuple            # weekdays (0 = Mon) on which the cue happens
    start: int             # hour
    end: int               # hour (may exceed 24 to run past midnight)
    boost: float           # added to "effective craving" during those hours (puffing) and to lapse risk that day
    alcohol: bool = False  # alcohol cues multiply lapse risk


@dataclass
class Profile:
    name: str
    story: str
    puffs_per_day: float       # baseline at 20 mg/ml
    elasticity: float          # puff increase per unit of RELATIVE dose drop (0 = none, 1 = full compensation)
    craving_sensitivity: float # how hard a cut hits craving (0-10 scale per unit relative drop)
    craving_decay: float       # fraction of excess craving that fades each day
    relapse_threshold: float   # craving level above which relapse risk becomes real
    noise: float               # extra day-to-day randomness (coefficient of variation on the daily rate)
    wake: dict                 # {"wd": hour, "we": hour}
    bed: dict                  # {"wd": hour, "we": hour}  (may exceed 24)
    weekday_routine: np.ndarray   # 24 weights, relative puffing intensity per hour on Mon-Fri
    weekend_routine: np.ndarray   # 24 weights, Sat-Sun
    weekend_factor: float = 1.0   # multiplier on total puffs Sat/Sun
    cues: list = field(default_factory=list)

    def routine(self, dow):
        return self.weekend_routine if dow in WEEKEND else self.weekday_routine

    def day_type(self, dow):
        return "we" if dow in WEEKEND else "wd"

    def cue_vector(self, dow):
        """24-vector of cue boost per hour for this weekday; cues that run past midnight spill into the next day."""
        v = np.zeros(24)
        for c in self.cues:
            if dow in c.days:
                for h in range(c.start, c.end):
                    if h < 24:
                        v[h] += c.boost
            prev = (dow - 1) % 7
            if prev in c.days and c.end > 24:
                for h in range(0, c.end - 24):
                    v[h] += c.boost
        return v

    def has_alcohol(self, dow):
        return any(c.alcohol and dow in c.days for c in self.cues)


PROFILES = [
    Profile("Marta", "27, office job. Heavy all-day vaper; the pod is never out of reach. Coffee at 8, wine on Saturdays.",
            250, 0.45, 5.5, 0.12, 6.5, 0.10,
            wake={"wd": 7, "we": 9}, bed={"wd": 23.5, "we": 25},
            weekday_routine=_slots((7, 9, 3), (9, 13, 1.5), (13, 14, 3), (14, 18, 1.5), (18, 24, 3)),
            weekend_routine=_slots((9, 12, 2), (12, 20, 2), (20, 25, 3)),
            cues=[Cue((0, 1, 2, 3, 4), 7, 9, 1.0), Cue((5,), 20, 26, 1.5, alcohol=True)]),
    Profile("Diego", "22, student. Nothing before 19:00, then a dense evening block. Parties Friday and Saturday.",
            40, 0.35, 5.0, 0.12, 6.5, 0.20,
            wake={"wd": 9, "we": 11}, bed={"wd": 26, "we": 28},
            weekday_routine=_slots((19, 26, 3)),
            weekend_routine=_slots((14, 20, 1), (20, 28, 4)),
            weekend_factor=1.2,
            cues=[Cue((4, 5), 21, 27, 2.0, alcohol=True)]),
    Profile("Lucía", "31, quit twice on the ladder and relapsed both times. Compensates hard. Stressful Mondays.",
            150, 0.70, 7.0, 0.09, 6.0, 0.12,
            wake={"wd": 6.5, "we": 8.5}, bed={"wd": 23, "we": 24},
            weekday_routine=_slots((6, 9, 3), (9, 12, 1.5), (12, 13, 3), (13, 18, 1.5), (18, 23, 2.5)),
            weekend_routine=_slots((8, 24, 2)),
            cues=[Cue((0,), 8, 18, 1.0)]),
    Profile("Karim", "25, no vaping allowed at work 9-13 and 14-18. Fine Monday to Friday, doubles at the weekend with friends and alcohol.",
            120, 0.50, 6.0, 0.12, 6.0, 0.15,
            wake={"wd": 7.5, "we": 11}, bed={"wd": 24, "we": 27},
            weekday_routine=_slots((7, 9, 4), (13, 14, 4), (18, 24, 3)),
            weekend_routine=_slots((11, 18, 2), (18, 27, 4)),
            weekend_factor=2.0,
            cues=[Cue((4, 5), 21, 27, 2.0, alcohol=True)]),
    Profile("Ana", "34, motivated, low cravings, stable routine. Could go faster than a standard taper.",
            100, 0.15, 2.5, 0.22, 8.0, 0.10,
            wake={"wd": 6.5, "we": 8}, bed={"wd": 22.5, "we": 23.5},
            weekday_routine=_slots((6, 8, 3), (12, 13, 3), (17, 22, 2)),
            weekend_routine=_slots((8, 23, 1.5)),
            cues=[]),
]

START_MG = 20.0


class Vaper:
    """One simulated person. Call .day(dose) once per day; it returns what they did, with puff timestamps."""

    def __init__(self, p: Profile, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.craving = 2.0          # resting craving, 0-10  (withdrawal, internal)
        self.pending_withdrawal = 0.0
        self.last_dose = START_MG
        self.relapsed = False
        self.t = 0                  # day index; t % 7 = weekday, 0 = Monday

    def day(self, dose_mg):
        p, rng = self.p, self.rng
        dow = self.t % 7
        dt = p.day_type(dow)

        # --- craving dynamics: a cut adds withdrawal (accumulates); it arrives over days and decays ---
        rel_drop = max(0.0, (self.last_dose - dose_mg) / self.last_dose)
        if rel_drop > 0:
            self.pending_withdrawal += p.craving_sensitivity * rel_drop * 4
        release = self.pending_withdrawal * 0.35
        self.pending_withdrawal -= release
        self.craving += release
        self.craving -= p.craving_decay * (self.craving - 2.0)
        self.craving = float(np.clip(self.craving + rng.normal(0, 0.3), 0, 10))
        self.last_dose = dose_mg

        # --- how much today, overall ---
        dose_ratio = dose_mg / START_MG
        comp = 1 + p.elasticity * (1 / max(dose_ratio, 0.05) - 1) * 0.5      # compensation
        day_factor = p.weekend_factor if dow in WEEKEND else 1.0
        rate = p.puffs_per_day * comp * day_factor * max(0.2, rng.normal(1, p.noise))

        # --- spread over the hours: routine shape x (withdrawal + cue) ---
        routine = p.routine(dow)
        cue = p.cue_vector(dow)
        shape = routine * (1 + 0.06 * (self.craving - 2) + 0.06 * cue)
        hourly = rate * shape / max(shape.sum(), 1e-9) * (1 + 0.06 * (self.craving - 2) + 0.03 * cue.mean())
        counts = rng.poisson(np.clip(hourly, 0, None))

        # --- withdrawal shows at night: waking up to puff inside the sleep window ---
        wake, bed = p.wake[dt], p.bed[dt]
        sleep_hours = [h % 24 for h in range(int(np.ceil(bed)), int(np.ceil(bed)) + int(24 - (bed - wake)))]
        sleep_hours = [h for h in sleep_hours if h != int(wake)]
        night_mean = rate * 0.02 * (1 + max(0.0, self.craving - 4))
        night_counts = rng.multinomial(rng.poisson(night_mean), np.ones(len(sleep_hours)) / len(sleep_hours)) if sleep_hours else []
        for h, c in zip(sleep_hours, night_counts):
            counts[h] += c

        # --- time to first puff after waking: shorter when craving is up (Fagerström logic) ---
        ttfc_min = max(1.0, rng.exponential(30 * np.exp(-0.25 * (self.craving - 2))))
        first = wake + ttfc_min / 60
        times = []
        for h in range(24):
            if counts[h]:
                times.extend(h + rng.uniform(0, 1, counts[h]))
        times = [t for t in times if not (wake <= t < first)]     # nothing between waking and the first puff
        times.append(first)
        times.sort()
        puffs = len(times)
        night_puffs = sum(1 for t in times if int(t) % 24 in sleep_hours)

        # --- relapse: craving (plus cue pressure) above threshold -> daily chance; alcohol and weekends add temptation ---
        pressure = self.craving + 0.5 * cue.max()
        if pressure > p.relapse_threshold:
            excess = pressure - p.relapse_threshold
            daily_p = 0.04 + 0.06 * excess
            if dow in WEEKEND or p.has_alcohol(dow):
                daily_p *= 1.5
            if rng.random() < daily_p:
                self.relapsed = True

        self.t += 1
        return dict(day=self.t, dow=dow, dose=dose_mg, puffs=puffs, craving=self.craving,
                    night_puffs=night_puffs, ttfc_min=ttfc_min, hours=[int(c) for c in counts],
                    times=times, relapsed=self.relapsed)
