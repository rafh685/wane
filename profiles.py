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
    tonic_gain: float = 1.8       # how much RESTING craving rises as nicotine intake falls toward zero (0-10 scale)

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
            cues=[Cue((0, 1, 2, 3, 4), 7, 9, 1.0), Cue((5,), 20, 26, 1.5, alcohol=True)], tonic_gain=1.8),
    Profile("Diego", "22, student. Nothing before 19:00, then a dense evening block. Parties Friday and Saturday.",
            60, 0.35, 5.0, 0.12, 6.5, 0.20,
            wake={"wd": 9, "we": 11}, bed={"wd": 26, "we": 28},
            weekday_routine=_slots((19, 26, 3)),
            weekend_routine=_slots((14, 20, 1), (20, 28, 4)),
            weekend_factor=1.2,
            cues=[Cue((4, 5), 21, 27, 2.0, alcohol=True)], tonic_gain=1.5),
    Profile("Lucía", "31, quit twice on the ladder and relapsed both times. Compensates hard. Stressful Mondays.",
            150, 0.70, 7.0, 0.09, 6.0, 0.12,
            wake={"wd": 6.5, "we": 8.5}, bed={"wd": 23, "we": 24},
            weekday_routine=_slots((6, 9, 3), (9, 12, 1.5), (12, 13, 3), (13, 18, 1.5), (18, 23, 2.5)),
            weekend_routine=_slots((8, 24, 2)),
            cues=[Cue((0,), 8, 18, 1.0)], tonic_gain=2.7),
    Profile("Karim", "25, no vaping allowed at work 9-13 and 14-18. Fine Monday to Friday, doubles at the weekend with friends and alcohol.",
            120, 0.50, 6.0, 0.12, 6.0, 0.15,
            wake={"wd": 7.5, "we": 11}, bed={"wd": 24, "we": 27},
            weekday_routine=_slots((7, 9, 4), (13, 14, 4), (18, 24, 3)),
            weekend_routine=_slots((11, 18, 2), (18, 27, 4)),
            weekend_factor=2.0,
            cues=[Cue((4, 5), 21, 27, 2.0, alcohol=True)], tonic_gain=2.1),
    Profile("Ana", "34, motivated, low cravings, stable routine. Could go faster than a standard taper.",
            100, 0.15, 2.5, 0.22, 8.0, 0.10,
            wake={"wd": 6.5, "we": 8}, bed={"wd": 22.5, "we": 23.5},
            weekday_routine=_slots((6, 8, 3), (12, 13, 3), (17, 22, 2)),
            weekend_routine=_slots((8, 23, 1.5)),
            cues=[], tonic_gain=0.9),
]

START_MG = 20.0
REF_MG = 3.0        # below this, a cut is judged by its absolute size (see Vaper.day)
COMP_CAP = 2.5      # compensation saturates: LSBU data max total-puffing ratio 2.15 for a 67 % cut
PUFF_DUR_S = 3.4    # baseline puff duration, seconds (PR-ENDS field study mean 3.44 s)
WITHDRAWAL_SCALE = 0.6   # was 4.0; calibrated on the LSBU 18 -> 6 mg data (see docs/model_assumptions.md)
# how compensation splits between MORE puffs and LONGER puffs (LSBU: count x1.18, duration x1.26, total x1.47)
COMP_COUNT_SHARE, COMP_DUR_SHARE = 0.43, 0.57
BOUT_MODE = False   # True: puffs come in bouts (bouts.py, built on Ilian's generator) instead of hourly Poisson counts


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

    def _comp(self, dose_mg):
        """Total puffing compensation at a dose: 1 at 20 mg, rising as the dose falls, saturating at COMP_CAP.
        Sourced: LSBU real-world data, 18 -> 6 mg gave total puffing x1.47 median (IQR 1.34-1.68)."""
        dose_ratio = dose_mg / START_MG
        return min(COMP_CAP, 1 + self.p.elasticity * (1 / max(dose_ratio, 0.05) - 1) * 0.5)

    def day(self, dose_mg):
        p, rng = self.p, self.rng
        dow = self.t % 7
        dt = p.day_type(dow)

        # --- craving dynamics: a cut adds withdrawal (accumulates); it arrives over days and decays ---
        # Relative drop, but measured against at least REF_MG. At 20 mg a 12 % cut is a 12 % drop, unchanged.
        # Near zero the relative formula explodes (1 -> 0 would be a "100 % cut"); the body is adapted to an
        # absolute intake, and below REF_MG that intake is small, so the drop is scaled by REF_MG instead.
        # 1 -> 0 counts as a 33 % drop. 3 -> 0 is still a 100 % drop: the cliff is real if you jump from high.
        # Withdrawal follows the drop in nicotine INTAKE, not the drop in liquid strength: a person who compensates
        # by puffing more gets part of the nicotine back. Half of the compensation is credited (deeper, longer puffs
        # deliver less than proportionally).
        intake_now = dose_mg * self._comp(dose_mg) ** 0.5
        intake_last = self.last_dose * self._comp(self.last_dose) ** 0.5
        rel_drop = max(0.0, (intake_last - intake_now) / max(intake_last, REF_MG))
        if rel_drop > 0:
            self.pending_withdrawal += p.craving_sensitivity * rel_drop * WITHDRAWAL_SCALE
        release = self.pending_withdrawal * 0.35
        self.pending_withdrawal -= release
        self.craving += release
        # Resting craving is not fixed at 2: at low nicotine intake it sits higher for as long as intake is low
        # (reduced-nicotine cigarette trials: craving stays elevated for weeks at very low nicotine). The acute
        # withdrawal pulse decays toward THIS level, so the person never fully settles once the dose is low.
        tonic = 2.0 + p.tonic_gain * (1 - min(1.0, intake_now / START_MG))
        self.craving -= p.craving_decay * (self.craving - tonic)
        self.craving = float(np.clip(self.craving + rng.normal(0, 0.3), 0, 10))
        self.last_dose = dose_mg

        # --- how much today, overall: compensation splits into more puffs and longer puffs ---
        comp = self._comp(dose_mg)
        comp_count, comp_dur = comp ** COMP_COUNT_SHARE, comp ** COMP_DUR_SHARE
        day_factor = p.weekend_factor if dow in WEEKEND else 1.0
        rate = p.puffs_per_day * comp_count * day_factor * max(0.2, rng.normal(1, p.noise))
        puff_dur = PUFF_DUR_S * comp_dur * (1 + 0.03 * (self.craving - 2)) * max(0.5, rng.normal(1, 0.08))

        if BOUT_MODE:
            import bouts
            wake, bed = p.wake[dt], p.bed[dt]
            target = rate * (1 + 0.06 * (self.craving - 2) + 0.03 * p.cue_vector(dow).mean())
            plist = bouts.day(rng, self.craving, wake, bed, target, p.routine(dow), p.cue_vector(dow), dur_mult=comp_dur)
            sleep_hours = [h % 24 for h in range(int(np.ceil(bed)), int(np.ceil(bed)) + int(24 - (bed - wake)))]
            sleep_hours = [h for h in sleep_hours if h != int(wake)]
            times = [t for t, _, _ in plist]; puffs = len(times)
            counts = np.zeros(24, dtype=int)
            for t in times: counts[int(t) % 24] += 1
            night_puffs = sum(1 for t in times if int(t) % 24 in sleep_hours)
            first_awake = min([t for t in times if not (int(t) % 24 in sleep_hours)], default=wake + 0.5)
            ttfc_min = max(1.0, (first_awake - wake) * 60)
            puff_dur = float(np.mean([d for _, d, _ in plist])) if plist else PUFF_DUR_S
            flows = [f for _, _, f in plist]
            # relapse and return, same as below
            pressure = self.craving + 0.5 * p.cue_vector(dow).max()
            if pressure > p.relapse_threshold:
                excess = pressure - p.relapse_threshold
                daily_p = 0.04 + 0.06 * excess
                if dow in WEEKEND or p.has_alcohol(dow): daily_p *= 1.5
                if rng.random() < daily_p: self.relapsed = True
            self.t += 1
            return dict(day=self.t, dow=dow, dose=dose_mg, puffs=puffs, puff_dur=puff_dur, craving=self.craving,
                        night_puffs=night_puffs, ttfc_min=ttfc_min, hours=[int(c) for c in counts],
                        times=times, flows=flows, relapsed=self.relapsed)

        # --- spread over the hours: routine shape x (withdrawal + cue) ---
        routine = p.routine(dow)
        cue = p.cue_vector(dow)
        # hourly shape: routine, with cue hours boosted; the total for the day rises once with craving and cues
        shape = routine * (1 + 0.06 * cue)
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
        return dict(day=self.t, dow=dow, dose=dose_mg, puffs=puffs, puff_dur=puff_dur, craving=self.craving,
                    night_puffs=night_puffs, ttfc_min=ttfc_min, hours=[int(c) for c in counts],
                    times=times, relapsed=self.relapsed)
