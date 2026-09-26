"""Per-puff closed-loop simulator for the fast layer. SIMULATION ONLY.

Why it exists: engine.py and simulate.py work in whole days, so a per-puff dose never reaches the craving
or relapse model. Here every puff's dose changes the person's nicotine level minute by minute, the level
drives withdrawal, withdrawal drives puffing and relapse, and the controller only sees what a device sees.

The person (hidden from every controller)
  C   nicotine level, arbitrary units, rises with each delivered puff, decays with the person's half-life
  S   tolerance: the level the body has adapted to. Follows C over days (neuroadaptation, tau 3 to 14 days)
  W   withdrawal = max(0, S - C) / (S + S_FLOOR): relative to tolerance, fading as tolerance itself fades.
      Tapering works when S can follow C down without W building up
  bouts start at a rate set by the person's routine and cues, raised by W (compensation: more bouts)
  puff duration rises with W (compensation: longer puffs, the bigger channel in the LSBU data)
  relapse: daily hazard rises steeply once awake withdrawal sits above the person's own baseline

What a controller sees: puff timestamps, puff durations, and the doses it chose. Nothing else. No wake time,
no sleep label, no craving rating. It chooses a dose BEFORE the puff, so pulling harder never buys nicotine.

Parameters marked (assumption) are guesses, not measurements. Results from this file are relative
comparisons between controllers on synthetic people, never predictions about real users.
"""
import math

import numpy as np

from population import population
from profiles import PROFILES, WEEKEND

DAY_START_H = 5.0     # the device's budget day runs 05:00 to 05:00 (it cannot know wake time)
HALF_LIFE_H = 2.0     # population nicotine half-life (Benowitz 2009); each person varies around it
IPI_S = 9.8           # median in-session inter-puff interval, Robinson 2016 (data/external)
DUR_S = 2.1           # median puff duration, Robinson 2016
BOUT_MEAN = 8.0       # mean puffs per bout at baseline (assumption)
DUR_EXP = 0.7         # delivery grows sub-linearly with puff duration (assumption)
WARMUP_DAYS = 21      # baseline weeks at full strength, used by every controller to learn the person
S_FLOOR = 0.3         # below this tolerance, withdrawal fades out instead of staying relative (assumption)


class Person:
    """Hidden physiology built from a profiles.Profile (routine, cues, elasticity, craving decay, threshold)."""

    def __init__(self, profile, rng, half_life_h=HALF_LIFE_H, dur_exp=DUR_EXP):
        p = profile
        self.dur_exp = dur_exp
        self.p = p
        e = float(np.clip(p.elasticity, 0.05, 0.85))
        a = e / (1 - e)                                  # steady state: puff rise = e * dose drop
        self.rate_gain, self.dur_gain = 0.6 * a, 0.4 * a
        self.tau_min = float(np.clip(1 / p.craving_decay, 3, 14)) * 1440
        self.half_life_h = half_life_h * float(np.exp(rng.normal(0, 0.2)))
        self.decay = math.exp(-math.log(2) / (self.half_life_h * 60))
        self.kappa = 24 * math.log(2) / (max(p.puffs_per_day, 10) * self.half_life_h)   # daily mean C = 1 at baseline
        self.theta = 0.08 + 0.03 * (p.relapse_threshold - 5.5)                           # tolerated excess withdrawal (assumption)
        self.dur0 = DUR_S * float(np.exp(rng.normal(0, 0.15)))
        self.C = 0.0
        self.S = (5 + 2 * p.weekend_factor) / 7         # start near the weekly mean level, converges in warm-up
        self.baseline_w = None
        self._rates = {}

    def hourly_bout_rate(self, dow):
        """Bouts per hour for each clock hour of this weekday, before withdrawal."""
        if dow not in self._rates:
            p = self.p
            kind = "we" if dow in WEEKEND else "wd"
            wake, bed = p.wake[kind], p.bed[kind]
            hours = np.arange(24)
            awake = np.array([(wake <= h < bed) or (bed > 24 and h < bed - 24) for h in hours], dtype=float)
            w = p.routine(dow) * awake
            if w.sum() <= 0:
                w = awake
            daily = p.puffs_per_day * (p.weekend_factor if dow in WEEKEND else 1.0)
            r = w / w.sum() * daily / BOUT_MEAN
            r = r * (1 + 0.25 * p.cue_vector(dow))
            self._rates[dow] = ([float(x) for x in r], [bool(x) for x in awake])
        return self._rates[dow]

    def withdrawal(self):
        return max(0.0, self.S - self.C) / (self.S + S_FLOOR)


def minute_step(person, n_min=1):
    for _ in range(n_min):
        person.C *= person.decay
        person.S += (person.C - person.S) / person.tau_min


def simulate(profile, make_controller, slow, taper_days=182, follow_days=28, seed=0, record=None, routine_change=None,
             physiology=None):
    """Run one person with one controller. Returns a result dict; per-puff rows go to `record` if given.

    make_controller(): a fresh controller with begin(baseline), start_day(k, u, weekend), dose(t_h), observe(t_h, dur_s, dose)
    slow: a SlowSchedule; gives the per-puff target level u for each device day (same for every controller)
    routine_change: (taper_day, other_profile) to swap in another person's routine and sleep times mid-taper
                    (new job, new term): the device's baseline habit goes stale and has to be relearned
    physiology: {"half_life_h": ..., "dur_exp": ...} to make the hidden person differ from what controllers assume
    """
    rng = np.random.default_rng(seed)
    person = Person(profile, rng, **(physiology or {}))
    ctrl = make_controller()
    log = BaselineLog()
    total_days = WARMUP_DAYS + taper_days + follow_days
    minute = int(DAY_START_H * 60)                    # the simulation starts at 05:00 on a Monday
    end_minute = minute + total_days * 1440
    day_w, day_awake_min = 0.0, 0
    k = 0
    u = 1.0
    out = dict(name=profile.name, relapsed=False, relapse_day=None, zero_day=None, days=[])
    day_puffs, day_delivered, day_dose = 0, 0.0, 0.0
    next_boundary = minute + 1440

    def close_day(k):
        nonlocal day_w, day_awake_min, day_puffs, day_delivered, day_dose
        mean_w = day_w / max(day_awake_min, 1)
        out["days"].append(dict(day=k, u=u, puffs=day_puffs, delivered=day_delivered, dose_sum=day_dose,
                                mean_w=mean_w, S=person.S))
        day_w, day_awake_min, day_puffs, day_delivered, day_dose = 0.0, 0, 0, 0.0, 0.0
        return mean_w

    ctrl_started = False
    while minute < end_minute:
        clock_day, clock_min = divmod(minute, 1440)
        dow = clock_day % 7
        rates, awake = person.hourly_bout_rate(dow)
        h = clock_min // 60
        W = person.withdrawal()
        if awake[h]:
            day_w += W
            day_awake_min += 1
            lam = rates[h] * (1 + person.rate_gain * W) / 60
        else:
            lam = 0.03 * W * W / 60                   # night waking from withdrawal (assumption)
        if rng.random() < lam:
            n = 1 + rng.poisson(BOUT_MEAN - 1)
            t = minute / 60.0
            for i in range(n):
                if i:
                    t += rng.lognormal(math.log(IPI_S), 0.5) / 3600
                W = person.withdrawal()
                if k < WARMUP_DAYS:
                    d = 1.0
                else:
                    d = float(ctrl.dose(t))
                    if not math.isfinite(d) or d < 0:
                        raise ValueError(f"{getattr(ctrl, 'name', ctrl)} returned an invalid dose {d}")
                dur = float(np.clip(person.dur0 * (1 + person.dur_gain * W) * rng.lognormal(0, 0.3), 0.5, 8.0))
                delivered = d * (dur / person.dur0) ** person.dur_exp
                person.C += delivered * person.kappa
                if k < WARMUP_DAYS:
                    log.observe(t, dur, d)
                else:
                    ctrl.observe(t, dur, d)
                if record is not None:
                    record.append((k, t, dur, d))
                day_puffs += 1
                day_delivered += delivered
                day_dose += d
            skip = max(1, int(math.ceil((t - minute / 60.0) * 60)))
            minute_step(person, skip)
            minute += skip
        else:
            minute_step(person)
            minute += 1

        if minute >= next_boundary:
            mean_w = close_day(k)
            k += 1
            next_boundary += 1440
            if k == WARMUP_DAYS:
                person.baseline_w = {d["day"] % 7: d["mean_w"] for d in out["days"][WARMUP_DAYS - 7:WARMUP_DAYS]}
                excess_hist = []
                ctrl.begin(log.summary())
                slow.begin(log.summary())
                ctrl_started = True
            if routine_change and k == WARMUP_DAYS + routine_change[0]:
                q = routine_change[1]
                p = profile
                person.p = type(p)(**{**p.__dict__, "wake": q.wake, "bed": q.bed, "weekday_routine": q.weekday_routine,
                                      "weekend_routine": q.weekend_routine, "weekend_factor": q.weekend_factor})
                person._rates = {}
            if k > WARMUP_DAYS:
                excess_hist.append(mean_w - person.baseline_w[(k - 1) % 7])     # against the same weekday at baseline
                excess = float(np.mean(excess_hist[-3:]))
                hazard = 0.0003 + 0.03 / (1 + math.exp(-(excess - person.theta) / 0.015))
                if rng.random() < hazard:
                    out["relapsed"], out["relapse_day"] = True, k - 1 - WARMUP_DAYS
                    break
            if ctrl_started and k < total_days:
                last = out["days"][-1]
                u = slow.level(k - WARMUP_DAYS, last["puffs"])
                if u == 0.0 and out["zero_day"] is None:
                    out["zero_day"] = k - WARMUP_DAYS
                weekend = ((k + 0) % 7) in WEEKEND            # device day k starts on clock day k
                ctrl.start_day(k, u, weekend)
    out["baseline_w"] = person.baseline_w
    out["baseline_puffs"] = float(np.mean([d["puffs"] for d in out["days"][:WARMUP_DAYS]]))
    return out


class BaselineLog:
    """What the device learns about the person during the full-strength baseline weeks."""

    def __init__(self):
        self.puffs = []

    def observe(self, t, dur, dose):
        self.puffs.append((t, dur))

    def summary(self):
        t = np.array([p[0] for p in self.puffs])
        dur = np.array([p[1] for p in self.puffs])
        dev_day = np.floor((t - DAY_START_H) / 24).astype(int)
        clock_h = np.floor(t % 24).astype(int)
        clock_day = np.floor(t / 24).astype(int)
        hist = {"wd": np.zeros(24), "we": np.zeros(24)}
        n_days = {"wd": 0, "we": 0}
        per_day = {"wd": [], "we": []}
        for k in range(WARMUP_DAYS):
            kind = "we" if k % 7 in WEEKEND else "wd"
            n_days[kind] += 1
            per_day[kind].append(int((dev_day == k).sum()))
        for kind in ("wd", "we"):
            mask = np.array([(cd % 7 in WEEKEND) == (kind == "we") for cd in clock_day])
            hist[kind] = np.bincount(clock_h[mask], minlength=24) / max(n_days[kind], 1)
        gaps = np.diff(np.sort(t)) if len(t) > 1 else np.array([0.0])
        longest = []
        for k in range(WARMUP_DAYS):
            sel = (dev_day[1:] == k)
            longest.append(float(gaps[sel].max()) if sel.any() else 0.0)
        return dict(
            puffs_per_day=float(np.mean(per_day["wd"] + per_day["we"])),
            puffs_wd=float(np.mean(per_day["wd"])), puffs_we=float(np.mean(per_day["we"])),
            hourly={"wd": hist["wd"], "we": hist["we"]},
            dur=float(np.median(dur)),
            longest_gap_h=float(np.median(longest)),
            puffs=list(self.puffs),        # raw warm-up log, so a controller can replay what the device saw
        )


class SlowSchedule:
    """The slow layer, identical for every fast controller so the comparison isolates the fast layer.

    12 %/week relative cuts every Monday; below LOW, fixed absolute steps to zero (like engine.apply_cut).
    adaptive=True: hold the level for a week when last week's puffing rose more than 10 % over the week before
    (device-observable compensation brake, same rule for every controller).
    """
    LOW = 0.15

    def __init__(self, weekly_cut=0.12, adaptive=False):
        self.cut, self.adaptive = weekly_cut, adaptive
        self.u = 1.0
        self.week_puffs = []
        self.cur = 0

    def begin(self, baseline):
        self.base_week = baseline["puffs_per_day"] * 7

    def level(self, taper_day, last_day_puffs):
        self.cur += last_day_puffs
        if taper_day == 0:
            self.cur = 0
            self.prev_week = self.base_week
            return self.u
        if taper_day % 7 == 0:
            week = self.cur
            self.cur = 0
            hold = self.adaptive and week > 1.10 * self.prev_week
            self.prev_week = week if not hold else self.prev_week
            if not hold and self.u > 0:
                if self.u > self.LOW:
                    self.u = max(self.LOW, self.u * (1 - self.cut))
                else:
                    step = self.LOW * self.cut
                    self.u = 0.0 if self.u - step < step / 2 else self.u - step
        return self.u


def test_population(n=60, seed=12):
    """Population B (the one no model was fitted on) plus the five named people."""
    return list(PROFILES) + population(n, seed)
