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

# ---- realistic vapers (27 Sept 2026), from the research report and Codex's evidence review. REALISM = False
#      reproduces every earlier result exactly (the constants above).
REALISM = True
R_HALF_LIFE_SD = 0.30     # between-person spread of nicotine half-life: vaper SDs 29 to 35 % of the mean (St Helen
                          # 2016, 2020); CYP2A6 and sex give at least a threefold span (Benowitz 2006, 2016)
R_DUR_EXP = 1.2           # nicotine yield vs puff length: machine yields at 2, 4, 8 s give slopes 1.14 to 1.36
                          # (Talih 2015, our fit); the old 0.7 was the wrong direction
R_DUR_POD, R_DUR_TANK = 2.2, 3.2   # median puff length: pods 2.2 s (Dowd 2023), refillable 3.1 to 3.4 s (adults)
R_DUR_SD = 0.47           # puff-to-puff variation, CV about 0.5 (Dautzenberg 2015)
R_IPI_S, R_IPI_SD = 13.0, 0.7      # gap between puffs inside a bout: median 13 s, mean 16.6 s (Dautzenberg 2015)
R_P_SINGLE = 0.42         # 37 to 47 % of bouts are a single puff (60 s rule; Dautzenberg, Dowd)
R_BOUT_MEAN = 4.0         # mean puffs per bout: 4.0 refillable, 2.4 pods (60 s rule)
R_SESSION_CONT = 0.6      # a bout is followed by another within 1 to 5 min with this chance: sessions of about 10
                          # puffs, about 15 a day (Kosmider 2018: 10.2 puffs per session, 15.3 sessions)
R_NOISE_SCALE = 2.0       # day-to-day swings: profile noise 0.08 to 0.25 -> CV 0.16 to 0.5, median about 0.33
                          # (Dautzenberg 59 % within-person variance; Gao 2023 CVs 0 to 160 %)
R_NIGHT_WAKERS = 0.065    # base share; x exp(0.6 z) for harder people gives about 8 %: 7.1 to 9.5 % of adult vapers (Du 2019)
R_NIGHT_PER_NIGHT = 0.57  # wakers vape on about 4 nights a week (Du 2019, derived); more with withdrawal
R_EPS_MEAN, R_EPS_SD = 0.36, 0.15  # lasting compensation: puffing x (dose ratio)^-eps; LSBU 3x cut gives total puff
                          # time x1.50 (Dawkins 2018); it builds over days and persists (Cox 2021, Etter 2016)
R_COMP_TAU_H = 48.0       # compensation follows the per-puff dose with a 2-day time constant (Cox 2021, derived)
R_COMP_CAP = 2.2          # most total puffing recovered in the LSBU data: x2.15
R_COMP_FADE_BELOW = 0.1   # below about 2 mg/ml equivalent, compensation fades over about 3 weeks (extrapolated)
R_ACUTE = 0.3             # the old withdrawal-driven puffing kept at 30 %: craving still adds some puffs


class Person:
    """Hidden physiology built from a profiles.Profile (routine, cues, elasticity, craving decay, threshold)."""

    def __init__(self, profile, rng, half_life_h=HALF_LIFE_H, dur_exp=None, dur_share=None, irregular=None, slow_rng=None,
                 realism=None):
        p = profile
        self.realism = REALISM if realism is None else realism
        R = self.realism
        dur_share = (0.57 if R else 0.4) if dur_share is None else dur_share
        self.dur_share = dur_share
        self.dur_exp = (R_DUR_EXP if R else DUR_EXP) if dur_exp is None else dur_exp
        self.p = p
        e = float(np.clip(p.elasticity, 0.05, 0.85))
        a = e / (1 - e)                                  # steady state: puff rise = e * dose drop
        # how compensation splits between more bouts and longer draws. 0.4 was used for v3/v4; LSBU's
        # measured split (profiles.COMP_DUR_SHARE) is 0.57 on duration, used for the difficult-people tests
        acute = R_ACUTE if R else 1.0
        self.rate_gain, self.dur_gain = acute * (1 - dur_share) * a, acute * dur_share * a
        self.irregular = (True if R else False) if irregular is None else irregular
        self.noise_sd = p.noise * (R_NOISE_SCALE if R else 1.0)
        self._day_mult = {}
        tr = slow_rng or np.random.default_rng(0)
        # slow clock (relapse model v2): the long craving tail after a reduction. Urge strength among verified
        # abstainers falls exponentially with a half-life of about 19 weeks (Ussher et al. 2013, n = 452, 52 weeks),
        # so tau is about 190 d per person (lognormal spread, clipped 100 to 400 d). Receptor imaging normalises
        # faster (6 to 12 weeks, Cosgrove 2009), but craving, which drives relapse, follows the slower curve.
        # Drawn from its own stream so adding it left every earlier result bit-identical.
        self.tau_r_min = float(np.clip(190 * np.exp(tr.normal(0, 0.35)), 100, 400)) * 1440
        # realistic-vaper traits, all from the same separate stream (new draws only after the earlier ones)
        z_diff = -(p.relapse_threshold - 7.0) / 0.866
        self.tank = bool(tr.random() < 0.5)                              # device: refillable tank or pod
        self.night_waker = bool(tr.random() < float(np.clip(R_NIGHT_WAKERS * np.exp(0.6 * z_diff), 0.02, 0.3)))
        self.eps = float(np.clip(R_EPS_MEAN + 0.6 * (e - 0.47) + 0.1 * tr.normal(), -0.05, 0.75))
        hl_sd = R_HALF_LIFE_SD if R else 0.2
        self.q = 1.0                                     # per-puff dose the person has got used to (2-day memory)
        self.q_t = None
        self.low_days = 0.0
        self.tau_min = float(np.clip(1 / p.craving_decay, 3, 14)) * 1440
        self.half_life_h = half_life_h * float(np.exp(rng.normal(0, 0.2) * hl_sd / 0.2))
        self.decay = math.exp(-math.log(2) / (self.half_life_h * 60))
        self.kappa = 24 * math.log(2) / (max(p.puffs_per_day, 10) * self.half_life_h)   # daily mean C = 1 at baseline
        self.theta = 0.08 + 0.03 * (p.relapse_threshold - 5.5)                           # tolerated excess withdrawal (assumption)
        self.dur0 = ((R_DUR_TANK if self.tank else R_DUR_POD) if R else DUR_S) * float(np.exp(rng.normal(0, 0.15)))
        self.C = 0.0
        self.S = (5 + 2 * p.weekend_factor) / 7         # start near the weekly mean level, converges in warm-up
        self.S2 = self.S
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
            per_start = R_BOUT_MEAN / (1 - R_SESSION_CONT) if self.realism else BOUT_MEAN   # puffs per session start
            r = w / w.sum() * daily / per_start
            r = r * (1 + 0.25 * p.cue_vector(dow))
            self._rates[dow] = ([float(x) for x in r], [bool(x) for x in awake])
        return self._rates[dow]

    def day_multiplier(self, clock_day, rng):
        if not self.irregular:
            return 1.0
        if clock_day not in self._day_mult:
            sd = self.noise_sd
            self._day_mult[clock_day] = float(rng.lognormal(-0.5 * sd ** 2, sd))
        return self._day_mult[clock_day]

    def compensation(self):
        """Lasting compensation (realistic mode): (rate multiplier, puff-length multiplier) from the per-puff dose
        the person has got used to. Power law, split 43 % more puffs / 57 % longer puffs (LSBU)."""
        if not self.realism:
            return 1.0, 1.0
        eps = self.eps * math.exp(-self.low_days / 21.0)
        m = min(R_COMP_CAP, max(1.0, max(self.q, 0.05) ** (-eps)))
        return m ** (1 - self.dur_share), m ** self.dur_share

    def note_dose(self, t_h, d):
        """Update the per-puff dose the person has got used to (2-day exponential memory)."""
        if self.q_t is not None:
            a = 1 - math.exp(-(t_h - self.q_t) / R_COMP_TAU_H)
            self.q += a * (d - self.q)
        self.q_t = t_h

    def bout_size(self, rng):
        if not self.realism:
            return 1 + rng.poisson(BOUT_MEAN - 1)
        if rng.random() < R_P_SINGLE:
            return 1
        return 2 + int(rng.geometric(1 / (1 + (R_BOUT_MEAN - R_P_SINGLE) / (1 - R_P_SINGLE) - 2)) - 1)

    def slow_craving(self):
        """Craving from the slow clock: how far long-term adaptation still sits above current tolerance."""
        return max(0.0, self.S2 - self.S) / (self.S2 + S_FLOOR)

    def withdrawal(self):
        return max(0.0, self.S - self.C) / (self.S + S_FLOOR)


def sleep_hours(profile, dow):
    kind = "we" if dow in WEEKEND else "wd"
    return max(4.0, 24 - (profile.bed[kind] - profile.wake[kind]))


def minute_step(person, n_min=1):
    for _ in range(n_min):
        person.C *= person.decay
        person.S += (person.C - person.S) / person.tau_min
        person.S2 += (person.C - person.S2) / person.tau_r_min


def simulate(profile, make_controller, slow, taper_days=182, follow_days=28, seed=0, record=None, routine_change=None,
             physiology=None, stop_at_relapse=True, burn_in_days=0):
    """Run one person with one controller. Returns a result dict; per-puff rows go to `record` if given.

    make_controller(): a fresh controller with begin(baseline), start_day(k, u, weekend), dose(t_h), observe(t_h, dur_s, dose)
    slow: a SlowSchedule; gives the per-puff target level u for each device day (same for every controller)
    routine_change: (taper_day, other_profile) to swap in another person's routine and sleep times mid-taper
                    (new job, new term): the device's baseline habit goes stale and has to be relearned
    physiology: {"half_life_h": ..., "dur_exp": ..., "dur_share": ..., "irregular": ...} to make the hidden person
                differ from what controllers assume, compensate more through longer draws, or vary day to day
    A controller may set `delivery_cap_s` (after begin): the device stops adding nicotine after that many seconds
    of a draw, so a longer draw gives flavour only (a hardware option; its implementation stays private).
    stop_at_relapse=False keeps simulating after the first relapse draw (recorded as usual), so the full path's
    expected relapse risk, 1 - exp(-sum of daily hazards), can be compared without outcome noise
    burn_in_days: unrecorded days at full strength before the baseline weeks, so tolerance has settled before
                  anything is measured (removes the start-up spike found on 27 Sept; relapse model v2 uses 28).
    Every recorded day also stores its awake slow craving (`mean_r`) and whether it had an alcohol or a stress cue,
    for relapse.py (relapse model v2), which works on these daily records after the run.
    """
    rng = np.random.default_rng(seed)
    day_rng = np.random.default_rng(seed + 7919)          # separate stream, so irregular days leave the rest unchanged
    person = Person(profile, rng, slow_rng=np.random.default_rng(seed + 104729), **(physiology or {}))
    cap = None
    ctrl = make_controller()
    log = BaselineLog()
    total_days = WARMUP_DAYS + taper_days + follow_days
    minute = int(DAY_START_H * 60) - burn_in_days * 1440   # recording starts at 05:00 on a Monday
    end_minute = int(DAY_START_H * 60) + total_days * 1440
    day_w, day_r, day_awake_min = 0.0, 0.0, 0
    k = -burn_in_days
    u = 1.0
    out = dict(name=profile.name, relapsed=False, relapse_day=None, zero_day=None, days=[], cum_hazard=0.0)
    day_puffs, day_delivered, day_dose = 0, 0.0, 0.0
    next_boundary = minute + 1440

    def close_day(k):
        nonlocal day_w, day_r, day_awake_min, day_puffs, day_delivered, day_dose
        mean_w = day_w / max(day_awake_min, 1)
        if k >= 0:
            dow = k % 7
            cues = [c for c in profile.cues if dow in c.days]
            out["days"].append(dict(day=k, u=u, puffs=day_puffs, delivered=day_delivered, dose_sum=day_dose,
                                    mean_w=mean_w, mean_r=day_r / max(day_awake_min, 1), S=person.S,
                                    alcohol=any(c.alcohol for c in cues), stress=any(not c.alcohol for c in cues)))
        day_w, day_r, day_awake_min, day_puffs, day_delivered, day_dose = 0.0, 0.0, 0, 0, 0.0, 0.0
        return mean_w

    ctrl_started = False
    while minute < end_minute:
        clock_day, clock_min = divmod(minute, 1440)
        dow = clock_day % 7
        rates, awake = person.hourly_bout_rate(dow)
        h = clock_min // 60
        W = person.withdrawal()
        comp_rate, comp_dur = person.compensation()
        if awake[h]:
            day_w += W
            day_r += person.slow_craving()
            day_awake_min += 1
            lam = rates[h] * comp_rate * (1 + person.rate_gain * W) / 60 * person.day_multiplier(clock_day, day_rng)
        elif person.realism:
            # night waking is a trait (about 8 % of vapers, about 4 nights a week), stronger with withdrawal
            per_night = R_NIGHT_PER_NIGHT * (0.5 + W) if person.night_waker else 0.01
            lam = per_night / max(60 * sleep_hours(profile, dow), 60)
        else:
            lam = 0.03 * W * W / 60                   # night waking from withdrawal (assumption)
        if rng.random() < lam:
            t = minute / 60.0
            n_bouts = 1
            if person.realism and awake[h]:
                while rng.random() < R_SESSION_CONT:  # a session: bouts 1 to 5 min apart
                    n_bouts += 1
            ipi, ipi_sd, dur_sd = (R_IPI_S, R_IPI_SD, R_DUR_SD) if person.realism else (IPI_S, 0.5, 0.3)
            for b in range(n_bouts):
                if b:
                    t += rng.uniform(60, 300) / 3600
                n = person.bout_size(rng)
                for i in range(n):
                    if i:
                        t += rng.lognormal(math.log(ipi), ipi_sd) / 3600
                    W = person.withdrawal()
                    if k < WARMUP_DAYS:
                        d = 1.0
                    else:
                        d = float(ctrl.dose(t))
                        if not math.isfinite(d) or d < 0:
                            raise ValueError(f"{getattr(ctrl, 'name', ctrl)} returned an invalid dose {d}")
                    dur = float(np.clip(person.dur0 * comp_dur * (1 + person.dur_gain * W) * rng.lognormal(0, dur_sd),
                                        0.5, 10.0 if person.realism else 8.0))
                    eff = min(dur, cap) if (cap and k >= WARMUP_DAYS) else dur
                    delivered = d * (eff / person.dur0) ** person.dur_exp
                    person.C += delivered * person.kappa
                    if person.realism:
                        person.note_dose(t, d)
                    if 0 <= k < WARMUP_DAYS:
                        log.observe(t, dur, d)
                    elif k >= WARMUP_DAYS:
                        ctrl.observe(t, dur, d)
                    if record is not None and k >= 0:
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
            if person.realism and person.q < R_COMP_FADE_BELOW:
                person.low_days += 1
            mean_w = close_day(k)
            k += 1
            next_boundary += 1440
            if k == WARMUP_DAYS:
                person.baseline_w = {d["day"] % 7: d["mean_w"] for d in out["days"][WARMUP_DAYS - 7:WARMUP_DAYS]}
                excess_hist = []
                ctrl.begin(log.summary())
                slow.begin(log.summary())
                cap = getattr(ctrl, "delivery_cap_s", None)
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
                out["cum_hazard"] += hazard
                out["days"][-1]["hazard"] = hazard
                if rng.random() < hazard and not out["relapsed"]:
                    out["relapsed"], out["relapse_day"] = True, k - 1 - WARMUP_DAYS
                    if stop_at_relapse:
                        break
            if ctrl_started and k < total_days:
                last = out["days"][-1]
                u = slow.level(k - WARMUP_DAYS, last["puffs"])
                if u == 0.0 and out["zero_day"] is None:
                    out["zero_day"] = k - WARMUP_DAYS
                weekend = ((k + 0) % 7) in WEEKEND            # device day k starts on clock day k
                ctrl.start_day(k, u, weekend)
    out["expected_risk"] = 1 - math.exp(-out["cum_hazard"])
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
    bottom_steps: below LOW, reach zero in this many equal weekly steps instead (END_PLAN, 27 Sept 2026, night:
    front-loaded plan chosen in experiments/end_of_taper.py). Default None keeps every earlier result identical.
    """
    LOW = 0.15

    def __init__(self, weekly_cut=0.12, adaptive=False, bottom_steps=None):
        self.cut, self.adaptive, self.bottom_steps = weekly_cut, adaptive, bottom_steps
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
                    step = self.LOW / self.bottom_steps if self.bottom_steps else self.LOW * self.cut
                    self.u = 0.0 if self.u - step < step / 2 else self.u - step
        return self.u


END_PLAN = dict(weekly_cut=0.20, bottom_steps=14)   # 20 % a week to 15 % of the start (week 9), then 14 small steps: zero in week 23


class ManualTaper(SlowSchedule):
    """Traditional taper done by hand: what a vaper does today without Wane (realistic control, 27 Sept 2026).

    Bottles as sold in shops: 18 -> 12 -> 6 -> 3 -> 0 mg/ml, i.e. 1, 2/3, 1/3, 1/6 and 0 of the starting strength.
    The person buys the next weaker bottle every 4 to 7 weeks (drawn per step), so zero comes in week 22 on
    average, about when Wane's weekly plan gets there (week 23). Every puff has the bottle's strength.
    Imperfect adherence (assumption): when the first week on a new bottle brings back_rise more puffs than the
    last week before the step, a share back_p of people go back to the stronger bottle for back_weeks, then try
    the same step again; each step is undone at most once. The plan ends by end_day at the latest (Wane's window),
    so a late last step happens then. Fixed before running; no parameter was tuned on results.

    Stuck on the ladder (27 Sept 2026, night; Rafael: people get stuck on a bottle and never finish). Two routes,
    both assumptions (no trial measures stalling in self-tapers; ITC 4CV 2016-18: 85 % of vapers kept the same
    strength over two years, and 35 % of 0 mg vapers went back up to nicotine):
      stall        chance of never buying the next bottle, per step (to 12, 6, 3, 0 mg/ml): the person stays
                   on the current strength for good. The last step, to no nicotine at all, is the likeliest to stall
      give_up      failing the same step twice (puffing jumps again on the retry): with chance back_p the person
                   goes back to the stronger bottle and stops tapering
    With stuck people allowed there is no deadline (end_day=None): a slow ladder simply finishes late. Stall draws
    come from their own stream, so step timing is identical with and without them.
    stall=None, give_up=False, end_day=168 reproduces the first version (27 Sept, evening).
    """
    LADDER = (1.0, 2 / 3, 1 / 3, 1 / 6, 0.0)

    def __init__(self, seed=0, weeks=(4, 7), back_rise=0.15, back_p=0.5, back_weeks=2, end_day=None,
                 stall=(0.10, 0.10, 0.10, 0.25), give_up=True):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.stall_rng = np.random.default_rng(seed + 7)
        self.weeks, self.back_rise, self.back_p, self.back_weeks, self.end_day = weeks, back_rise, back_p, back_weeks, end_day
        self.stall, self.give_up = stall, give_up
        self.i = 0
        self.next_step = 7 * int(self.rng.integers(weeks[0], weeks[1] + 1))
        self.check_day = None           # end of the first week on a new bottle
        self.undone = set()
        self.tried = set()
        self.stuck = False
        self.step_backs = 0
        self.weekly = {}

    def level(self, taper_day, last_day_puffs):
        self.cur += last_day_puffs
        if taper_day == 0:
            self.cur = 0
            return self.LADDER[0]
        if taper_day % 7 == 0:
            self.weekly[taper_day] = self.cur
            self.cur = 0
            if taper_day == self.check_day:
                self.check_day = None
                before, after = self.weekly.get(taper_day - 7), self.weekly[taper_day]
                jumped = bool(before) and after > (1 + self.back_rise) * before
                if jumped and self.i not in self.undone and self.rng.random() < self.back_p:
                    self.undone.add(self.i)
                    self.i -= 1
                    self.step_backs += 1
                    self.next_step = taper_day + 7 * self.back_weeks
                elif jumped and self.i in self.undone and self.give_up and self.stall_rng.random() < self.back_p:
                    self.i -= 1                  # second failure of the same step: back up, and stop tapering
                    self.step_backs += 1
                    self.stuck = True
            if taper_day >= self.next_step and self.i < len(self.LADDER) - 1 and not self.stuck:
                first_try = self.i + 1 not in self.tried
                self.tried.add(self.i + 1)
                if first_try and self.stall and self.stall_rng.random() < self.stall[self.i]:
                    self.stuck = True            # never buys the next bottle
                else:
                    self.i += 1
                    self.check_day = taper_day + 7
                    self.next_step = taper_day + 7 * int(self.rng.integers(self.weeks[0], self.weeks[1] + 1))
        if self.end_day is not None and taper_day >= self.end_day:
            self.i = len(self.LADDER) - 1
        return self.LADDER[self.i]


def test_population(n=60, seed=12):
    """Population B (the one no model was fitted on) plus the five named people."""
    return list(PROFILES) + population(n, seed)
