"""Fast layer v3: a neural network with frozen base weights and an unfrozen head that adapts to each user.

Rafael's design (26 Sept 2026): the base network carries what is common to all vapers, learned offline;
the unfrozen part changes as this person vapes; the dose is regulated on the spot, at every puff.

At every puff, BEFORE the puff happens (so pulling harder never buys nicotine):
  1. state      device-only features: estimated nicotine level from the doses it gave (population half-life),
                a tolerance proxy (its own 3-day average of that level), time since the last puff, puffing in the
                last 10 min / hour / day against this person's own baseline habit for that hour, puff length trend,
                clock hour, weekend, today's budget used. No wake time, no sleep label, no time to first puff.
  2. network    frozen base (2 hidden layers, trained on synthetic population A) -> 16 features
                unfrozen head (16 -> 1) predicts how many puffs this person has left today
  3. optimiser  spread what is left of today's budget over the predicted remaining puffs, and give more to
                puffs where the nicotine estimate sits furthest below tolerance (the first puffs after a gap:
                withdrawal relief) and less to chain puffs (habit)
  4. safety     fixed rules the network cannot override: never above the starting per-puff level, never more
                than 2x today's target per puff, never above what is left of today's budget, smooth changes
                inside a bout, a finite fallback if the network misbehaves

Learning: every night the head is updated with the day's real answer (how many puffs actually followed each
puff), by recursive least squares with forgetting, starting from the population head, and it can only move a
bounded distance from it. The base weights never change after training.

SIMULATION ONLY: trained and tested on synthetic people from puffsim.py. Dose units are fractions of the
starting per-puff level, not mg, not delivered nicotine; a device calibration maps them to hardware.
"""
import bisect
import json
import math
import pathlib
from collections import deque

import numpy as np

from puffsim import DAY_START_H, WEEKEND

FEATURES = ["c_rel", "s_rel", "relief", "since_last", "last10", "hour_vs_habit", "today_vs_habit", "day_vs_habit",
            "dur_trend", "hour_sin", "hour_cos", "weekend", "habit_rest", "budget_used", "level", "gap_vs_habit"]
HALF_LIFE_H = 2.0
WEIGHTS_FILE = pathlib.Path(__file__).with_name("puff_nn_base.json")


class DeviceState:
    """Everything the device can know, updated after each puff. Shared by the controllers and the trainer."""

    def __init__(self, baseline):
        self.b = baseline
        self.decay_per_h = math.log(2) / HALF_LIFE_H
        self.c = 0.0                      # estimated level from own doses (units: dose x duration factor)
        self.c_t = None
        self.s = None                     # 3-day moving average of c, the tolerance proxy
        self.times = []                   # every puff time (searched with bisect)
        self.durs = deque(maxlen=20)
        self.day_times = []
        self.day = None
        self.u = 1.0
        self.budget = 0.0
        self.used = 0.0
        self.weekend = False
        self.c_base = baseline.get("c_base", 1.0)

    # --- the device's own nicotine estimate: its doses, the measured puff length, a population half-life
    def level_at(self, t):
        if self.c_t is None:
            return 0.0
        return self.c * math.exp(-self.decay_per_h * (t - self.c_t))

    def advance(self, t):
        c = self.level_at(t)
        if self.c_t is not None:
            dt = t - self.c_t
            a = 1 - math.exp(-dt / 72.0)                            # 3-day time constant
            self.s = c if self.s is None else self.s + a * (c - self.s)
        self.c, self.c_t = c, t

    def start_day(self, k, u, weekend):
        self.day, self.u, self.weekend = k, u, weekend
        per_day = self.b["puffs_we"] if weekend else self.b["puffs_wd"]
        self.budget = u * per_day
        self.used = 0.0
        self.day_times = []

    def observe(self, t, dur, dose):
        self.advance(t)
        self.c += dose * (dur / self.b["dur"]) ** 0.7
        self.times.append(t)
        self.durs.append(dur)
        self.day_times.append(t)
        self.used += dose

    def habit(self, t):
        hist = self.b["hourly"]["we" if self.weekend else "wd"]
        h = t % 24
        hi = int(h)
        this_hour = hist[hi]
        # expected puffs from now to the end of the device day (05:00), from the baseline habit
        day_end = (math.floor((t - DAY_START_H) / 24) + 1) * 24 + DAY_START_H
        rest = hist[hi] * (1 - (h - hi))
        cur = hi + 1
        tt = t - h + cur
        while tt < day_end - 1e-9:
            rest += hist[cur % 24]
            cur += 1
            tt += 1
        done = self.b["puffs_we" if self.weekend else "puffs_wd"] - rest
        return this_hour, rest, max(done, 0.0)

    def features(self, t):
        c = self.level_at(t)
        s = self.s if self.s is not None else c
        cb = max(self.c_base, 1e-6)
        n = len(self.times)
        i24 = bisect.bisect_right(self.times, t - 24)
        n10 = n - bisect.bisect_right(self.times, t - 1 / 6)
        n60 = n - bisect.bisect_right(self.times, t - 1)
        n24 = n - i24
        since = (t - self.times[-1]) * 60 if self.times else 600.0
        this_hour, rest, done = self.habit(t)
        window = self.times[i24:]
        longest = max((b - a for a, b in zip(window, window[1:])), default=0.0)
        dur_trend = (np.mean(self.durs) / self.b["dur"] - 1) if self.durs else 0.0
        h = 2 * math.pi * (t % 24) / 24
        return np.array([
            c / cb, s / cb, min(max((s - c) / (s + 1e-6), 0.0), 1.0) if s > 0 else 0.0,
            math.log1p(since) / 6, n10 / 10,
            math.log((n60 + 1) / (this_hour + 1)),
            math.log((len(self.day_times) + 1) / (done + 1)),
            math.log((n24 + 1) / (self.b["puffs_per_day"] + 1)),
            dur_trend, math.sin(h), math.cos(h), float(self.weekend),
            math.log1p(rest) / 6, self.used / max(self.budget, 1e-6) if self.budget > 0 else 1.0,
            self.u, longest / max(self.b["longest_gap_h"], 0.5) - 1,
        ])

    def relief(self, t):
        c = self.level_at(t)
        s = self.s if self.s is not None else c
        return min(max((s - c) / (s + 1e-6), 0.0), 1.0) if s > 0 else 0.0


def baseline_with_level(baseline):
    """Add the device's baseline level estimate: mean of its own c estimate at full strength."""
    b = dict(baseline)
    # at full strength every puff is dose 1; mean level = puffs/day x mean duration factor x mean life / 24
    b["c_base"] = b["puffs_per_day"] * (HALF_LIFE_H / math.log(2)) / 24
    return b


# ---------------------------------------------------------------- the network

class BaseNet:
    """Frozen feature extractor: standardise -> tanh(32) -> tanh(16). Trained once, never updated online."""

    def __init__(self, params):
        self.mu, self.sd = np.array(params["mu"]), np.array(params["sd"])
        self.W1, self.b1 = np.array(params["W1"]), np.array(params["b1"])
        self.W2, self.b2 = np.array(params["W2"]), np.array(params["b2"])
        self.head = np.array(params["head"])           # 17: 16 weights + bias, the population head
        for a in (self.W1, self.b1, self.W2, self.b2):
            a.setflags(write=False)

    def hidden(self, x):
        z = (x - self.mu) / self.sd
        return np.tanh(np.tanh(z @ self.W1 + self.b1) @ self.W2 + self.b2)

    @classmethod
    def load(cls, path=WEIGHTS_FILE):
        return cls(json.loads(pathlib.Path(path).read_text()))


class AdaptiveHead:
    """The unfrozen part: 16 -> 1 linear head, updated online by recursive least squares.

    Starts at the population head. Forgetting lets it follow a person whose habits change, and the uncertainty
    it may regain is capped at the prior's (bounded forgetting), so directions the data never excites cannot
    blow up. Drift from the population head is bounded too: the head personalises, it cannot run away."""

    def __init__(self, prior, p0=0.02, noise=0.2, forget=0.995, max_drift=2.0):
        self.theta0 = np.array(prior, dtype=float)
        self.theta = self.theta0.copy()
        self.P = np.eye(len(prior)) * p0
        self.p_cap = p0 * len(prior)
        self.noise, self.forget, self.max_drift = noise, forget, max_drift
        self.updates = 0

    def predict(self, h):
        return float(np.dot(self.theta[:-1], h) + self.theta[-1])

    def update(self, h, y):
        x = np.append(h, 1.0)
        Px = self.P @ x
        k = Px / (self.noise + x @ Px)
        self.theta = self.theta + k * (y - x @ self.theta)
        self.P = (self.P - np.outer(k, Px)) / self.forget
        tr = np.trace(self.P)
        if tr > self.p_cap:
            self.P *= self.p_cap / tr
        drift = self.theta - self.theta0
        n = np.linalg.norm(drift)
        if n > self.max_drift:
            self.theta = self.theta0 + drift * (self.max_drift / n)
        self.updates += 1


# ---------------------------------------------------------------- controllers (same interface as puffsim expects)

class Flat:
    """Traditional taper: every puff at today's level, no daily cap. What a weaker bottle does."""
    name = "flat (weaker bottle)"

    def begin(self, baseline):
        pass

    def start_day(self, k, u, weekend):
        self.u = u

    def dose(self, t):
        return self.u

    def observe(self, t, dur, dose):
        pass


class CodexRules:
    """Codex's deterministic per-puff budget allocator (puff_controller.py, 25 Sept 2026), unchanged."""
    name = "rules (Codex budget)"

    def begin(self, baseline):
        from puff_controller import PuffBudgetController      # lives in the main checkout, see AGENTS.md
        self.c = PuffBudgetController(baseline["puffs_per_day"])
        self.last = None

    def start_day(self, k, u, weekend):
        self.k = k
        self.c.start_day(k, u)

    def dose(self, t):
        t = t - DAY_START_H
        hi = (self.k + 1) * 24 - 1e-3
        if t >= hi:                                            # a bout that crosses 05:00 stays in its day
            t = hi if self.last is None or self.last < hi else self.last + 1e-7
        self.last = t
        return self.c.allocate(t).dose_units

    def observe(self, t, dur, dose):
        pass


class OnTheSpot:
    """The fast layer. demand='nn' uses the network (frozen base + adaptive head); demand='habit' uses only the
    person's baseline habit histogram, to measure what the network adds. adapt=False freezes the head too."""

    RELIEF_LO, RELIEF_HI = 0.6, 1.4          # dose multiplier from no relief need to full relief need
    CEILING_X = 2.0                          # at most 2x today's target per puff
    STEP_X = 0.5                             # inside a bout (< 30 min), change by at most 0.5x target per puff

    def __init__(self, demand="nn", adapt=True, net=None):
        self.demand, self.adapt = demand, adapt
        self.net = net if net is not None or demand != "nn" else BaseNet.load()
        self.name = {"habit": "on the spot, habit only (no NN)"}.get(demand) or (
            "NN frozen base + adaptive head" if adapt else "NN fully frozen")

    def begin(self, baseline):
        self.state = DeviceState(baseline_with_level(baseline))
        for t, dur in baseline.get("puffs", []):                  # the device saw the baseline weeks at full strength
            self.state.observe(t, dur, 1.0)
        self.head = AdaptiveHead(self.net.head) if self.net is not None else None
        self.pending = []
        self.last_dose, self.last_t = None, None
        self.m_mean = 1.0
        self.day_err = []          # mean |predicted - real| per day, log scale, before that night's update

    def start_day(self, k, u, weekend):
        if self.head is not None and self.pending:
            n = len(self.state.day_times)
            self.day_err.append(float(np.mean([abs(y - math.log1p(n - i)) for i, h, y in self.pending])))
            if self.adapt:
                for i, h, y in self.pending[::5]:
                    self.head.update(h, math.log1p(n - i))       # the real answer: puffs from then to day end
        self.pending = []
        self.state.start_day(k, u, weekend)

    def predicted_rest(self, t, x):
        if self.demand == "nn":
            h = self.net.hidden(x)
            y = self.head.predict(h)
            self.pending.append((len(self.state.day_times), h, y))
            n = math.expm1(y) if math.isfinite(y) else float("nan")
        else:
            n = self.state.habit(t)[1]
        if not math.isfinite(n):
            n = self.state.habit(t)[1]                            # fallback: the person's own habit
        return min(max(n, 1.0), 3000.0)

    def dose(self, t):
        s = self.state
        s.advance(t)
        remaining = max(0.0, s.budget - s.used)
        if remaining <= 0 or s.u <= 0:
            return 0.0
        x = s.features(t)
        n_rest = self.predicted_rest(t, x)
        m = self.RELIEF_LO + (self.RELIEF_HI - self.RELIEF_LO) * s.relief(t)
        self.m_mean += 0.005 * (m - self.m_mean)             # running mean, so relief reshapes but does not shrink the day
        d = remaining / n_rest * m / self.m_mean
        # --- safety layer: fixed, the network cannot override it
        d = min(d, 1.0, self.CEILING_X * s.u, remaining)
        if self.last_t is not None and t - self.last_t < 0.5 and self.last_dose is not None:
            d = min(d, self.last_dose + self.STEP_X * s.u)
        d = max(0.0, d)
        self.last_dose, self.last_t = d, t
        return d

    def observe(self, t, dur, dose):
        self.state.observe(t, dur, dose)


def controllers():
    """The five fast layers compared in experiments/per_puff_comparison.py."""
    net = BaseNet.load()
    return {
        "flat": Flat,
        "codex_rules": CodexRules,
        "habit_only": lambda: OnTheSpot("habit"),
        "nn_frozen": lambda: OnTheSpot("nn", adapt=False, net=net),
        "nn_adaptive": lambda: OnTheSpot("nn", adapt=True, net=net),
    }
