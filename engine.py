"""Two tapering engines, same interface: engine.next_dose(dose, window, prev_window) -> mg/ml until the next decision.

FixedTaper   - the traditional approach: cut a fixed % at every decision, no matter what.
AdaptiveTaper- Wane: the JITAI loop.
                 decision point : every `period` days (7 = weekly refill, 1 = hardware that meters the dose)
                 tailoring vars : features computed from the last 7 days of puff data (rolling window)
                 decision rule  : features -> risk score -> action
                 intervention   : cut / half cut / hold  (never raise)
               plus per-person LEARNING: the tolerated rate is updated from evidence.

Both engines are given the WEEKLY rate and the period; they work out the cut per decision so that
the compounded weekly rate is the same whatever the period. That keeps comparisons fair.

Only the risk score is a candidate for ML. The dose arithmetic stays rules, and monotonic.
"""
import numpy as np

LOW_MG = 3.0        # below this, cuts are ABSOLUTE steps, not percentages (the way patch ladders end: 7 -> 0, not 7 -> 6.2 -> 5.4 ...)


def apply_cut(dose, cut):
    """The one place a dose changes. Never raises.

    Above LOW_MG: multiply by (1 - cut), the usual percentage taper.
    Below LOW_MG: subtract a fixed step of LOW_MG * cut, so every step near the end is the same size in
    absolute terms and the last one to zero is no bigger than the others. A percentage taper never reaches
    zero and then has to jump; this reaches zero in a few equal steps.
    """
    if dose <= 0 or cut <= 0:
        return max(0.0, dose)
    if dose > LOW_MG:
        return max(LOW_MG * (1 - cut), dose * (1 - cut)) if dose * (1 - cut) >= LOW_MG else LOW_MG - LOW_MG * cut
    step = LOW_MG * cut
    new = dose - step
    return 0.0 if new < step / 2 else new


def per_period_cut(weekly_cut, period):
    """Cut per decision so the compounded rate per week stays `weekly_cut`."""
    return 1 - (1 - weekly_cut) ** (period / 7)


def is_weekend(row):
    """profiles.Vaper counts day index t (0-based) with t % 7 in (5, 6) as the weekend; row['day'] is t + 1."""
    return (row["day"] - 1) % 7 in (5, 6)


class FixedTaper:
    name = "Fixed taper (traditional)"

    def __init__(self, weekly_cut=0.12, period=7):
        self.cut = per_period_cut(weekly_cut, period)

    def next_dose(self, dose, window, prev_window):
        return apply_cut(dose, self.cut)


# ---------- tailoring variables: computed from puff timestamps, no self-report ----------

FEATURE_NAMES = ["puff_trend", "dur_trend", "night_share", "ttfc_min", "weekend_ratio", "volatility", "dose_ratio"]


def feature_vector(f):
    return [f[k] for k in FEATURE_NAMES]


def features(window, prev_window, dose=None):
    puffs = np.array([r["puffs"] for r in window])
    prev = np.array([r["puffs"] for r in prev_window]) if prev_window else puffs
    f = {}
    f["dose_ratio"] = (dose if dose is not None else window[-1]["dose"]) / 20.0     # the engine knows the dose it set
    f["puff_trend"] = (puffs.mean() - prev.mean()) / max(prev.mean(), 1)          # +0.2 = 20 % more puffs
    dur = np.array([r.get("puff_dur", 3.4) for r in window]); pdur = np.array([r.get("puff_dur", 3.4) for r in prev_window]) if prev_window else dur
    f["dur_trend"] = (dur.mean() - pdur.mean()) / max(pdur.mean(), 0.1)            # longer puffs: the bigger compensation channel (LSBU)
    f["night_share"] = np.mean([r["night_puffs"] for r in window]) / max(puffs.mean(), 1)
    f["ttfc_min"] = np.mean([r["ttfc_min"] for r in window])                      # time to first puff
    wk = [r["puffs"] for r in window if is_weekend(r)]
    wd = [r["puffs"] for r in window if not is_weekend(r)]
    f["weekend_ratio"] = (np.mean(wk) / max(np.mean(wd), 1)) if (wk and wd) else 1.0
    f["volatility"] = puffs.std() / max(puffs.mean(), 1)
    return f


def derived_craving(f):
    """Craving inferred from behaviour alone (0-10). This is what replaces self-report."""
    c = 2.0
    c += 8.0 * max(0, f["puff_trend"])          # compensation
    c += 40.0 * max(0, f["night_share"] - 0.02) # withdrawal at night
    c += max(0, (10 - f["ttfc_min"]) / 4)       # reaching for it within minutes of waking
    return float(np.clip(c, 0, 10))


def risk_score(f):
    """Hand-set weights from the literature. In production these are the learned parameters."""
    r = 0.0
    r += 1.5 * max(0, f["puff_trend"])
    r += 6.0 * max(0, f["night_share"] - 0.02)
    r += 0.02 * max(0, 15 - f["ttfc_min"])
    r += 0.25 * max(0, f["weekend_ratio"] - 1.5)
    return float(np.clip(r, 0, 1))


class AdaptiveTaper:
    name = "Wane adaptive engine"

    # learning, expressed per WEEK so behaviour is the same whatever the period
    UP_PER_WEEK, DOWN_PER_WEEK = 1.05, 0.75     # tolerated: a little faster; struggled: a lot slower
    MAX_WEEKLY, MIN_WEEKLY = 0.15, 0.03         # bounds on the learned weekly rate

    def __init__(self, weekly_cut=0.12, period=7):
        self.period = period
        self.base_cut = per_period_cut(weekly_cut, period)
        self.personal_cut = self.base_cut       # LEARNED per person: starts at the population rate
        self.up = self.UP_PER_WEEK ** (period / 7)
        self.down = self.DOWN_PER_WEEK ** (period / 7)
        self.max_cut = per_period_cut(self.MAX_WEEKLY, period)
        self.min_cut = per_period_cut(self.MIN_WEEKLY, period)
        self.log = []                           # what it decided and why, for the demo

    def weekly_rate(self):
        """The learned rate expressed per week, for display."""
        return 1 - (1 - self.personal_cut) ** (7 / self.period)

    # hand-set scoring; FittedTaper overrides these two with learned weights
    HOLD, HALF = 0.6, 0.3
    def risk(self, f):
        return risk_score(f)
    def crave(self, f):
        return derived_craving(f)

    def next_dose(self, dose, window, prev_window):
        f = features(window, prev_window, dose=dose)
        risk = self.risk(f)
        crave = self.crave(f)
        day = window[-1]["day"]

        # --- learning: was the recent taper tolerated? move the personal rate ---
        #     thresholds scale with the engine's HOLD / HALF so the rule works for a 0-1 score and for a probability
        if risk < self.HALF * 0.6 and crave < 4:
            self.personal_cut = min(self.max_cut, self.personal_cut * self.up)
        elif risk > self.HOLD * 0.8 or crave > 6:
            self.personal_cut = max(self.min_cut, self.personal_cut * self.down)

        # --- decision rule: risk -> action (never raise) ---
        if risk > self.HOLD or crave > 7:
            cut, action = 0.0, "HOLD"
        elif risk > self.HALF or crave > 5:
            cut, action = self.personal_cut / 2, "HALF CUT"
        else:
            cut, action = self.personal_cut, "CUT"

        # --- no full week of history yet: never more than half a cut, but a HOLD is still a HOLD ---
        if len(prev_window) < 7 and action == "CUT":
            cut, action = self.personal_cut / 2, "FIRST CUT (half)"

        new = apply_cut(dose, cut)
        if new == 0.0 and dose > 0:
            action = action + " -> ZERO"
        self.log.append(dict(day=day, risk=risk, crave=crave, action=action, cut=cut, rate=self.weekly_rate(), **f))
        return new


class FittedTaper(AdaptiveTaper):
    """Same decision rule and learning as AdaptiveTaper, but risk and craving come from weights FITTED on a
    synthetic population (fit.py -> engine_weights.json). risk is now a probability of relapse within 7 days,
    so its thresholds are much lower than the hand-set 0.6 / 0.3."""
    name = "Wane fitted engine"

    def __init__(self, weekly_cut=0.12, period=7, weights=None):
        super().__init__(weekly_cut, period)
        if weights is None:
            import json, pathlib
            weights = json.load(open(pathlib.Path(__file__).with_name("engine_weights.json")))
        self.wts = weights
        self.HOLD, self.HALF = weights["hold"], weights["half"]

    def _lin(self, m, f):
        x = (np.array(feature_vector(f)) - np.array(m["mu"])) / np.array(m["sd"])
        return m["w"][0] + x @ np.array(m["w"][1:])

    def risk(self, f):
        return float(1 / (1 + np.exp(-self._lin(self.wts["risk"], f))))

    def crave(self, f):
        return float(np.clip(self._lin(self.wts["craving"], f), 0, 10))
