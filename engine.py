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
import json
import pathlib

import numpy as np

LOW_MG = 3.0        # below this, cuts are ABSOLUTE steps, not percentages (the way patch ladders end: 7 -> 0, not 7 -> 6.2 -> 5.4 ...)


DEFAULT_RESPONSE_CALIBRATION = {
    "reference_dose_reduction": 2 / 3,
    "total_puffing": {"p75_ratio": 1.68},
    "policy": {"high_pressure": 1.0, "immediate_pressure": 3.0, "calm_pressure": 0.35},
}


def load_response_calibration(path=None):
    """Load the reproducible, real-data response calibration.

    The calibration is deliberately small: it does not claim to predict relapse.
    It only says how unusual a measured puff-duration response is for the size of
    the preceding dose reduction. If the generated file is absent, the checked-in
    defaults preserve the same conservative guardrail.
    """
    path = pathlib.Path(path or pathlib.Path(__file__).with_name("engine_calibration.json"))
    try:
        with path.open() as handle:
            loaded = json.load(handle)
        if loaded["reference_dose_reduction"] <= 0 or loaded["total_puffing"]["p75_ratio"] <= 1:
            raise ValueError("invalid response calibration")
        return loaded
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return DEFAULT_RESPONSE_CALIBRATION


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

FEATURE_NAMES = ["puff_trend", "dur_trend", "night_share", "ttfc_min", "weekend_ratio", "volatility", "dose_ratio", "weekend_low",
                 "night_rel", "ttfc_rel", "intake_trend"]


def feature_vector(f):
    return [f[k] for k in FEATURE_NAMES]


def features(window, prev_window, dose=None, baseline=None):
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
    f["weekend_low"] = max(0.0, f["weekend_ratio"] - 1.0) * (1 - f["dose_ratio"])   # heavy weekends AND low dose: the Karim pattern
    # --- self-referenced: this person against THEIR OWN first week at full strength ---
    #     a night share of 0.025 means nothing across people; a rise from one's own 0.025 to 0.04 does
    b = baseline or {"night_share": f["night_share"], "ttfc_min": f["ttfc_min"], "intake": 1.0}
    f["night_rel"] = f["night_share"] / max(b["night_share"], 0.005) - 1
    f["ttfc_rel"] = f["ttfc_min"] / max(b["ttfc_min"], 1.0) - 1
    f["intake_trend"] = (1 + f["puff_trend"]) * (1 + f["dur_trend"]) - 1          # puffs x duration: total puffing vs last week
    return f


def make_baseline(window):
    """The person's own reference, taken from their first week at full strength."""
    f = features(window, [], dose=window[-1]["dose"])
    return {"night_share": f["night_share"], "ttfc_min": f["ttfc_min"], "intake": 1.0}


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

    def __init__(self, weekly_cut=0.12, period=7, response_calibration=None):
        self.period = period
        self.base_cut = per_period_cut(weekly_cut, period)
        self.personal_cut = self.base_cut       # LEARNED per person: starts at the population rate
        self.up = self.UP_PER_WEEK ** (period / 7)
        self.down = self.DOWN_PER_WEEK ** (period / 7)
        self.max_cut = per_period_cut(self.MAX_WEEKLY, period)
        self.min_cut = per_period_cut(self.MIN_WEEKLY, period)
        self.log = []                           # what it decided and why, for the demo
        self.baseline = None                    # this person's own first-week reference (set at the first decision)
        self.calm_streak = 0                    # consecutive decisions with a demonstrably small response to the last cut
        self.high_response_streak = 0           # consecutive high compensation responses after a cut
        self.response_calibration = response_calibration or load_response_calibration()

    def weekly_rate(self):
        """The learned rate expressed per week, for display."""
        return 1 - (1 - self.personal_cut) ** (7 / self.period)

    # hand-set scoring; FittedTaper overrides these two with learned weights
    HOLD, HALF = 0.6, 0.3
    def risk(self, f):
        return risk_score(f)
    def crave(self, f):
        return derived_craving(f)

    def response_pressure(self, f, dose, prev_window):
        """How strong was this person's compensation for their *last* dose cut?

        The observed rise in total puffing is scaled to the 18 -> 6 mg reduction
        measured in the real paired study. A pressure of 1.0 is its upper-quartile
        response, not a clinical threshold. No prior dose history means there is no
        response evidence, so the method returns zero rather than inventing it.
        """
        if not prev_window:
            return 0.0
        previous_dose = float(prev_window[-1].get("dose", dose))
        reduction = max(0.0, 1 - dose / max(previous_dose, 1e-6))
        if reduction < 0.01:
            return 0.0
        observed_increase = max(0.0, f["intake_trend"])
        reference_drop = self.response_calibration["reference_dose_reduction"]
        reference_increase = self.response_calibration["total_puffing"]["p75_ratio"] - 1
        equivalent_increase = observed_increase * reference_drop / reduction
        return float(equivalent_increase / max(reference_increase, 1e-6))

    def update_personal_rate(self, f, risk, crave, dose, prev_window, risk_brake, calm_risk):
        """Update the per-person taper speed from measured response to the last cut.

        High compensation is new evidence of strain even when the synthetic risk
        score is quiet. To avoid treating weekly noise as strain, it must appear in
        two consecutive windows unless the response is extreme. Conversely, the
        engine speeds up only after two calm, low-compensation windows. This keeps
        the rule adaptive without treating a 20-person study as a relapse model.
        """
        pressure = self.response_pressure(f, dose, prev_window)
        policy = self.response_calibration.get("policy", DEFAULT_RESPONSE_CALIBRATION["policy"])
        high_pressure = policy.get("high_pressure", 1.0)
        immediate_pressure = policy.get("immediate_pressure", 3.0)
        calm_pressure = policy.get("calm_pressure", 0.35)
        self.high_response_streak = self.high_response_streak + 1 if pressure >= high_pressure else 0
        high_response = pressure >= immediate_pressure or self.high_response_streak >= 2
        tolerated = (
            f["intake_trend"] < 0.05
            and f["night_rel"] < 0.25
            and f["ttfc_rel"] > -0.25
            and pressure < calm_pressure
            and risk < calm_risk
            and crave < 4
        )
        self.calm_streak = self.calm_streak + 1 if tolerated else 0
        if risk > risk_brake or crave > 6 or high_response:
            self.personal_cut = max(self.min_cut, self.personal_cut * self.down)
            self.calm_streak = 0
        elif self.calm_streak >= 2:
            self.personal_cut = min(self.max_cut, self.personal_cut * self.up)
        return pressure

    def next_dose(self, dose, window, prev_window):
        if self.baseline is None:
            self.baseline = make_baseline(window)     # first decision: this week at full strength IS the reference
        f = features(window, prev_window, dose=dose, baseline=self.baseline)
        risk = self.risk(f)
        crave = self.crave(f)
        day = window[-1]["day"]

        # --- learning: was the recent taper tolerated? move the personal rate ---
        #     SLOW DOWN on any sign of trouble. SPEED UP only on positive evidence: the response to the last cut
        #     was demonstrably small, against this person's own baseline, for two decisions in a row.
        #     Absence of alarm is not evidence of coping (an unfamiliar person can look calm and be sinking).
        pressure = self.update_personal_rate(
            f, risk, crave, dose, prev_window,
            risk_brake=self.HOLD * 0.8,
            calm_risk=self.HALF * 0.6,
        )

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
        self.log.append(dict(day=day, risk=risk, crave=crave, response_pressure=pressure,
                             response_streak=self.high_response_streak,
                             action=action, cut=cut, rate=self.weekly_rate(), **f))
        return new


class FittedTaper(AdaptiveTaper):
    """Same decision rule and learning as AdaptiveTaper, but risk and craving come from weights FITTED on a
    synthetic population (fit.py -> engine_weights.json). risk is now a probability of relapse within 7 days,
    so its thresholds are much lower than the hand-set 0.6 / 0.3."""
    name = "Wane fitted engine"

    def __init__(self, weekly_cut=0.12, period=7, weights=None, response_calibration=None):
        super().__init__(weekly_cut, period, response_calibration=response_calibration)
        if weights is None:
            with pathlib.Path(__file__).with_name("engine_weights.json").open() as handle:
                weights = json.load(handle)
        self.wts = weights
        self.BUDGET = weights.get("budget", 0.02)          # max predicted relapse risk per week we are willing to take
        self.CRAVE_HALF, self.CRAVE_HOLD = weights.get("crave_half", 4.5), weights.get("crave_hold", 5.5)
        self.HOLD, self.HALF = weights.get("hold", self.BUDGET), weights.get("half", self.BUDGET / 2)

    def _lin(self, m, f):
        x = (np.array(feature_vector(f)) - np.array(m["mu"])) / np.array(m["sd"])
        return m["w"][0] + x @ np.array(m["w"][1:])

    def risk(self, f):
        return float(1 / (1 + np.exp(-self._lin(self.wts["risk"], f))))

    def crave(self, f):
        return float(np.clip(self._lin(self.wts["craving"], f), 0, 10))

    def risk_at(self, f, new_dose):
        """Predicted relapse risk for the coming week IF the dose becomes new_dose. Dose is an input to the risk
        model, so the same features scored at a lower dose give a higher risk. This is what turns the model
        from an alarm into a planner."""
        g = dict(f)
        g["dose_ratio"] = new_dose / 20.0
        g["weekend_low"] = max(0.0, g["weekend_ratio"] - 1.0) * (1 - g["dose_ratio"])
        return self.risk(g)

    def next_dose(self, dose, window, prev_window):
        """Risk-budget rule. Candidates: the full personal cut, half of it, or hold. Take the largest cut whose
        predicted risk for the coming week stays under BUDGET; if none does, hold. A person whose weekly risk is
        1 to 2 % never trips an alarm, but 2 % a week for twenty weeks is a third of them lost: the budget sees that,
        a threshold does not. Learning of the personal rate is unchanged (see AdaptiveTaper)."""
        if self.baseline is None:
            self.baseline = make_baseline(window)
        f = features(window, prev_window, dose=dose, baseline=self.baseline)
        risk, crave = self.risk(f), self.crave(f)
        day = window[-1]["day"]
        pressure = self.update_personal_rate(
            f, risk, crave, dose, prev_window,
            risk_brake=self.BUDGET,
            calm_risk=self.BUDGET * 0.5,
        )

        full = self.personal_cut if len(prev_window) >= 7 else self.personal_cut / 2
        options = [(full, "CUT" if len(prev_window) >= 7 else "FIRST CUT (half)"), (full / 2, "HALF CUT"), (0.0, "HOLD")]
        chosen = None
        for cut, action in options:
            new = apply_cut(dose, cut)
            if self.risk_at(f, new) <= self.BUDGET or cut == 0.0:
                chosen = (cut, action, new); break
        cut, action, new = chosen
        # --- craving budget: the risk model sees one week ahead; estimated craving sees the slow build-up.
        #     Above CRAVE_HOLD hold; above CRAVE_HALF never more than half a cut. (Sofia, the unseen night-shift
        #     nurse, was lost by the risk budget alone: her weekly risk never spiked, her craving crept.)
        if crave > self.CRAVE_HOLD and cut > 0:
            cut, action, new = 0.0, "HOLD (craving)", dose
        elif crave > self.CRAVE_HALF and cut > full / 2:
            cut, action, new = full / 2, "HALF CUT (craving)", apply_cut(dose, full / 2)
        if new == 0.0 and dose > 0:
            action += " -> ZERO"
        self.log.append(dict(day=day, risk=risk, crave=crave, response_pressure=pressure,
                             response_streak=self.high_response_streak,
                             action=action, cut=cut, rate=self.weekly_rate(), risk_next=self.risk_at(f, new), **f))
        return new
