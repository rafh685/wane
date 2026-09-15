"""Two tapering engines, same interface: engine.next_dose(history) -> mg/ml for the coming week.

FixedTaper   - the traditional approach: cut a fixed % every week, no matter what.
AdaptiveTaper- Wane: the JITAI loop.
                 decision point : every week (and a pre-emptive check before risky days)
                 tailoring vars : features computed from the last 7 days of puff data
                 decision rule  : features -> risk score -> action
                 intervention   : cut / half cut / hold  (never raise)
               plus per-person LEARNING: the tolerated weekly cut is updated from evidence.

Only the risk score is a candidate for ML. The dose arithmetic stays rules, and monotonic.
"""
import numpy as np

FLOOR_MG = 0.5


class FixedTaper:
    name = "Fixed taper (traditional)"

    def __init__(self, weekly_cut=0.12):
        self.cut = weekly_cut

    def next_dose(self, dose, week_rows, prev_week_rows):
        return max(FLOOR_MG, dose * (1 - self.cut))


# ---------- tailoring variables: what the device can measure, no self-report ----------

def features(week_rows, prev_week_rows):
    puffs = np.array([r["puffs"] for r in week_rows])
    prev = np.array([r["puffs"] for r in prev_week_rows]) if prev_week_rows else puffs
    f = {}
    f["puff_trend"] = (puffs.mean() - prev.mean()) / max(prev.mean(), 1)          # +0.2 = 20 % more puffs
    f["night_share"] = np.mean([r["night_puffs"] for r in week_rows]) / max(puffs.mean(), 1)
    f["ttfc_min"] = np.mean([r["ttfc_min"] for r in week_rows])                    # time to first puff
    f["weekend_ratio"] = (puffs[-2:].mean() / max(puffs[:-2].mean(), 1)) if len(puffs) >= 7 else 1.0
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

    def __init__(self, weekly_cut=0.12):
        self.base_cut = weekly_cut
        self.personal_cut = weekly_cut   # LEARNED per person: starts at the population rate
        self.log = []                    # what it decided and why, for the demo

    def next_dose(self, dose, week_rows, prev_week_rows):
        f = features(week_rows, prev_week_rows)
        risk = risk_score(f)
        crave = derived_craving(f)

        # --- learning: was last week's cut tolerated? move the personal rate ---
        if risk < 0.2 and crave < 4:
            self.personal_cut = min(0.15, self.personal_cut * 1.05)   # tolerated well: a little faster
        elif risk > 0.5 or crave > 6:
            self.personal_cut = max(0.03, self.personal_cut * 0.75)   # struggled: slow the personal rate

        # --- first decision has no history to judge by: start conservative ---
        if not prev_week_rows:
            cut = self.personal_cut / 2
            new = max(FLOOR_MG, dose * (1 - cut))
            self.log.append(dict(risk=risk, crave=crave, action="FIRST CUT (half)", cut=cut, rate=self.personal_cut, **f))
            return new

        # --- decision rule: risk -> action (never raise) ---
        if risk > 0.6 or crave > 7:
            cut, action = 0.0, "HOLD"
        elif risk > 0.3 or crave > 5:
            cut, action = self.personal_cut / 2, "HALF CUT"
        else:
            cut, action = self.personal_cut, "CUT"

        new = max(FLOOR_MG, dose * (1 - cut))
        self.log.append(dict(risk=risk, crave=crave, action=action, cut=cut, rate=self.personal_cut, **f))
        return new
