"""Onboarding interview: questions, synthetic answers, and how each answer changes the algorithm. SIMULATION ONLY.

Rafael (27 Sept 2026): ask at setup what the device cannot see, and let every answer change the algorithm's
behaviour. The answers only set the STARTING plan; the device's own data corrects them afterwards.

Questions (wording as in the app). Questions 0, 1, 4, 6 and 7 route or set the starting level and are not
simulated here (the simulator has no cigarettes, pregnancy or goal, and normalises each person's starting level):
  0 age 18+                       -> under 18: the app stops
  1 current device and strength   -> starting level matched to today's intake
  2 time to first vape on waking  -> taper pace, stronger relief for the first bout after the night
  3 wakes at night to vape        -> taper pace, night puffs protected from the pre-gap cut
  4 smokes cigarettes too         -> daily smokers get a different plan (not simulated)
  5 past quit attempts            -> taper pace
  6 pregnant or breastfeeding     -> no taper, see a doctor (not simulated)
  7 goal: stop or cut down        -> end point (not simulated)
  8 optional: often anxious or low; drinks a lot on nights out -> taper pace; drinking nights protected

How synthetic people answer. Each person has a hidden "true difficulty" (low relapse threshold and slow
tolerance adaptation, the two traits that drive simulated relapse). Answers are noisy readings of it, with
strengths set from the research (docs: reports/Nicotine dependence parameters for Wane.md): time to first vape
is the strongest predictor, night waking and quit history moderate, mood weak. Category shares follow published
distributions where they exist (time to first vape: PATH adult daily vapers; night waking: 7 to 10 % of vapers);
the quit-history shares are assumptions. 15 % of answers are random, 15 % shift one step toward "less
dependent" (optimism), the optional question is skipped 20 % of the time. The drinking answer is read from the
person's actual alcohol cues. Careless answers follow the usual answer frequencies. The strength of these links is an ASSUMPTION: the interview can only help as much
as its answers carry real information, so the test also runs with weaker links and more wrong answers.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from population import population
from puff_nn import V4

QUESTIONS = [
    ("age", "Are you 18 or over?", ["Yes", "No"]),
    ("device", "What do you vape now, and what strength?", ["device type", "mg/ml"]),
    ("ttfv", "How soon after waking do you usually vape?", ["Within 5 minutes", "6 to 30 minutes", "31 to 60 minutes", "After 60 minutes"]),
    ("night", "Do you ever wake up at night to vape?", ["Never", "Rarely", "Some nights", "Most nights"]),
    ("smoke", "Do you smoke cigarettes too?", ["No", "Sometimes", "Every day"]),
    ("history", "Have you tried to stop or cut down before?", ["Never", "Yes, and it lasted", "Yes, back within a week", "Several times"]),
    ("pregnant", "Are you pregnant, trying to be, or breastfeeding?", ["No", "Yes"]),
    ("goal", "What is your goal?", ["Stop completely", "Cut down"]),
    ("mood", "Optional: do you often feel anxious or low?", ["No", "Yes", "Skip"]),
    ("alcohol", "Optional: do you sometimes drink a lot on nights out?", ["No", "Yes", "Skip"]),
]

# ---------------------------------------------------------------- hidden difficulty and synthetic answers

_REF = population(3000, seed=999)
_THR = np.array([p.relapse_threshold for p in _REF])
_TAU = np.array([np.clip(1 / p.craving_decay, 3, 14) for p in _REF])


def true_difficulty(p):
    """Standardised hidden difficulty: low relapse threshold, slow tolerance adaptation (the simulator's drivers)."""
    z_thr = -(p.relapse_threshold - _THR.mean()) / _THR.std()
    z_tau = (np.clip(1 / p.craving_decay, 3, 14) - _TAU.mean()) / _TAU.std()
    return float((z_thr + 0.6 * z_tau) / math.sqrt(1 + 0.36))


LINK = dict(ttfv=0.55, night=0.4, history=0.4, mood=0.2)          # correlation of each answer with difficulty
SHARES = dict(ttfv=[0.20, 0.37, 0.23, 0.20],                      # within 5 / 6-30 / 31-60 / after 60: PATH Wave 1 daily
                                                                  # vapers, weighted (experiments/path_vapers.py)
              night=[0.85, 0.06, 0.05, 0.04],                     # never / rarely / some / most (7-10 % wake)
              history=[0.35, 0.25, 0.25, 0.15],                   # never / lasted / back in a week / several (assumed)
              mood=[0.70, 0.30])                                  # no / yes


def _category(latent, shares, most_dependent_first):
    """Map a standard-normal latent to a category with the given shares; high latent = more dependent."""
    from statistics import NormalDist
    order = list(range(len(shares)))
    if most_dependent_first:
        order = order[::-1]            # shares listed from most dependent (index 0) to least
    cum, cuts = 0.0, []
    for i in order[:-1]:
        cum += shares[i]
        cuts.append(NormalDist().inv_cdf(cum))
    k = sum(latent > c for c in cuts)
    return order[k]


def synthetic_answers(p, rng, link_scale=1.0, random_rate=0.15, optimism_rate=0.15, skip_rate=0.20):
    d = true_difficulty(p)
    ans = {}
    for q, most_dep_first in (("ttfv", True), ("night", False), ("history", False), ("mood", False)):
        rho = min(0.95, LINK[q] * link_scale)
        latent = rho * d + math.sqrt(1 - rho ** 2) * rng.normal()
        a = _category(latent, SHARES[q], most_dep_first)
        n = len(SHARES[q])
        if rng.random() < random_rate:
            a = int(rng.choice(n, p=SHARES[q]))                   # a careless answer, at the usual frequencies
        elif rng.random() < optimism_rate:
            a = min(n - 1, a + 1) if most_dep_first else max(0, a - 1)
        ans[q] = a
    if rng.random() < skip_rate:
        ans["mood"] = None
    ans["alcohol"] = None if rng.random() < skip_rate else int(any(c.alcohol for c in p.cues))
    return ans


def answer_levels(ans):
    """Each answer as a 0 (easy) to 1 (hard) level. History: 'lasted' is easiest, 'never tried' is unknown."""
    return dict(
        ttfv=(3 - ans["ttfv"]) / 3,                               # within 5 min -> 1
        night=ans["night"] / 3,                                   # most nights -> 1
        history={1: 0.0, 0: 0.4, 2: 0.7, 3: 1.0}[ans["history"]],
        mood=0.5 if ans["mood"] is None else float(ans["mood"]),
    )


# ---------------------------------------------------------------- what each answer changes

@dataclass
class InterviewPolicy:
    """Maps answers to a starting plan. Every answer moves at least one behaviour:
      pace         weekly cut = base_cut x exp(-k x (score - 0.5)), clipped to [min_cut, max_cut];
                   score = weighted mean of the ttfv, night, history and mood levels
      ttfv         also: the first bout after the night gets relief_boost x (1 + morning_x x ttfv level)
      night        'some' or 'most nights': puffs inside the long gap are not cut (night_relief)
      alcohol      'yes': on Friday and Saturday the pre-gap cut is softened to pregap_social
    """
    base_cut: float = 0.12
    k: float = 0.0
    w: dict = field(default_factory=lambda: dict(ttfv=1.0, night=0.5, history=0.7, mood=0.3))
    min_cut: float = 0.06
    max_cut: float = 0.20        # fastest average weekly cut tolerated in the gradual arm of Hatsukami 2018 (research report)
    morning_x: float = 0.0
    night_relief: bool = False
    pregap_social: float = V4["pregap_cut"]
    oracle: bool = False                                          # use the hidden difficulty (upper bound only)

    def score(self, ans, p=None):
        if self.oracle:
            from statistics import NormalDist
            return NormalDist().cdf(true_difficulty(p))
        lv = answer_levels(ans)
        tot = sum(self.w.values())
        return sum(self.w[q] * lv[q] for q in self.w) / tot if tot > 0 else 0.5

    def plan(self, ans, p=None):
        s = self.score(ans, p)
        cut = float(np.clip(self.base_cut * math.exp(-self.k * (s - 0.5)), self.min_cut, self.max_cut))
        shape = dict(V4)
        lv = answer_levels(ans)
        if self.morning_x:
            shape["morning_x"] = self.morning_x * lv["ttfv"]
        if self.night_relief and ans["night"] >= 2:
            shape["night_relief"] = True
        if ans.get("alcohol") == 1 and self.pregap_social != V4["pregap_cut"]:
            shape["social_dows"] = (4, 5)
            shape["pregap_social"] = self.pregap_social
        return dict(cut=cut, shape=shape, score=s)


def chosen_policy(model="v2"):
    """The setting chosen on development people (experiments/interview_eval.py, rule fixed before running)."""
    import json
    import pathlib
    f = pathlib.Path(__file__).parent / "experiments" / f"interview-choice{'-v2' if model == 'v2' else ''}.json"
    return InterviewPolicy(**json.loads(f.read_text())["best"])


def weeks_to_zero(cut, low=0.15):
    """Weeks the slow layer takes from 1.0 to 0 at this weekly cut (same rule as puffsim.SlowSchedule)."""
    u, w = 1.0, 0
    while u > 0 and w < 520:
        w += 1
        if u > low:
            u = max(low, u * (1 - cut))
        else:
            step = low * cut
            u = 0.0 if u - step < step / 2 else u - step
    return w
