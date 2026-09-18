"""The Lucía problem: a hard compensator with slow craving decay who fails every automatic schedule.

On the manual taper she protects herself by skipping steps (11 of 20 runs still on the way at week 40, 6 relapsed).
A blind device removes that protection (13 of 20 relapse). Wane as fitted gets her to 7 free / 10 relapsed.
Variants tried on 18 Sept 2026, 30 runs, 40 weeks, 12 %/week, automatic:

  readiness gate      hold while estimated craving > own resting level + 1.5
                      -> Lucía relapse 43 -> 10 %, but nobody finishes (off 28 %): at low dose resting craving is
                         permanently higher (tonic term), so "back to baseline" never comes. The manual taper's stall, reproduced.
  first-cut rate      set the personal rate from the puffing response to the first cut
                      -> Lucía 27 %/27 % but off collapses to 52 %: the first cut is half-size, the response is noise.
  rising gate         hold only while estimated craving is climbing week on week
                      -> Lucía 43 -> 33 %, Sofia 20 -> 13 %, but Karim 27 -> 43 % and everyone slower (23 -> 28 weeks).
  first-cut brake     after two full cuts, if the two-week puffing response is large, halve the rate; never speed up from it
                      -> Lucía 43 -> 37 % but off 83 -> 74 % overall and B 94 -> 70 %: the brake fires on noise.
                         Her real two-week response is +3 to +16 % across seeds; Ana's is -18 to +11 %. The distributions overlap.

Conclusion so far: Lucía is identifiable only by her RESPONSE to cuts, and that signal is small against weekly noise
in the first weeks. Rules that catch her early also catch normal noise and slow everyone. The pilot's job is to
measure real response sizes; a longer horizon (60 weeks) lets the current engine finish about half of her runs.
Run: .venv/bin/python experiments/lucia.py
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from unseen_profile import run_sofia
from simulate import run_one, per_run
from profiles import PROFILES
from population import population
from engine import FixedTaper, FittedTaper
W = json.load(open(pathlib.Path(__file__).resolve().parents[1] / "engine_weights.json"))


class Brake(FittedTaper):
    name = "first-cut brake"
    def __init__(self, cut=0.12, period=7): super().__init__(cut, period, weights=W); self.done = False
    def next_dose(self, dose, window, prev_window):
        new = super().next_dose(dose, window, prev_window)
        if not self.done and len(self.log) == 4:                       # after the half cut and two full cuts
            resp = np.mean([self.log[-1]["intake_trend"], self.log[-2]["intake_trend"]])
            if resp > 0.20: self.personal_cut = max(self.min_cut, self.personal_cut * 0.35)
            elif resp > 0.12: self.personal_cut = max(self.min_cut, self.personal_cut * 0.5)
            self.done = True; self.log[-1]["action"] += f" (resp {resp:+.0%})"
        return new


def five(E, n=30, weeks=40):
    r = per_run(pd.concat([run_one(p, E, weeks, seed=s, weekly_cut=0.12, period=7)[0] for p in PROFILES for s in range(n)]))
    pp = r.groupby("profile"); rel = pp.relapsed.mean(); off = pp.reached_zero.mean()
    return f"all {r.relapsed.mean():3.0%}/{r.reached_zero.mean():3.0%}/{r.weeks_to_zero[r.reached_zero].median():2.0f}wk  Luc {rel['Lucía']:.0%}/{off['Lucía']:.0%}  Kar {rel['Karim']:.0%}/{off['Karim']:.0%}  Mar {rel['Marta']:.0%}/{off['Marta']:.0%}"
def popB(E):
    r = per_run(pd.concat([run_one(p, E, 40, seed=0, weekly_cut=0.12, period=7)[0] for p in population(80, seed=2)]))
    return f"B {r.relapsed.mean():.0%}/{r.reached_zero.mean():.0%}/{r.weeks_to_zero[r.reached_zero].median():.0f}wk"
def sofia(E, n=30):
    r = per_run(pd.concat([run_sofia(E, s, None) for s in range(n)])); return f"Sofia {r.relapsed.mean():.0%}/{r.reached_zero.mean():.0%}"


if __name__ == "__main__":
    for label, E in (("blind", FixedTaper), ("Wane now", lambda c, p: FittedTaper(c, p, weights=W)), ("first-cut brake", Brake)):
        print(f"{label:<18} {five(E)}   {popB(E)}   {sofia(E)}")
    print("Lucía's measured two-week response after the first full cuts, 10 seeds:",
          [round(float(np.mean([e["intake_trend"] for e in run_one(PROFILES[2], lambda c, p: FittedTaper(c, p, weights=W), 4, seed=s)[1][2:4]])), 2) for s in range(10)])
    print("Ana's:", [round(float(np.mean([e["intake_trend"] for e in run_one(PROFILES[4], lambda c, p: FittedTaper(c, p, weights=W), 4, seed=s)[1][2:4]])), 2) for s in range(10)])
