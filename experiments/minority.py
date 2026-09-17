"""Can the engine catch the minority a blind automatic schedule loses?

Levers tried on the fitted engine (17 Sept 2026), five profiles x 30 runs + unseen population B, 40 weeks, 12 %/week:
  - slow harder after trouble (DOWN 0.75 -> 0.5), lower floor: no effect
  - hold at half the fitted risk threshold: Lucía 57 -> 47 %, small cost in speed
  - weekend guard (never more than half a cut when weekend_low > x): Karim 40 -> 27 % at x = 0.15,
    but population B slows from 19 to 33 weeks to zero, because many ordinary people have mild weekend patterns.
Conclusion: the minority is partly catchable, at a price in speed for everyone with a weekend pattern.
Choosing that price is a product decision; the pilot is where per-person learning replaces the population rule.
Run: .venv/bin/python experiments/minority.py
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from simulate import run_one, per_run
from profiles import PROFILES
from population import population
from engine import FixedTaper, FittedTaper, apply_cut

W = json.load(open(pathlib.Path(__file__).resolve().parents[1] / "engine_weights.json"))


def make(name, hold_mult=1.0, weekend_guard=None, down=0.75):
    w = dict(W); w["hold"], w["half"] = W["hold"] * hold_mult, W["half"] * hold_mult
    class E(FittedTaper):
        DOWN_PER_WEEK = down
        def __init__(self, cut=0.12, period=7): super().__init__(cut, period, weights=w)
        def next_dose(self, dose, window, prev_window):
            new = super().next_dose(dose, window, prev_window)
            if weekend_guard is not None and self.log:
                f = self.log[-1]
                if f["weekend_low"] > weekend_guard and new < dose:
                    half = apply_cut(dose, self.personal_cut / 2)
                    if half > new:
                        new = half; self.log[-1]["action"] += " (weekend guard)"
            return new
    E.name = name
    return E


def evalp(E, profiles, seeds):
    return per_run(pd.concat([run_one(p, E, 40, seed=s, weekly_cut=0.12, period=7)[0] for p in profiles for s in seeds]))


if __name__ == "__main__":
    B = population(80, seed=2)
    variants = [("Fixed pod (blind 12 %)", FixedTaper), ("Wane fitted", make("a")),
                ("hold at half the risk", make("b", 0.5)), ("weekend guard 0.15", make("d", 1.0, 0.15)),
                ("hold at half + guard 0.15", make("e", 0.5, 0.15))]
    print(f"{'variant':<30}{'five: rel/off/wks':>20}   Luc   Kar   Mar | pop B rel/off/wks")
    for label, E in variants:
        r5 = evalp(E, PROFILES, range(30)); pp = r5.groupby("profile").relapsed.mean(); rb = evalp(E, B, (0,))
        print(f"{label:<30}{r5.relapsed.mean():>6.0%}/{r5.reached_zero.mean():.0%}/{r5.weeks_to_zero[r5.reached_zero].median():>3.0f}   "
              f"{pp['Lucía']:>4.0%}  {pp['Karim']:>4.0%}  {pp['Marta']:>4.0%} | {rb.relapsed.mean():.0%}/{rb.reached_zero.mean():.0%}/{rb.weeks_to_zero[rb.reached_zero].median():.0f}")
