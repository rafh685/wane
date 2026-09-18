"""A person the engine never saw, run through the three products.

Sofia, 38, night-shift nurse. Sleeps 09:00 to 16:00, first puff on the drive in at 19:00, break puffs at 23:00
and 03:00, a dense block after the shift at 07:00 to 09:00. Every third week she rotates to day shifts and her
whole pattern flips. Moderate compensation, slow craving decay, a low relapse threshold: a fragile person whose
"night" is everyone else's afternoon. Nothing about her was in the fitting population's routine templates.

Run: .venv/bin/python experiments/unseen_profile.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from profiles import Profile, Cue, Vaper, _slots, START_MG, WEEKEND
from simulate import run_one, per_run
from engine import FixedTaper, FittedTaper

night_wd = _slots((19, 20, 4), (23, 24, 3), (3, 4, 3), (7, 9, 4))          # night shift
day_wd = _slots((6, 8, 3), (12, 13, 3), (17, 22, 2))                       # rotation week: an ordinary day person
SOFIA = Profile("Sofia", "38, night-shift nurse. Sleeps by day, first puff at 19:00, rotates to day shifts every third week.",
                140, 0.55, 6.0, 0.10, 6.0, 0.15,
                wake={"wd": 16, "we": 11}, bed={"wd": 33, "we": 26},        # bed 09:00 next day, weekends more normal
                weekday_routine=night_wd, weekend_routine=_slots((11, 24, 2), (24, 26, 3)),
                weekend_factor=1.3, cues=[Cue((4,), 7, 10, 1.5, alcohol=True)], tonic_gain=2.4)


class RotatingVaper(Vaper):
    """Every third week Sofia works days: routine, wake and bed flip for that week."""
    def day(self, dose_mg):
        week = self.t // 7
        p = self.p
        if week % 3 == 2:
            p.weekday_routine, p.wake["wd"], p.bed["wd"] = day_wd, 6.5, 22.5
        else:
            p.weekday_routine, p.wake["wd"], p.bed["wd"] = night_wd, 16, 33
        return super().day(dose_mg)


def run_sofia(E, seed, adherence, weeks=40):
    """run_one with the rotating vaper swapped in."""
    import simulate
    orig = simulate.Vaper
    simulate.Vaper = RotatingVaper
    try:
        return run_one(SOFIA, E, weeks, seed=seed, weekly_cut=0.12, period=7, adherence=adherence)[0]
    finally:
        simulate.Vaper = orig


if __name__ == "__main__":
    N = 40
    cases = [("Manual taper, user switches (adherence 0.85)", FixedTaper, 0.85),
             ("Automatic taper, same schedule for everyone", FixedTaper, None),
             ("Wane, automatic, adaptive", FittedTaper, None)]
    rows = []
    for label, E, adh in cases:
        r = per_run(pd.concat([run_sofia(E, s, adh) for s in range(N)]))
        rows.append(dict(product=label, relapse=r.relapsed.mean(), stuck=(r.stalled | ~(r.relapsed | r.reached_zero)).mean(),
                         off=r.reached_zero.mean(), weeks=r.weeks_to_zero[r.reached_zero].median()))
    d = pd.DataFrame(rows)
    print(f"Sofia, night-shift nurse, never seen by the engine. {N} runs, 40 weeks, 12 %/week\n")
    print(d.to_string(index=False, formatters={c: "{:.0%}".format for c in ("relapse", "stuck", "off")} | {"weeks": "{:.0f}".format}))

    # what does the engine see on her? one run, the decision log
    import simulate
    simulate.Vaper = RotatingVaper
    df, log = run_one(SOFIA, FittedTaper, 40, seed=3, weekly_cut=0.12, period=7)
    lg = pd.DataFrame(log)
    print("\nEngine decisions on one run (seed 3):")
    print(lg[["day", "action", "cut", "rate", "risk", "crave", "night_share", "ttfc_min", "weekend_ratio"]].head(16)
            .to_string(index=False, formatters={"cut": "{:.0%}".format, "rate": "{:.0%}".format, "risk": "{:.2f}".format,
                                                "crave": "{:.1f}".format, "night_share": "{:.3f}".format, "ttfc_min": "{:.0f}".format, "weekend_ratio": "{:.2f}".format}))
    print("\nWhat her 'night share' and 'time to first puff' look like at 20 mg, calm (first two weeks):")
    calm = df[df.week <= 2]
    print(f"  night share {calm.night_puffs.sum() / max(calm.puffs.sum(), 1):.3f}   time to first puff {calm.ttfc_min.mean():.0f} min   puffs/day {calm.puffs.mean():.0f}")
    from profiles import PROFILES
    for p in PROFILES[:1]:
        d0, _ = run_one(p, FittedTaper, 2, seed=3, weekly_cut=0.12, period=7)
        print(f"  for comparison, {p.name}: night share {d0.night_puffs.sum() / max(d0.puffs.sum(), 1):.3f}   time to first puff {d0.ttfc_min.mean():.0f} min")
