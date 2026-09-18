"""Everything, once, with the engine as committed. Run: .venv/bin/python experiments/full_report.py"""
import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd, profiles
from unseen_profile import run_sofia
from simulate import run_one, per_run
from profiles import PROFILES
from population import population
from engine import FixedTaper, FittedTaper

WEEKS, N5, NB, NS = 40, 30, 80, 30
t0 = time.time()

def stats(r):
    stuck = (r.stalled | ~(r.relapsed | r.reached_zero)).mean()
    return dict(relapse=r.relapsed.mean(), stuck=stuck, off=r.reached_zero.mean(), weeks=r.weeks_to_zero[r.reached_zero].median())

def five(E, adh):
    r = per_run(pd.concat([run_one(p, E, WEEKS, seed=s, weekly_cut=0.12, period=7, adherence=adh)[0] for p in PROFILES for s in range(N5)]))
    per = {p: stats(r[r.profile == p]) for p in [x.name for x in PROFILES]}
    return stats(r), per

def popB(E, adh):
    return stats(per_run(pd.concat([run_one(p, E, WEEKS, seed=0, weekly_cut=0.12, period=7, adherence=adh)[0] for p in population(NB, seed=2)])))

def sofia(E, adh):
    return stats(per_run(pd.concat([run_sofia(E, s, adh) for s in range(NS)])))

def fmt(s):
    return f"{s['relapse']:4.0%} / {s['stuck']:4.0%} / {s['off']:4.0%} / {s['weeks']:3.0f}wk" if not np.isnan(s['weeks']) else f"{s['relapse']:4.0%} / {s['stuck']:4.0%} / {s['off']:4.0%} /  -"

for mode in (False, True):
    profiles.BOUT_MODE = mode
    W = json.load(open(pathlib.Path(__file__).resolve().parents[1] / ("engine_weights_bouts.json" if mode else "engine_weights.json")))
    wane = lambda c, p: FittedTaper(c, p, weights=W)
    print(f"\n{'='*100}\n{'BOUT-LEVEL GENERATOR (Ilian)' if mode else 'HOURLY GENERATOR (demo default)'}   40 weeks, 12 %/week   columns: relapse / stuck or still tapering / nicotine-free / median weeks to zero\n{'='*100}")
    rows = [("Manual taper, user switches (adherence 0.85)", FixedTaper, 0.85),
            ("Manual, less diligent (0.70)", FixedTaper, 0.70),
            ("Wane software, manual steps (0.85)", wane, 0.85),
            ("Automatic taper, same schedule for all", FixedTaper, None),
            ("Wane, automatic", wane, None)]
    print(f"{'product':<46}{'five profiles (' + str(N5) + ' runs each)':>32}{'unseen pop (' + str(NB) + ')':>28}{'Sofia, unseen':>28}")
    pers = {}
    for label, E, adh in rows:
        f, per = five(E, adh); pers[label] = per
        print(f"{label:<46}{fmt(f):>32}{fmt(popB(E, adh)):>28}{fmt(sofia(E, adh)):>28}")
    print(f"\nper profile, relapse / nicotine-free:")
    print(f"{'':<46}" + "".join(f"{p.name:>11}" for p in PROFILES))
    for label in pers:
        print(f"{label:<46}" + "".join(f"{pers[label][p.name]['relapse']:>5.0%}/{pers[label][p.name]['off']:<5.0%}" for p in PROFILES))
print(f"\n{time.time()-t0:.0f}s")
