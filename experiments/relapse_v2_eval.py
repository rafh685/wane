"""Flat, v3 and v4 under relapse model v2, plus an abrupt quit, with a check against published outcomes. SIMULATION ONLY.

People: difficult test set (52, irregular habits, long-draw compensation) and population B (65). 3 seeds.
Every run: 28 unrecorded burn-in days, 3 baseline weeks, 24 weeks of taper (12 %/week reaches zero in week 23),
12 weeks at zero. Relapse risk from relapse.evaluate (model v2, 500 replays of the lapse process per run).
Published checks (research report): gradual vs abrupt quitting show no long-term difference (RR 1.01, 22 trials);
supported drug-free cessation gives about 13 to 20 % validated abstinence at 4 to 6 months.
Run: WANE_MAIN=/path/to/main python3 experiments/relapse_v2_eval.py
"""
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from calibrate_relapse import Abrupt                                         # noqa: E402
from hard_people import HARD, load                                          # noqa: E402
from puff_nn import V3, V4, BaseNet, Flat, OnTheSpot                        # noqa: E402
from puffsim import SlowSchedule, simulate, test_population                  # noqa: E402
from relapse import evaluate                                                 # noqa: E402

KEYS = ["abrupt quit", "flat 12%", "v3", "v4"]


def people():
    return [(p, "difficult") for p in load("test")] + [(p, "general") for p in test_population()]


def run(args):
    i, key, seed = args
    p, group = people()[i]
    net = BaseNet.load()
    make = {"abrupt quit": Flat, "flat 12%": Flat, "v3": lambda: OnTheSpot("nn", net=net, shape=V3),
            "v4": lambda: OnTheSpot("nn", net=net, shape=V4)}[key]
    slow = Abrupt() if key == "abrupt quit" else SlowSchedule()
    r = simulate(p, make, slow, taper_days=168, follow_days=84, seed=1000 * seed + i,
                 physiology=HARD if group == "difficult" else None, stop_at_relapse=False, burn_in_days=28)
    ev = evaluate(r, p, seed=1000 * seed + i, n_mc=500)
    return dict(person=p.name, group=group, controller=key, seed=seed, **ev,
                v1_expected_risk=r["expected_risk"])


if __name__ == "__main__":
    n = len(people())
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(run, [(i, k, s) for i in range(n) for k in KEYS for s in (0, 1, 2)], chunksize=4))
    out = ROOT / "experiments"
    with open(out / "relapse-v2-runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    names = sorted({(r["person"], r["group"]) for r in rows})
    per = lambda k, f: np.array([np.mean([r[f] for r in rows if r["controller"] == k and (r["person"], r["group"]) == nm]) for nm in names])
    grp = np.array([g for _, g in names])
    summary = {}
    print(f"{'':12s} {'group':10s} {'risk v2':>8s} {'surv 14d':>8s} {'surv 90d':>8s} {'surv 180d':>9s} {'risk v1':>8s}")
    for g in ("difficult", "general"):
        for k in KEYS:
            m = {f: float(np.nanmean(per(k, f)[grp == g])) for f in ("expected_risk", "survival_14", "survival_90", "survival_180", "v1_expected_risk")}
            summary[f"{k} | {g}"] = m
            print(f"{k:12s} {g:10s} {m['expected_risk']:8.3f} {m['survival_14']:8.3f} {m['survival_90']:8.3f} {m['survival_180']:9.3f} {m['v1_expected_risk']:8.3f}")
    rng = np.random.default_rng(0)
    pairs = {}
    for a, b in (("v4", "flat 12%"), ("v4", "v3"), ("flat 12%", "abrupt quit"), ("v4", "abrupt quit")):
        for g in ("difficult", "general"):
            d = (per(a, "expected_risk") - per(b, "expected_risk"))[grp == g]
            bt = [rng.choice(d, len(d)).mean() for _ in range(4000)]
            pairs[f"{a} - {b} | {g}"] = dict(diff=float(d.mean()), lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)),
                                             better=int((d < -0.005).sum()), worse=int((d > 0.005).sum()), n=len(d))
            p_ = pairs[f"{a} - {b} | {g}"]
            print(f"  {a:8s} - {b:12s} {g:10s} {p_['diff']:+.4f} [{p_['lo']:+.4f}, {p_['hi']:+.4f}] better {p_['better']} worse {p_['worse']} of {p_['n']}")
    for nm in ("Lucía", "Karim"):
        print(f"  {nm}: " + "  ".join(f"{k} {np.mean([r['expected_risk'] for r in rows if r['person'] == nm and r['controller'] == k]):.3f}" for k in KEYS))
    (out / "relapse-v2-summary.json").write_text(json.dumps(dict(label="SIMULATION ONLY", table=summary, pairs=pairs), indent=1, ensure_ascii=False) + "\n")
