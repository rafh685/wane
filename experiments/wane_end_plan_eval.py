"""Wane with the front-loaded end plan (puffsim.END_PLAN) in the same 9-month comparison as the other methods. SIMULATION ONLY.

Same people, seeds, windows and outcome as experiments/traditional_taper_eval.py (24 weeks of plan + 12 weeks after,
nicotine-free at 9 months). The plan was chosen on development people in experiments/end_of_taper.py and tested
there once; this run only puts it on the same footing as the headline numbers.
Run: WANE_MAIN=/path/to/main python3 experiments/wane_end_plan_eval.py
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
from puff_nn import V4, BaseNet, OnTheSpot                                  # noqa: E402
from puffsim import END_PLAN, SlowSchedule, simulate                          # noqa: E402
from relapse import evaluate                                                 # noqa: E402
from traditional_taper_eval import HARD, OUT, outcomes, paired, people, per_person, wane_rows  # noqa: E402


def run(args):
    i, seed = args
    p, group = people()[i]
    net = BaseNet.load()
    r = simulate(p, lambda: OnTheSpot("nn", net=net, shape=V4), SlowSchedule(**END_PLAN), taper_days=168, follow_days=84,
                 seed=1000 * seed + i, physiology=HARD if group == "difficult" else None, stop_at_relapse=False,
                 burn_in_days=28)
    ev = evaluate(r, p, seed=1000 * seed + i, n_mc=500)
    return dict(person=p.name, group=group, controller="v4 end plan", seed=seed, **ev, final_u=r["days"][-1]["u"],
                zero_day=r["zero_day"])


if __name__ == "__main__":
    n = len(people())
    with ProcessPoolExecutor() as ex:
        new = list(ex.map(run, [(i, s) for i in range(n) for s in (0, 1, 2)], chunksize=2))
    with open(OUT / "wane-end-plan-runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(new[0])); w.writeheader(); w.writerows(new)
    with open(OUT / "traditional-taper-runs.csv") as fh:
        trad = list(csv.DictReader(fh))
    rows = [r for r in wane_rows() if r["controller"] == "v4"] + trad + new
    names = sorted({(r["person"], r["group"]) for r in rows})
    grp = np.array([g for _, g in names])
    fields = ("relapse_during_taper", "stuck", "relapse_after_zero", "free", "relapse")
    table = {}
    for g in ("general", "difficult"):
        m = {f: float(per_person(rows, names, "v4 end plan", f)[grp == g].mean()) for f in fields}
        table[g] = m
        print(f"Wane with end plan  {g:10s} " + "  ".join(f"{f} {100 * v:5.1f}" for f, v in m.items()))
    rng = np.random.default_rng(0)
    pairs = {}
    for b in ("v4", "traditional"):
        for g in ("general", "difficult"):
            pairs[f"v4 end plan - {b} | {g}"] = p_ = paired(per_person(rows, names, "v4 end plan", "free"),
                                                            per_person(rows, names, b, "free"), grp, g, rng)
            print(f"  nicotine-free, end plan - {b:12s} {g:10s} {100 * p_['diff']:+.2f} [{100 * p_['lo']:+.2f}, "
                  f"{100 * p_['hi']:+.2f}] better {p_['better']} worse {p_['worse']} of {p_['n']}")
    (OUT / "wane-end-plan-summary.json").write_text(json.dumps(dict(label="SIMULATION ONLY", end_plan=END_PLAN,
                                                                    table=table, pairs=pairs), indent=1) + "\n")
