"""Wane against the traditional taper people do by hand today. SIMULATION ONLY.

Rafael (27 Sept 2026): show Wane's efficacy against a real traditional taper, not only against the idealised
"weaker bottle" control (Wane's own smooth weekly plan with the same dose in every puff).
Traditional taper = puffsim.ManualTaper: shop bottles 18 -> 12 -> 6 -> 3 -> 0 mg/ml, a step every 4 to 7 weeks,
back to the stronger bottle for 2 weeks when puffing jumps. Since 27 Sept (night) people can get stuck on a bottle
(Rafael: "people that are stuck on the ladder"): no deadline, a chance of never buying the next bottle, and giving up
after failing the same step twice. Settings fixed before running.
Main outcome since then: NICOTINE-FREE at 9 months = not relapsed and on 0 mg/ml on the last day. Someone still on
a bottle who has not relapsed is "stuck": neither relapsed nor free.
Same people, seeds, windows and relapse model as experiments/relapse_v2_eval.py (whose runs are reused):
difficult test set (52) and population B (65), 3 seeds, 28 burn-in days, 3 baseline weeks, 24 weeks of taper,
12 weeks after. Relapse from relapse.evaluate (model v2, 500 replays per run).
Run: WANE_MAIN=/path/to/main python3 experiments/traditional_taper_eval.py
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
from puff_nn import Flat                                                     # noqa: E402
from puffsim import ManualTaper, simulate                                    # noqa: E402
from relapse import evaluate                                                 # noqa: E402
from relapse_v2_eval import HARD, people                                     # noqa: E402

OUT = ROOT / "experiments"
KEYS = ["abrupt quit", "traditional", "flat 12%", "v4"]
LABEL = {"abrupt quit": "Stop all at once", "traditional": "Traditional taper by hand",
         "flat 12%": "Wane's weekly plan only", "v4": "Wane"}
FIRST_VERSION = dict(stall=None, give_up=False, end_day=168)     # 27 Sept evening: nobody stuck, zero by week 24


def run_traditional(i, seed, schedule=None, **kw):
    p, group = people()[i]
    slow = (schedule or ManualTaper)(seed=1000 * seed + i + 555, **kw)
    r = simulate(p, Flat, slow, taper_days=168, follow_days=84, seed=1000 * seed + i,
                 physiology=HARD if group == "difficult" else None, stop_at_relapse=False, burn_in_days=28)
    ev = evaluate(r, p, seed=1000 * seed + i, n_mc=500)
    return dict(person=p.name, group=group, controller="traditional", seed=seed, **ev,
                v1_expected_risk=r["expected_risk"], zero_day=r["zero_day"], step_backs=slow.step_backs,
                stuck=int(slow.stuck), final_u=r["days"][-1]["u"])


def run(args):
    return run_traditional(*args)


def outcomes(r):
    """Where one run ends at 9 months, as shares that add up to 1 (averaged over the relapse replays)."""
    risk, before = float(r["expected_risk"]), float(r["relapse_before_zero"])
    if float(r["final_u"]) > 0:              # still on a nicotine bottle on the last day
        return dict(relapse_during_taper=risk, stuck=1 - risk, relapse_after_zero=0.0, free=0.0, relapse=risk)
    return dict(relapse_during_taper=before, stuck=0.0, relapse_after_zero=risk - before, free=1 - risk, relapse=risk)


def wane_rows():
    with open(OUT / "relapse-v2-runs.csv") as fh:
        return [dict(r, final_u=0.0) for r in csv.DictReader(fh)]     # Wane's plan and stopping at once end at 0 mg/ml


def per_person(rows, names, key, field):
    return np.array([np.mean([outcomes(r)[field] for r in rows if r["controller"] == key and (r["person"], r["group"]) == nm])
                     for nm in names])


def paired(a, b, grp, g, rng):
    d = (a - b)[grp == g]
    bt = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    return dict(diff=float(d.mean()), lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)),
                better=int((d > 0.005).sum()), worse=int((d < -0.005).sum()), n=len(d))


if __name__ == "__main__":
    n = len(people())
    with ProcessPoolExecutor() as ex:
        new = list(ex.map(run, [(i, s) for i in range(n) for s in (0, 1, 2)], chunksize=2))
    with open(OUT / "traditional-taper-runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(new[0])); w.writeheader(); w.writerows(new)
    rows = [r for r in wane_rows() if r["controller"] in KEYS] + new
    names = sorted({(r["person"], r["group"]) for r in rows})
    grp = np.array([g for _, g in names])
    fields = ("relapse_during_taper", "stuck", "relapse_after_zero", "free", "relapse")

    summary = {}
    print(f"{'':28s} {'group':10s} {'relapse':>8s} {'in taper':>8s} {'stuck':>7s} {'after 0':>8s} {'free':>7s}")
    for g in ("general", "difficult"):
        for k in KEYS:
            m = {f: float(per_person(rows, names, k, f)[grp == g].mean()) for f in fields}
            m["reach_zero"] = m["relapse_after_zero"] + m["free"]
            summary[f"{k} | {g}"] = m
            print(f"{LABEL[k]:28s} {g:10s} {m['relapse']:8.3f} {m['relapse_during_taper']:8.3f} {m['stuck']:7.3f} "
                  f"{m['relapse_after_zero']:8.3f} {m['free']:7.3f}")
    extra = dict(zero_week_mean_of_those_reaching_zero=float(np.mean([r["zero_day"] for r in new if r["final_u"] == 0]) / 7),
                 share_ending_on_a_bottle=float(np.mean([r["final_u"] > 0 for r in new])),
                 share_stuck_flag=float(np.mean([r["stuck"] for r in new])),
                 step_backs_per_person=float(np.mean([r["step_backs"] for r in new])),
                 share_with_a_step_back=float(np.mean([r["step_backs"] > 0 for r in new])))
    print("traditional taper:", {k: round(v, 3) for k, v in extra.items()})
    rng = np.random.default_rng(0)
    pairs = {}
    print("nicotine-free at 9 months, per 100 people (positive = first is better):")
    for a, b in (("v4", "traditional"), ("flat 12%", "traditional"), ("v4", "flat 12%"), ("traditional", "abrupt quit"),
                 ("v4", "abrupt quit")):
        for g in ("general", "difficult"):
            pairs[f"{a} - {b} | {g}"] = p_ = paired(per_person(rows, names, a, "free"), per_person(rows, names, b, "free"),
                                                     grp, g, rng)
            print(f"  {a:12s} - {b:12s} {g:10s} {100 * p_['diff']:+.2f} [{100 * p_['lo']:+.2f}, {100 * p_['hi']:+.2f}] "
                  f"better {p_['better']} worse {p_['worse']} of {p_['n']}")
    for nm in ("Lucía", "Karim"):
        print(f"  {nm} free: " + "  ".join(
            f"{LABEL[k]} {np.mean([outcomes(r)['free'] for r in rows if r['person'] == nm and r['controller'] == k]):.3f}"
            for k in KEYS))
    (OUT / "traditional-taper-summary.json").write_text(json.dumps(
        dict(label="SIMULATION ONLY", outcome="nicotine-free at 9 months = not relapsed and on 0 mg/ml on the last day",
             names=LABEL, table=summary, traditional=extra, pairs=pairs), indent=1, ensure_ascii=False) + "\n")
