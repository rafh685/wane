"""Wane against the traditional taper people do by hand today. SIMULATION ONLY.

Rafael (27 Sept 2026): show Wane's efficacy against a real traditional taper, not only against the idealised
"weaker bottle" control (Wane's own smooth weekly plan with the same dose in every puff).
Traditional taper = puffsim.ManualTaper: shop bottles 18 -> 12 -> 6 -> 3 -> 0 mg/ml, a step every 4 to 7 weeks,
back to the stronger bottle for 2 weeks when puffing jumps, zero by week 24. Its settings were fixed before running.
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


def run(args):
    i, seed = args
    p, group = people()[i]
    slow = ManualTaper(seed=1000 * seed + i + 555)
    r = simulate(p, Flat, slow, taper_days=168, follow_days=84, seed=1000 * seed + i,
                 physiology=HARD if group == "difficult" else None, stop_at_relapse=False, burn_in_days=28)
    ev = evaluate(r, p, seed=1000 * seed + i, n_mc=500)
    return dict(person=p.name, group=group, controller="traditional", seed=seed, **ev,
                v1_expected_risk=r["expected_risk"], zero_day=r["zero_day"], step_backs=slow.step_backs)


if __name__ == "__main__":
    n = len(people())
    with ProcessPoolExecutor() as ex:
        new = list(ex.map(run, [(i, s) for i in range(n) for s in (0, 1, 2)], chunksize=2))
    with open(OUT / "traditional-taper-runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(new[0])); w.writeheader(); w.writerows(new)
    with open(OUT / "relapse-v2-runs.csv") as fh:
        old = [dict(r, **{k: float(r[k]) for k in r if k not in ("person", "group", "controller", "seed")})
               for r in csv.DictReader(fh)]
    rows = [r for r in old if r["controller"] in KEYS] + new
    names = sorted({(r["person"], r["group"]) for r in rows})
    grp = np.array([g for _, g in names])

    def per(k, f):
        return np.array([np.mean([float(r[f]) for r in rows if r["controller"] == k and (r["person"], r["group"]) == nm])
                         for nm in names])

    summary, funnel = {}, {}
    print(f"{'':28s} {'group':10s} {'relapse':>8s} {'in taper':>8s} {'after 0':>8s} {'still off':>9s}")
    for g in ("general", "difficult"):
        for k in KEYS:
            risk, before = per(k, "expected_risk")[grp == g], per(k, "relapse_before_zero")[grp == g]
            m = dict(relapse=float(risk.mean()), relapse_during_taper=float(before.mean()),
                     reach_zero=float(1 - before.mean()), relapse_after_zero=float((risk - before).mean()),
                     still_off=float(1 - risk.mean()), survival_90=float(per(k, "survival_90")[grp == g].mean()),
                     survival_180=float(per(k, "survival_180")[grp == g].mean()))
            summary[f"{k} | {g}"] = m
            print(f"{LABEL[k]:28s} {g:10s} {m['relapse']:8.3f} {m['relapse_during_taper']:8.3f} "
                  f"{m['relapse_after_zero']:8.3f} {m['still_off']:9.3f}")
    zd = [r["zero_day"] for r in new]
    extra = dict(zero_week_mean=float(np.mean(zd) / 7), zero_week_range=[min(zd) / 7, max(zd) / 7],
                 step_backs_per_person=float(np.mean([r["step_backs"] for r in new])),
                 share_with_a_step_back=float(np.mean([r["step_backs"] > 0 for r in new])))
    print("traditional taper:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in extra.items()})
    rng = np.random.default_rng(0)
    pairs = {}
    for a, b in (("v4", "traditional"), ("flat 12%", "traditional"), ("v4", "flat 12%"), ("traditional", "abrupt quit"),
                 ("v4", "abrupt quit")):
        for g in ("general", "difficult"):
            d = (per(a, "expected_risk") - per(b, "expected_risk"))[grp == g]
            bt = [rng.choice(d, len(d)).mean() for _ in range(4000)]
            pairs[f"{a} - {b} | {g}"] = p_ = dict(diff=float(d.mean()), lo=float(np.percentile(bt, 2.5)),
                                                  hi=float(np.percentile(bt, 97.5)), better=int((d < -0.005).sum()),
                                                  worse=int((d > 0.005).sum()), n=len(d))
            print(f"  {a:12s} - {b:12s} {g:10s} {p_['diff']:+.4f} [{p_['lo']:+.4f}, {p_['hi']:+.4f}] "
                  f"better {p_['better']} worse {p_['worse']} of {p_['n']}")
    for nm in ("Lucía", "Karim"):
        print(f"  {nm}: " + "  ".join(f"{LABEL[k]} {np.mean([float(r['expected_risk']) for r in rows if r['person'] == nm and r['controller'] == k]):.3f}"
                                      for k in KEYS))
    (OUT / "traditional-taper-summary.json").write_text(json.dumps(
        dict(label="SIMULATION ONLY", names=LABEL, table=summary, traditional=extra, pairs=pairs), indent=1,
        ensure_ascii=False) + "\n")
