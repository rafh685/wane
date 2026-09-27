"""Protect the end of the taper: different shapes for Wane's weekly plan near zero. SIMULATION ONLY.

Rafael (27 Sept 2026, night): next step on the roadmap. About half of the people who reach zero on Wane relapse in
the following 12 weeks. Why, in the simulator: at zero the fast tolerance S drops within days, but the slow
adaptation S2 (tau about 190 days) is still high, so the slow craving (S2 - S) peaks right after zero; and the
"nicotine still in the plan" protection ends that day.

Candidates (all use the v4 per-puff layer; only the weekly plan changes):
  same end date (zero in week 23, like today)
    A  current        12 % a week to 15 % of the start, then 8 equal steps to zero
    B  front-loaded   16 % a week to 15 % (week 11), then 12 equal smaller steps to zero
    C  front-loaded+  20 % a week to 15 % (week 9), then 14 equal smaller steps to zero
  longer plans
    E  +4 weeks       as A, but 12 equal steps at the bottom (zero in week 27)
    F  +8 weeks       as A, but 16 equal steps at the bottom (zero in week 31)
    D  relative       12 % a week all the way down to 2 %, then zero (week 31)
Outcome, fixed before running: NICOTINE-FREE 12 WEEKS AFTER ZERO (no relapse from the first taper day until
12 weeks after that person's zero day), so every plan gets the same time after zero.
Selection rule, fixed before running: on development people (difficult dev 50 + population seed 31, 40; 2 seeds)
adopt the best same-end-date design if it beats A by at least 0.5 per 100 on the general group and does not cost
difficult people more than 0.5 per 100. Longer plans are reported, not adopted: the simulator has no dropout, so a
longer plan cannot be penalised for people giving up on it. Then one run on the test sets (difficult test 52 +
population B 65, 3 seeds).
Run: WANE_MAIN=/path/to/main python3 experiments/end_of_taper.py dev|test [comma-separated plan names]
Test run on 27 Sept: A, C (the choice), E and F (longer plans, for information).
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
from hard_people import HARD, load                                          # noqa: E402
from population import population                                           # noqa: E402
from puff_nn import V4, BaseNet, OnTheSpot                                  # noqa: E402
from puffsim import SlowSchedule, simulate, test_population                  # noqa: E402
from relapse import evaluate                                                 # noqa: E402

OUT = ROOT / "experiments"
LOW = 0.15


def plan(first_cut, bottom_steps, low=LOW):
    """Weekly levels: relative cuts to `low`, then equal absolute steps to zero."""
    u, weeks = 1.0, [1.0]
    while u > low:
        u = max(low, u * (1 - first_cut))
        weeks.append(u)
    for k in range(1, bottom_steps + 1):
        weeks.append(low * (1 - k / bottom_steps) if k < bottom_steps else 0.0)
    return weeks


def relative_plan(cut=0.12, floor=0.02):
    u, weeks = 1.0, [1.0]
    while u > floor:
        u *= 1 - cut
        weeks.append(u if u > floor else 0.0)
    if weeks[-1] > 0:
        weeks.append(0.0)
    return weeks


def current_plan():
    s = SlowSchedule()
    s.begin({"puffs_per_day": 100})
    lv = [s.level(d, 100) for d in range(0, 60 * 7, 7)]
    return lv[:lv.index(0.0) + 1]


PLANS = {
    "A current": current_plan(),
    "B front-loaded": plan(0.16, 12),
    "C front-loaded+": plan(0.20, 14),
    "E +4 weeks": current_plan()[:16] + [current_plan()[15] * (1 - k / 12) for k in range(1, 13)],
    "F +8 weeks": current_plan()[:16] + [current_plan()[15] * (1 - k / 16) for k in range(1, 17)],
    "D relative to 2 %": relative_plan(),
}
SAME_END = ("A current", "B front-loaded", "C front-loaded+")


class WeeklyPlan(SlowSchedule):
    def __init__(self, weeks):
        super().__init__()
        self.weeks = weeks

    def level(self, taper_day, last_day_puffs):
        w = taper_day // 7
        return self.weeks[w] if w < len(self.weeks) else 0.0


def people(which):
    if which == "dev":
        return [(p, "difficult") for p in load("dev")] + [(p, "general") for p in population(40, seed=31)]
    return [(p, "difficult") for p in load("test")] + [(p, "general") for p in test_population()]


def run(args):
    which, i, name, seed = args
    p, group = people(which)[i]
    weeks = PLANS[name]
    zero_day = 7 * weeks.index(0.0)
    net = BaseNet.load()
    r = simulate(p, lambda: OnTheSpot("nn", net=net, shape=V4), WeeklyPlan(weeks), taper_days=zero_day, follow_days=84,
                 seed=1000 * seed + i, physiology=HARD if group == "difficult" else None, stop_at_relapse=False,
                 burn_in_days=28)
    ev = evaluate(r, p, seed=1000 * seed + i, n_mc=500)
    assert r["zero_day"] == zero_day, (r["zero_day"], zero_day)
    return dict(person=p.name, group=group, plan=name, seed=seed, zero_week=zero_day // 7,
                free=1 - ev["expected_risk"], relapse_during_taper=ev["relapse_before_zero"],
                relapse_after_zero=ev["expected_risk"] - ev["relapse_before_zero"],
                mean_u=float(np.mean([d["u"] for d in r["days"][21:21 + zero_day]])))


if __name__ == "__main__":
    which = sys.argv[1]
    if len(sys.argv) > 2:                        # a subset of plans, always including A (the reference)
        keep = ["A current"] + [k for k in sys.argv[2].split(",") if k != "A current"]
        PLANS = {k: PLANS[k] for k in keep}
    seeds = (0, 1) if which == "dev" else (0, 1, 2)
    n = len(people(which))
    for k, w in PLANS.items():
        print(f"{k:20s} zero in week {w.index(0.0):2d}: " + " ".join(f"{x:.3f}" for x in w))
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(run, [(which, i, k, s) for i in range(n) for k in PLANS for s in seeds], chunksize=2))
    with open(OUT / f"end-of-taper-{which}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    names = sorted({(r["person"], r["group"]) for r in rows})
    grp = np.array([g for _, g in names])
    per = lambda k, f: np.array([np.mean([r[f] for r in rows if r["plan"] == k and (r["person"], r["group"]) == nm]) for nm in names])
    rng = np.random.default_rng(0)
    summary = {}
    print(f"\n{'plan':20s} {'group':10s} {'zero wk':>7s} {'free':>6s} {'in taper':>8s} {'after 0':>8s} {'mean dose':>9s}  vs A (free, per 100)")
    for g in ("general", "difficult"):
        for k in PLANS:
            d = (per(k, "free") - per("A current", "free"))[grp == g]
            bt = [rng.choice(d, len(d)).mean() for _ in range(4000)]
            m = dict(zero_week=PLANS[k].index(0.0), free=float(per(k, "free")[grp == g].mean()),
                     relapse_during_taper=float(per(k, "relapse_during_taper")[grp == g].mean()),
                     relapse_after_zero=float(per(k, "relapse_after_zero")[grp == g].mean()),
                     mean_u=float(per(k, "mean_u")[grp == g].mean()),
                     vs_A=dict(diff=float(d.mean()), lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)),
                               better=int((d > 0.005).sum()), worse=int((d < -0.005).sum()), n=len(d)))
            summary[f"{k} | {g}"] = m
            print(f"{k:20s} {g:10s} {m['zero_week']:7d} {m['free']:6.3f} {m['relapse_during_taper']:8.3f} "
                  f"{m['relapse_after_zero']:8.3f} {m['mean_u']:9.3f}  {100 * m['vs_A']['diff']:+.2f} "
                  f"[{100 * m['vs_A']['lo']:+.2f}, {100 * m['vs_A']['hi']:+.2f}] better {m['vs_A']['better']} worse {m['vs_A']['worse']}")
    out = dict(label="SIMULATION ONLY", which=which, plans={k: [round(x, 4) for x in w] for k, w in PLANS.items()}, table=summary)
    if which == "dev":
        ok = [k for k in SAME_END[1:] if summary[f"{k} | general"]["vs_A"]["diff"] >= 0.005
              and summary[f"{k} | difficult"]["vs_A"]["diff"] >= -0.005]
        best = max(ok, key=lambda k: summary[f"{k} | general"]["vs_A"]["diff"]) if ok else "A current"
        out["choice"] = best
        print(f"\nchosen by the fixed rule: {best}")
    (OUT / f"end-of-taper-{which}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
