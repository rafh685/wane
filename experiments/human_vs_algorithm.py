"""One synthetic person, a hand-written dosing plan against the algorithm. SIMULATION ONLY.

Protocol: Claude saw only the device view of P12-030's three baseline weeks (puff times and durations), wrote
the plan below, and froze it with a prediction before any run. Every controller gets the same daily budget,
the same safety layer, the same person, the same 20 seeds.

The plan (frozen 26 Sept 2026, before running):
  1. relief bout: first bout after >= 3 h without puffing (19:00 weekdays, about 14:30 weekends):
     its first 5 puffs at weight 2.0, the rest of that bout 1.0
  2. every other bout is front-loaded: puff 1, 2, 3 at 1.5, 1.3, 1.1; puff 4 onwards 0.7
  3. late night is cut first: 22:00 to 24:00 x0.85, after midnight x0.7 (nicotine before a 17 h gap does nothing
     for tomorrow)
  4. weights renormalised on this person's baseline log, so the day's total matches the budget
  5. guard: if the budget is burning more than 10 points ahead of the usual pace for this hour, x0.8
  6. same safety layer as OnTheSpot: <= 1.0, <= 2x target, <= what is left today, +0.5x target max inside a bout
Prediction: beats flat on withdrawal; within +-3 points of nn_adaptive on success.
"""
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from puff_nn import Flat, OnTheSpot, BaseNet                     # noqa: E402
from puffsim import DAY_START_H, WARMUP_DAYS, SlowSchedule, simulate, test_population   # noqa: E402

PERSON, SEEDS = "P12-030", range(20)


def plan_weight(t, pos, relief_bout):
    if relief_bout:
        w = 2.0 if pos < 5 else 1.0
    else:
        w = (1.5, 1.3, 1.1)[pos] if pos < 3 else 0.7
    h = t % 24
    if 22 <= h < 24:
        w *= 0.85
    elif h < DAY_START_H:
        w *= 0.7
    return w


def bouts(times):
    """(position in bout, relief bout?) for each puff time, bouts split at 5 min, relief after >= 3 h."""
    out, pos, relief, last = [], 0, True, None
    for t in times:
        if last is None or t - last > 5 / 60:
            pos, relief = 0, last is None or t - last >= 3
        else:
            pos += 1
        out.append((pos, relief))
        last = t
    return out


class HandPlan:
    name = "Claude's hand plan"

    def begin(self, b):
        self.b = b
        times = [p[0] for p in b["puffs"]]
        ws = {"wd": [], "we": []}
        for t, (pos, rel) in zip(times, bouts(times)):
            weekend = (math.floor((t - DAY_START_H) / 24) % 7) in (5, 6)
            ws["we" if weekend else "wd"].append(plan_weight(t, pos, rel))
        self.norm = {k: float(np.mean(v)) for k, v in ws.items()}
        self.last, self.pos, self.relief = None, 0, True
        self.last_dose = None

    def start_day(self, k, u, weekend):
        self.u, self.weekend = u, weekend
        self.budget = u * (self.b["puffs_we"] if weekend else self.b["puffs_wd"])
        self.used = 0.0

    def expected_fraction(self, t):
        hist = self.b["hourly"]["we" if self.weekend else "wd"]
        order = [(DAY_START_H + i) % 24 for i in range(24)]
        h = (t - DAY_START_H) % 24
        done = sum(hist[int(order[i])] for i in range(int(h))) + hist[int(order[int(h)])] * (h - int(h))
        return done / max(sum(hist), 1e-9)

    def dose(self, t):
        if self.last is None or t - self.last > 5 / 60:
            self.pos, self.relief = 0, self.last is None or t - self.last >= 3
        else:
            self.pos += 1
        remaining = max(0.0, self.budget - self.used)
        if remaining <= 0 or self.u <= 0:
            d = 0.0
        else:
            d = self.u * plan_weight(t, self.pos, self.relief) / self.norm["we" if self.weekend else "wd"]
            if self.used / self.budget > self.expected_fraction(t) + 0.10:
                d *= 0.8
            d = min(d, 1.0, 2 * self.u, remaining)
            if self.last is not None and t - self.last < 0.5 and self.last_dose is not None:
                d = min(d, self.last_dose + 0.5 * self.u)
        self.last_dose = d
        return d

    def observe(self, t, dur, dose):
        self.last = t
        self.used += dose


def run(args):
    key, seed = args
    person = [p for p in test_population() if p.name == PERSON][0]
    net = BaseNet.load()
    make = {"flat": Flat, "hand_plan": HandPlan, "habit_only": lambda: OnTheSpot("habit"),
            "nn_adaptive": lambda: OnTheSpot("nn", adapt=True, net=net)}[key]
    rec = []
    r = simulate(person, make, SlowSchedule(), seed=seed, record=rec)
    bw = r["baseline_w"]
    days = r["days"][WARMUP_DAYS:]
    excess = [d["mean_w"] - bw[d["day"] % 7] for d in days]
    week4 = [x for x in rec if WARMUP_DAYS + 21 <= x[0] < WARMUP_DAYS + 28]
    return dict(key=key, seed=seed, success=int(not r["relapsed"] and r["zero_day"] is not None),
                relapsed=int(r["relapsed"]), relapse_day=r["relapse_day"],
                mean_excess=float(np.mean(excess)), peak_week=float(max(np.mean(excess[i:i + 7]) for i in range(0, max(1, len(excess) - 6), 7))),
                week1_excess=float(np.mean(excess[:7])), weeks1_8_excess=float(np.mean(excess[:56])),
                dose_share=float(np.mean([d["dose_sum"] / (d["u"] * r["baseline_puffs"]) for d in days if d["u"] > 0])),
                puffs_ratio=float(np.mean([d["puffs"] for d in days[:56]]) / r["baseline_puffs"]),
                week4=[(x[1] % 24, x[3], (x[0] % 7) in (5, 6)) for x in week4] if seed == 0 else None,
                w4_days=[(d["day"] % 7, d["u"], d["mean_w"] - bw[d["day"] % 7]) for d in r["days"][WARMUP_DAYS + 21:WARMUP_DAYS + 28]] if seed == 0 else None)


if __name__ == "__main__":
    keys = ["flat", "hand_plan", "habit_only", "nn_adaptive"]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run, [(k, s) for k in keys for s in SEEDS]))
    by = {k: [r for r in rows if r["key"] == k] for k in keys}
    print(f"{PERSON}, {len(SEEDS)} seeds, SIMULATION ONLY")
    print(f"{'controller':12s} {'success':>8s} {'relapse':>8s} {'excess wk1':>10s} {'excess wk1-8':>12s} {'mean excess':>11s} {'peak week':>9s} {'budget used':>11s} {'puffs x':>8s}")
    for k in keys:
        m = lambda f: np.mean([r[f] for r in by[k]])
        print(f"{k:12s} {m('success'):8.2f} {m('relapsed'):8.2f} {m('week1_excess'):10.4f} {m('weeks1_8_excess'):12.4f} {m('mean_excess'):11.4f} {m('peak_week'):9.4f} {m('dose_share'):11.2f} {m('puffs_ratio'):8.2f}")
    rng = np.random.default_rng(0)
    for f in ("weeks1_8_excess", "mean_excess"):
        for other in ("flat", "nn_adaptive", "habit_only"):
            d = np.array([a[f] - b[f] for a, b in zip(by["hand_plan"], by[other])])
            bt = [rng.choice(d, len(d)).mean() for _ in range(4000)]
            print(f"  hand_plan - {other:11s} {f:16s} {d.mean():+.4f} [{np.percentile(bt, 2.5):+.4f}, {np.percentile(bt, 97.5):+.4f}]")
    # the week-4 dose pattern, seed 0: mean dose relative to target by clock hour, weekday vs weekend
    print("\nTaper week 4, seed 0: mean dose / today's target, by clock hour (weekdays)")
    hours = [19, 20, 21, 22, 23, 0, 1]
    print("hour        " + " ".join(f"{h:>5d}" for h in hours))
    for k in keys:
        rw = by[k][0]
        u = {dow: uu for dow, uu, _ in rw["w4_days"]}
        tgt = rw["w4_days"][0][1]
        vals = []
        for h in hours:
            xs = [d for (hh, d, we) in rw["week4"] if int(hh) == h and not we]
            vals.append(np.mean(xs) / tgt if xs else float("nan"))
        print(f"{k:12s}" + " ".join(f"{v:5.2f}" for v in vals))
    print("\nTaper week 4, seed 0: excess withdrawal by day (Mon..Sun)")
    for k in keys:
        print(f"{k:12s}" + " ".join(f"{e:+.3f}" for _, _, e in by[k][0]["w4_days"]))
