"""Rematch: Claude's hand plans against fast layer v4, on two synthetic people. SIMULATION ONLY.

P12-030: evening-only heavy vaper; plan frozen 26 Sept in experiments/human_vs_algorithm.py (reused unchanged).
P12-027: the hardest person of the test population (highest expected risk on the flat taper). Claude saw only
the device view of the three baseline weeks, then wrote and froze this plan with a prediction before any run:
  1. morning bout (first bout after >= 6 h): first 4 puffs x1.8, the rest x0.8
  2. any bout after a >= 45 min gap: puffs 1, 2, 3 at 1.5, 1.3, 1.1, then 0.7 (with a 2 h half-life their
     nicotine is mostly gone by then, so almost every bout is a relief bout)
  3. quick follow-up bouts (< 45 min after the last puff): 0.8 throughout
  4. after 22:00 (just before the night gap): x0.75
  5. pacing, because days vary from 23 to 114 puffs: > 10 points ahead of the usual pace x0.75; after 18:00 and
     > 15 points behind x1.15
  6. renormalised on the baseline log, same budget and same safety layer as the algorithm
  Prediction: clearly lower expected risk than flat; within +-0.03 of nn_v4.
Controllers: flat, a flat control at the hand plan's nicotine share, the hand plan, nn_v3, nn_v4. 20 seeds each.
"""
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from human_vs_algorithm import HandPlan as HandPlan030                      # noqa: E402
from puff_nn import V3, V4, BaseNet, Flat, OnTheSpot                        # noqa: E402
from puffsim import DAY_START_H, WARMUP_DAYS, SlowSchedule, simulate, test_population   # noqa: E402

SEEDS = range(20)


def weight_027(t, pos, gap_before_h, gap_since_bout_start_ok=True):
    if gap_before_h >= 6:
        w = 1.8 if pos < 4 else 0.8
    elif gap_before_h >= 0.75:
        w = (1.5, 1.3, 1.1)[pos] if pos < 3 else 0.7
    else:
        w = 0.8
    if 22 <= t % 24 or t % 24 < DAY_START_H:
        w *= 0.75
    return w


class HandPlan027(HandPlan030):
    name = "Claude's hand plan (P12-027)"

    def begin(self, b):
        self.b = b
        times = [p[0] for p in b["puffs"]]
        ws = {"wd": [], "we": []}
        last, pos, gap = None, 0, 99.0
        for t in times:
            if last is None or t - last > 5 / 60:
                pos, gap = 0, 99.0 if last is None else t - last
            else:
                pos += 1
            weekend = (math.floor((t - DAY_START_H) / 24) % 7) in (5, 6)
            ws["we" if weekend else "wd"].append(weight_027(t, pos, gap))
            last = t
        self.norm = {k: float(np.mean(v)) for k, v in ws.items()}
        self.last, self.pos, self.gap = None, 0, 99.0
        self.last_dose = None

    def dose(self, t):
        if self.last is None or t - self.last > 5 / 60:
            self.pos, self.gap = 0, 99.0 if self.last is None else t - self.last
        else:
            self.pos += 1
        remaining = max(0.0, self.budget - self.used)
        if remaining <= 0 or self.u <= 0:
            d = 0.0
        else:
            d = self.u * weight_027(t, self.pos, self.gap) / self.norm["we" if self.weekend else "wd"]
            pace = self.used / self.budget - self.expected_fraction(t)
            if pace > 0.10:
                d *= 0.75
            elif pace < -0.15 and (t % 24) >= 18:
                d *= 1.15
            d = min(d, 1.0, 2 * self.u, remaining)
            if self.last is not None and t - self.last < 0.5 and self.last_dose is not None:
                d = min(d, self.last_dose + 0.5 * self.u)
        self.last_dose = d
        return d


MATCH = {"P12-030": 0.88, "P12-027": None}      # P12-027's share is measured from its hand plan run below


class FlatAt(Flat):
    share = 1.0

    def start_day(self, k, u, weekend):
        self.u = u * self.share


def run(args):
    name, key, seed, share = args
    person = [p for p in test_population() if p.name == name][0]
    net = BaseNet.load()
    hand = HandPlan030 if name == "P12-030" else HandPlan027
    matched = type("FlatMatched", (FlatAt,), {"share": share or 1.0})
    make = {"flat": Flat, "flat_matched": matched, "hand_plan": hand,
            "nn_v3": lambda: OnTheSpot("nn", adapt=True, net=net, shape=V3),
            "nn_v4": lambda: OnTheSpot("nn", adapt=True, net=net, shape=V4)}[key]
    r = simulate(person, make, SlowSchedule(), seed=seed, stop_at_relapse=False)
    bw = r["baseline_w"]
    days = r["days"][WARMUP_DAYS:]
    excess = [d["mean_w"] - bw[d["day"] % 7] for d in days]
    return dict(name=name, key=key, seed=seed, expected_risk=r["expected_risk"], relapsed=int(r["relapsed"]),
                weeks1_8=float(np.mean(excess[:56])), mean_excess=float(np.mean(excess)),
                peak_week=float(max(np.mean(excess[i:i + 7]) for i in range(0, len(excess) - 6, 7))),
                budget=float(np.mean([d["dose_sum"] / (d["u"] * r["baseline_puffs"]) for d in days if d["u"] > 0])))


def ci(d, seed=0):
    rng = np.random.default_rng(seed)
    bt = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    return d.mean(), np.percentile(bt, 2.5), np.percentile(bt, 97.5)


if __name__ == "__main__":
    keys = ["flat", "flat_matched", "hand_plan", "nn_v3", "nn_v4"]
    with ProcessPoolExecutor() as pool:
        hp = list(pool.map(run, [("P12-027", "hand_plan", s, None) for s in SEEDS]))
        MATCH["P12-027"] = round(float(np.mean([r["budget"] for r in hp])), 3)
        rows = list(pool.map(run, [(n, k, s, MATCH[n]) for n in MATCH for k in keys for s in SEEDS]))
    for n in MATCH:
        print(f"\n{n}, {len(SEEDS)} seeds, SIMULATION ONLY (flat_matched runs at {MATCH[n]:.2f} of the level)")
        print(f"{'controller':13s} {'exp.risk':>8s} {'relapsed':>8s} {'excess wk1-8':>12s} {'mean excess':>11s} {'peak week':>9s} {'budget':>7s}")
        by = {k: sorted([r for r in rows if r["name"] == n and r["key"] == k], key=lambda r: r["seed"]) for k in keys}
        for k in keys:
            m = lambda f: np.mean([r[f] for r in by[k]])
            print(f"{k:13s} {m('expected_risk'):8.4f} {m('relapsed'):8.2f} {m('weeks1_8'):+12.4f} {m('mean_excess'):+11.4f} {m('peak_week'):9.4f} {m('budget'):7.2f}")
        for a, b in (("hand_plan", "nn_v4"), ("hand_plan", "nn_v3"), ("nn_v4", "nn_v3"), ("hand_plan", "flat_matched"), ("nn_v4", "flat")):
            for f in ("expected_risk", "weeks1_8"):
                d = np.array([x[f] - y[f] for x, y in zip(by[a], by[b])])
                m, lo, hi = ci(d)
                print(f"  {a:10s} - {b:12s} {f:13s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
