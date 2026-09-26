"""Choose the fast layer's shaping on a DEVELOPMENT population, never on the test population. SIMULATION ONLY.

Development: 40 synthetic people, seed 31 (not the network's training people, seed 21; not the test people, seed 12).
Stable scenario, 12 %/week slow layer, the adaptive NN as demand model, one seed per person.
Objective: mean expected relapse risk over the taper and follow-up, 1 - exp(-sum of daily hazards) along the
full path (relapse draws are recorded but do not stop the run), which is far less noisy than counting relapses.
Grid: pk relief on/off x front-loading 0 / 0.5 / 1.0 x relief-bout boost 1 / 1.5 / 2 x pre-gap cut 1 / 0.7 / 0.5.
Writes experiments/tune-fast-layer-v4.csv and prints the ranking and each component's main effect.
Run: python3 experiments/tune_fast_layer.py, then python3 experiments/tune_fast_layer.py --refine (same people,
finer grid around the edge where the first pass's best values sat).
"""
import csv
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from population import population                         # noqa: E402
from puff_nn import V3, BaseNet, OnTheSpot               # noqa: E402
from puffsim import WARMUP_DAYS, SlowSchedule, simulate   # noqa: E402

DEV = population(40, seed=31)
GRID = [dict(pk=pk, front=f, relief_boost=rb, pregap_cut=pc)
        for pk, f, rb, pc in itertools.product((True, False), (0.0, 0.5, 1.0), (1.0, 1.5, 2.0), (1.0, 0.7, 0.5))]
REFINE = "--refine" in sys.argv          # second pass: the first pass's best values sat on the grid's edge
if REFINE:
    GRID = [dict(pk=False, front=f, relief_boost=rb, pregap_cut=pc)
            for f, rb, pc in itertools.product((0.0, 0.5), (2.0, 2.5, 3.0), (0.5, 0.35, 0.2))] + [dict(V3)]


def run(args):
    gi, pi = args
    net = BaseNet.load()
    shape = GRID[gi]
    r = simulate(DEV[pi], lambda: OnTheSpot("nn", adapt=True, net=net, shape=shape), SlowSchedule(),
                 seed=7000 + pi, stop_at_relapse=False)
    bw = r["baseline_w"]
    days = r["days"][WARMUP_DAYS:]
    return dict(config=gi, person=pi, **{k: v for k, v in shape.items()}, expected_risk=r["expected_risk"],
                relapsed=int(r["relapsed"]),
                mean_excess=float(np.mean([d["mean_w"] - bw[d["day"] % 7] for d in days])),
                budget_used=float(np.mean([d["dose_sum"] / (d["u"] * r["baseline_puffs"]) for d in days if d["u"] > 0])))


if __name__ == "__main__":
    jobs = [(g, p) for g in range(len(GRID)) for p in range(len(DEV))]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(run, jobs, chunksize=6))
    with open(ROOT / "experiments" / ("tune-fast-layer-v4-refine.csv" if REFINE else "tune-fast-layer-v4.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    agg = []
    for g, shape in enumerate(GRID):
        rs = [r for r in rows if r["config"] == g]
        agg.append((np.mean([r["expected_risk"] for r in rs]), np.mean([r["mean_excess"] for r in rs]),
                    np.mean([r["budget_used"] for r in rs]), np.mean([r["relapsed"] for r in rs]), g, shape))
    agg.sort(key=lambda a: a[0])
    v3 = [a for a in agg if a[5] == dict(pk=True, front=0.0, relief_boost=1.0, pregap_cut=1.0)][0]
    print("rank  expected_risk  mean_excess  budget_used  relapsed  config")
    for i, a in enumerate(agg[:10]):
        print(f"{i + 1:4d}  {a[0]:.4f}        {a[1]:+.4f}      {a[2]:.2f}         {a[3]:.3f}     {a[5]}")
    print(f" v3   {v3[0]:.4f}        {v3[1]:+.4f}      {v3[2]:.2f}         {v3[3]:.3f}     {v3[5]}  (rank {agg.index(v3) + 1} of {len(agg)})")
    print(f"worst {agg[-1][0]:.4f}  {agg[-1][5]}")
    print("\nmain effect of each setting (mean expected risk over the configs that use it):")
    for key in ("pk", "front", "relief_boost", "pregap_cut"):
        vals = sorted({s[key] for s in GRID})
        print(f"  {key:13s} " + "  ".join(f"{v}: {np.mean([a[0] for a in agg if a[5][key] == v]):.4f}" for v in vals))
    # paired: best config vs v3, per person
    best = agg[0][4]
    d = np.array([r["expected_risk"] for r in sorted([r for r in rows if r["config"] == best], key=lambda r: r["person"])]) - \
        np.array([r["expected_risk"] for r in sorted([r for r in rows if r["config"] == v3[4]], key=lambda r: r["person"])])
    rng = np.random.default_rng(0)
    bt = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    print(f"\nbest - v3, expected risk per person: {d.mean():+.4f} [{np.percentile(bt, 2.5):+.4f}, {np.percentile(bt, 97.5):+.4f}], "
          f"better for {int((d < 0).sum())} of {len(d)} people")
