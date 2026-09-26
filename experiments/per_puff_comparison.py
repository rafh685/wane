"""Five fast layers on the same synthetic people, same slow layer. SIMULATION ONLY, relative comparison.

Population B (60 synthetic people, seed 12, never used for training) plus the five named profiles, 3 seeds each.
Slow layer: 12 %/week for everyone. 26 weeks of taper after 3 baseline weeks, then 4 weeks at zero.
Two scenarios: "stable" (the person keeps their routine) and "routine_change" (at taper week 6 the person takes
on another person's routine and sleep times, so the habit learned in the baseline weeks goes stale), and
"model_mismatch" (the hidden person clears nicotine with a 3 h half-life and puff length acts linearly, while
every controller still assumes 2 h and a 0.7 exponent: the controllers' nicotine model is wrong on purpose).

Needs Codex's puff_controller.py importable (main checkout); set WANE_MAIN if it is not on the path.
Run from the repo root: python3 experiments/per_puff_comparison.py
"""
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get("WANE_MAIN"):
    sys.path.append(os.environ["WANE_MAIN"])

from puff_nn import controllers                      # noqa: E402
from puffsim import WARMUP_DAYS, SlowSchedule, test_population, simulate   # noqa: E402

SEEDS = (0, 1, 2)
SCENARIOS = ("stable", "routine_change", "model_mismatch")
OUT = ROOT / "experiments"


def one(args):
    ci, key, scenario, seed = args
    pop = test_population()
    person = pop[ci]
    change = (42, pop[(ci + 7) % len(pop)]) if scenario == "routine_change" else None
    make = controllers()[key]
    holder = {}

    def factory():
        holder["c"] = make()
        return holder["c"]

    phys = {"half_life_h": 3.0, "dur_exp": 1.0} if scenario == "model_mismatch" else None
    r = simulate(person, factory, SlowSchedule(), seed=1000 * seed + ci, routine_change=change, physiology=phys)
    days = r["days"][WARMUP_DAYS:]
    base = r["days"][:WARMUP_DAYS]
    bw = r["baseline_w"]
    excess = [d["mean_w"] - bw[d["day"] % 7] for d in days]
    weekly = [np.mean(excess[i:i + 7]) for i in range(0, max(len(excess) - 6, 1), 7)] if excess else [0.0]
    base_puffs = np.mean([d["puffs"] for d in base])
    early = [d["puffs"] for d in days[:84]]
    err = getattr(holder.get("c"), "day_err", [])
    return dict(
        person=person.name, seed=seed, controller=key, scenario=scenario,
        relapsed=int(r["relapsed"]), relapse_day=r["relapse_day"],
        success=int((not r["relapsed"]) and r["zero_day"] is not None),
        zero_day=r["zero_day"],
        mean_excess_w=float(np.mean(excess)) if excess else 0.0,
        peak_week_excess_w=float(max(weekly)),
        puffs_ratio_first12w=float(np.mean(early) / base_puffs) if early else float("nan"),
        dose_per_day_first12w=float(np.mean([d["dose_sum"] for d in days[:84]])) / base_puffs if early else float("nan"),
        pred_err_first2w=float(np.mean(err[:14])) if len(err) >= 14 else float("nan"),
        pred_err_weeks7to10=float(np.mean(err[42:70])) if len(err) >= 70 else float("nan"),
        pred_err_last4w=float(np.mean(err[-28:])) if len(err) >= 84 else float("nan"),
    )


def paired_ci(a, b, n_boot=4000, seed=0):
    """Mean of a - b over people, 95 % bootstrap interval, resampling people."""
    d = np.array(a) - np.array(b)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


if __name__ == "__main__":
    people = test_population()
    keys = list(controllers())
    jobs = [(ci, k, s, seed) for ci in range(len(people)) for k in keys for s in SCENARIOS for seed in SEEDS]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(one, jobs, chunksize=4))
    with open(OUT / "per-puff-v3-runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    summary = {"label": "SIMULATION ONLY: synthetic people, relative comparison between fast layers",
               "people": len(people), "seeds": list(SEEDS), "runs": len(rows), "scenario": {}}
    for s in SCENARIOS:
        block = {}
        per = {k: {} for k in keys}
        for r in rows:
            if r["scenario"] == s:
                per[r["controller"]].setdefault(r["person"], []).append(r)
        names = sorted(per["flat"])
        avg = lambda k, f: [np.nanmean([x[f] for x in per[k][n]]) for n in names]
        for k in keys:
            block[k] = {f: float(np.nanmean(avg(k, f))) for f in
                        ("relapsed", "success", "mean_excess_w", "peak_week_excess_w", "puffs_ratio_first12w",
                         "dose_per_day_first12w", "pred_err_first2w", "pred_err_weeks7to10", "pred_err_last4w")}
            if k != "flat":
                block[k]["vs_flat_success"] = paired_ci(avg(k, "success"), avg("flat", "success"))
                block[k]["vs_flat_relapsed"] = paired_ci(avg(k, "relapsed"), avg("flat", "relapsed"))
            if k != "nn_frozen" and k.startswith("nn"):
                block[k]["vs_frozen_success"] = paired_ci(avg(k, "success"), avg("nn_frozen", "success"))
        block["named"] = {n: {k: float(np.mean([x["success"] for x in per[k][n]])) for k in keys}
                          for n in ("Marta", "Diego", "Lucía", "Karim", "Ana")}
        summary["scenario"][s] = block
    (OUT / "per-puff-v3-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    for s in SCENARIOS:
        print(f"\nscenario: {s}   (SIMULATION, {len(people)} people x {len(SEEDS)} seeds)")
        print(f"{'fast layer':16s} {'success':>8s} {'relapse':>8s} {'excessW':>8s} {'peakW':>7s} {'puffs x':>8s} {'dose/d':>7s} {'err0':>6s} {'err7-10':>7s} {'errEnd':>6s}")
        for k in keys:
            b = summary["scenario"][s][k]
            print(f"{k:16s} {b['success']:8.3f} {b['relapsed']:8.3f} {b['mean_excess_w']:8.3f} {b['peak_week_excess_w']:7.3f} "
                  f"{b['puffs_ratio_first12w']:8.2f} {b['dose_per_day_first12w']:7.3f} {b['pred_err_first2w']:6.3f} {b['pred_err_weeks7to10']:7.3f} {b['pred_err_last4w']:6.3f}")
        for k in keys[1:]:
            m, lo, hi = summary["scenario"][s][k]["vs_flat_success"]
            print(f"  {k} vs flat, success difference: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        m, lo, hi = summary["scenario"][s]["nn_adaptive"]["vs_frozen_success"]
        print(f"  nn_adaptive vs nn_frozen, success difference: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
