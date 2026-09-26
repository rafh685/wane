"""Final test of fast layer v4 against v3, flat, a nicotine-matched flat and Codex's rules. SIMULATION ONLY.

Test population B (60 synthetic people, seed 12) + the five named profiles, 3 seeds, three scenarios (stable,
routine change at taper week 6, controllers' nicotine model deliberately wrong). v4's settings were chosen on
the development population (seed 31, experiments/tune_fast_layer.py) before this file was run.
Relapse draws are recorded but do not stop a run, so every run also gives its full-path expected relapse risk.
Writes experiments/fast-layer-v4-runs.csv; analysis: experiments/fast_layer_v4_analysis.py.
Needs Codex's puff_controller.py importable (WANE_MAIN) for the codex_rules baseline.
"""
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get("WANE_MAIN"):
    sys.path.append(os.environ["WANE_MAIN"])

from puff_nn import V3, V4, BaseNet, CodexRules, Flat, OnTheSpot          # noqa: E402
from puffsim import WARMUP_DAYS, SlowSchedule, simulate, test_population   # noqa: E402

SEEDS = (0, 1, 2)
SCENARIOS = ("stable", "routine_change", "model_mismatch")
MATCH = float(os.environ.get("FLAT_MATCH", "0.9"))      # nicotine share of the matched flat control


class FlatScaled(Flat):
    """Flat, but at a fixed fraction of today's level: the same nicotine as v4 without v4's timing."""
    name = "flat, nicotine-matched"

    def start_day(self, k, u, weekend):
        self.u = u * MATCH


def make(key, net):
    return {
        "flat": Flat,
        "flat_matched": FlatScaled,
        "codex_rules": CodexRules,
        "habit_v3": lambda: OnTheSpot("habit", shape=V3),
        "nn_v3": lambda: OnTheSpot("nn", adapt=True, net=net, shape=V3),
        "habit_v4": lambda: OnTheSpot("habit", shape=V4),
        "nn_v4_frozen": lambda: OnTheSpot("nn", adapt=False, net=net, shape=V4),
        "nn_v4": lambda: OnTheSpot("nn", adapt=True, net=net, shape=V4),
    }[key]


KEYS = ["flat", "flat_matched", "codex_rules", "habit_v3", "nn_v3", "habit_v4", "nn_v4_frozen", "nn_v4"]
NAMED_KIND = {"Marta": "all_day", "Diego": "evening", "Lucía": "three_moments", "Karim": "work_ban", "Ana": "three_moments"}


def one(args):
    ci, key, scenario, seed = args
    pop = test_population()
    person = pop[ci]
    change = (42, pop[(ci + 7) % len(pop)]) if scenario == "routine_change" else None
    phys = {"half_life_h": 3.0, "dur_exp": 1.0} if scenario == "model_mismatch" else None
    net = BaseNet.load()
    holder, rec = {}, []

    def factory():
        holder["c"] = make(key, net)()
        return holder["c"]

    r = simulate(person, factory, SlowSchedule(), seed=1000 * seed + ci, routine_change=change, physiology=phys,
                 stop_at_relapse=False, record=rec)
    days = r["days"][WARMUP_DAYS:]
    bw = r["baseline_w"]
    excess = [d["mean_w"] - bw[d["day"] % 7] for d in days]
    weekly = [np.mean(excess[i:i + 7]) for i in range(0, len(excess) - 6, 7)]
    u = {d["day"]: d["u"] for d in r["days"]}
    tp = [(x[3], u[x[0]]) for x in rec if x[0] >= WARMUP_DAYS and u.get(x[0], 0) > 0]
    err = getattr(holder["c"], "day_err", [])
    base_puffs = r["baseline_puffs"]
    kind = person.story.replace("synthetic ", "") if person.story.startswith("synthetic") else NAMED_KIND.get(person.name, "named")
    return dict(
        person=person.name, kind=kind, puffs_per_day=round(person.puffs_per_day, 1), elasticity=round(person.elasticity, 3),
        craving_decay=round(person.craving_decay, 3), relapse_threshold=round(person.relapse_threshold, 2),
        seed=seed, controller=key, scenario=scenario,
        expected_risk=r["expected_risk"], relapsed=int(r["relapsed"]), relapse_day=r["relapse_day"],
        success=int(not r["relapsed"] and r["zero_day"] is not None),
        mean_excess=float(np.mean(excess)), peak_week_excess=float(max(weekly)), weeks1_8_excess=float(np.mean(excess[:56])),
        budget_used=float(np.mean([d["dose_sum"] / (d["u"] * base_puffs) for d in days if d["u"] > 0])),
        zero_dose_share=float(np.mean([d == 0 for d, uu in tp])) if tp else 0.0,
        max_dose_over_target=float(max(d / uu for d, uu in tp)) if tp else 0.0,
        max_dose=float(max(d for d, uu in tp)) if tp else 0.0,
        puffs_ratio_first12w=float(np.mean([d["puffs"] for d in days[:84]]) / base_puffs),
        pred_err_first2w=float(np.mean(err[:14])) if len(err) >= 14 else float("nan"),
        pred_err_weeks7to10=float(np.mean(err[42:70])) if len(err) >= 70 else float("nan"),
        pred_err_last4w=float(np.mean(err[-28:])) if len(err) >= 84 else float("nan"),
    )


if __name__ == "__main__":
    people = test_population()
    jobs = [(ci, k, s, seed) for ci in range(len(people)) for k in KEYS for s in SCENARIOS for seed in SEEDS]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(one, jobs, chunksize=4))
    with open(ROOT / "experiments" / "fast-layer-v4-runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} runs written")
