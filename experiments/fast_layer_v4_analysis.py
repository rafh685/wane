"""Analysis of experiments/fast-layer-v4-runs.csv. SIMULATION ONLY, relative comparisons.

Per person, the 3 seeds are averaged first; intervals are 95 % bootstrap over people (paired).
Writes experiments/fast-layer-v4-summary.json and prints the report.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "fast-layer-v4-runs.csv"
SCENARIOS = ("stable", "routine_change", "model_mismatch")
KEYS = ["flat", "flat_matched", "codex_rules", "habit_v3", "nn_v3", "habit_v4", "nn_v4_frozen", "nn_v4"]
METRICS = ["expected_risk", "success", "relapsed", "mean_excess", "peak_week_excess", "weeks1_8_excess", "budget_used",
           "zero_dose_share", "max_dose", "max_dose_over_target", "puffs_ratio_first12w"]
PAIRS = [("nn_v4", "flat"), ("nn_v4", "flat_matched"), ("nn_v4", "nn_v3"), ("habit_v4", "habit_v3"),
         ("nn_v4", "habit_v4"), ("nn_v4", "nn_v4_frozen"), ("nn_v4", "codex_rules")]


def load():
    rows = list(csv.DictReader(open(RUNS)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def per_person(rows, scenario, key, metric):
    acc = defaultdict(list)
    for r in rows:
        if r["scenario"] == scenario and r["controller"] == key:
            acc[r["person"]].append(r[metric])
    return {p: float(np.nanmean(v)) for p, v in acc.items()}


def paired(a, b, seed=0):
    people = sorted(set(a) & set(b))
    d = np.array([a[p] - b[p] for p in people])
    rng = np.random.default_rng(seed)
    bt = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    return dict(diff=float(d.mean()), lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)),
                better=int((d < -0.005).sum()), worse=int((d > 0.005).sum()), n=len(d))


def main():
    rows = load()
    people = sorted({r["person"] for r in rows})
    attrs = {r["person"]: r for r in rows}
    out = {"label": "SIMULATION ONLY: synthetic people, relative comparison", "people": len(people), "scenario": {}}
    for s in SCENARIOS:
        block = {"table": {}, "pairs": {}}
        print(f"\n=== scenario: {s} ({len(people)} people x 3 seeds) ===")
        print(f"{'controller':14s} {'exp.risk':>8s} {'success':>8s} {'relapse':>8s} {'excess':>8s} {'peakwk':>7s} {'wk1-8':>7s} {'budget':>7s} {'0-dose':>7s} {'maxdose':>7s} {'max/u':>6s} {'puffs x':>7s}")
        for k in KEYS:
            m = {mt: float(np.mean(list(per_person(rows, s, k, mt).values()))) for mt in METRICS}
            m["max_dose"] = float(max(per_person(rows, s, k, "max_dose").values()))
            m["max_dose_over_target"] = float(max(per_person(rows, s, k, "max_dose_over_target").values()))
            block["table"][k] = m
            print(f"{k:14s} {m['expected_risk']:8.4f} {m['success']:8.3f} {m['relapsed']:8.3f} {m['mean_excess']:+8.4f} {m['peak_week_excess']:7.4f} "
                  f"{m['weeks1_8_excess']:+7.4f} {m['budget_used']:7.2f} {m['zero_dose_share']:7.3f} {m['max_dose']:7.2f} {m['max_dose_over_target']:6.2f} {m['puffs_ratio_first12w']:7.2f}")
        print("paired differences (first minus second), 95 % interval, people better / worse by > 0.005 expected risk:")
        for a, b in PAIRS:
            er = paired(per_person(rows, s, a, "expected_risk"), per_person(rows, s, b, "expected_risk"))
            su = paired(per_person(rows, s, a, "success"), per_person(rows, s, b, "success"))
            block["pairs"][f"{a} - {b}"] = dict(expected_risk=er, success=su)
            print(f"  {a:12s} - {b:13s} risk {er['diff']:+.4f} [{er['lo']:+.4f}, {er['hi']:+.4f}]  better {er['better']:2d} worse {er['worse']:2d}   "
                  f"success {su['diff']:+.3f} [{su['lo']:+.3f}, {su['hi']:+.3f}]")
        out["scenario"][s] = block

    print("\n=== who gains: nn_v4 minus flat, expected risk, per person averaged over the 3 scenarios ===")
    avg = lambda key: {p: np.mean([per_person(rows, s, key, "expected_risk")[p] for s in SCENARIOS]) for p in people}
    v4, fl, v3 = avg("nn_v4"), avg("flat"), avg("nn_v3")
    groups = {}
    groups["routine"] = defaultdict(list)
    for p in people:
        groups["routine"][attrs[p]["kind"]].append(p)
    for attr, label in (("puffs_per_day", "puffs/day"), ("elasticity", "compensation"), ("craving_decay", "adaptation speed"),
                        ("relapse_threshold", "relapse threshold")):
        vals = np.array([attrs[p][attr] for p in people])
        q = np.quantile(vals, [1 / 3, 2 / 3])
        g = defaultdict(list)
        for p in people:
            x = attrs[p][attr]
            g["low" if x <= q[0] else "mid" if x <= q[1] else "high"].append(p)
        groups[label] = g
    out["subgroups"] = {}
    for label, g in groups.items():
        print(f"  by {label}:")
        out["subgroups"][label] = {}
        for name in sorted(g):
            ps = g[name]
            d_flat = np.mean([v4[p] - fl[p] for p in ps])
            d_v3 = np.mean([v4[p] - v3[p] for p in ps])
            out["subgroups"][label][name] = dict(n=len(ps), v4_minus_flat=float(d_flat), v4_minus_v3=float(d_v3),
                                                 flat_risk=float(np.mean([fl[p] for p in ps])))
            print(f"    {name:14s} n={len(ps):2d}  flat risk {np.mean([fl[p] for p in ps]):.3f}  v4 - flat {d_flat:+.4f}  v4 - v3 {d_v3:+.4f}")

    print("\n=== harm check: people for whom nn_v4 raises expected risk by > 0.02 over flat or nn_v3 in any scenario ===")
    harms = []
    for s in SCENARIOS:
        a, f, t = (per_person(rows, s, k, "expected_risk") for k in ("nn_v4", "flat", "nn_v3"))
        for p in people:
            for ref, refname in ((f, "flat"), (t, "nn_v3")):
                if a[p] - ref[p] > 0.02:
                    harms.append(dict(person=p, scenario=s, vs=refname, v4=a[p], ref=ref[p], kind=attrs[p]["kind"],
                                      puffs=attrs[p]["puffs_per_day"], elasticity=attrs[p]["elasticity"]))
    for h in harms:
        print(f"  {h['person']:8s} {h['scenario']:14s} vs {h['vs']:6s} v4 {h['v4']:.3f} ref {h['ref']:.3f}  ({h['kind']}, {h['puffs']:.0f} puffs/day, compensation {h['elasticity']:.2f})")
    if not harms:
        print("  none")
    out["harms"] = harms

    print("\n=== named people, success (mean of 3 seeds) and expected risk, stable scenario ===")
    for n in ("Marta", "Diego", "Lucía", "Karim", "Ana"):
        line = "  " + f"{n:6s}"
        for k in ("flat", "nn_v3", "nn_v4"):
            line += f"  {k} {per_person(rows, 'stable', k, 'success')[n]:.2f}/{per_person(rows, 'stable', k, 'expected_risk')[n]:.3f}"
        print(line)
    (ROOT / "experiments" / "fast-layer-v4-summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
