"""Draw-length fixes and settings, judged on difficult people with irregular habits. SIMULATION ONLY.

  python3 experiments/hard_people_eval.py fixes    development set: which draw-length fix works best
  python3 experiments/hard_people_eval.py retune   development set: shaping settings for that fix, on difficult people
  python3 experiments/hard_people_eval.py test     test set, once: flat, v3, v4, the best fix, the final version
  python3 experiments/hard_people_eval.py noharm   original population B: does the final version hurt ordinary people?
People: experiments/hard_people.py. Scenarios:
  hard           irregular days, compensation 57 % through longer draws (LSBU split)
  hard_long      irregular days, 80 % through longer draws (stress test)
  hard_change    as hard, plus a new routine at taper week 6
"""
import csv
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from hard_people import load                                     # noqa: E402
from puff_nn import V3, V4, BaseNet, Flat, OnTheSpot              # noqa: E402
from puffsim import WARMUP_DAYS, SlowSchedule, simulate, test_population   # noqa: E402

SCEN = {"hard": dict(dur_share=0.57, irregular=True), "hard_long": dict(dur_share=0.8, irregular=True),
        "hard_change": dict(dur_share=0.57, irregular=True),
        "stable": None, "routine_change": None}
FIXES = {"v4": dict(), "v4_deliv": dict(accounting="delivered"), "v4_cap15": dict(cap_x=1.5), "v4_cap20": dict(cap_x=2.0),
         "v4_deliv_cap15": dict(accounting="delivered", cap_x=1.5), "v4_deliv_cap20": dict(accounting="delivered", cap_x=2.0)}
OUT = ROOT / "experiments"
CHOICE = OUT / "hard-people-choice.json"


def people(which):
    return load(which) if which in ("dev", "test") else test_population()


def make(spec, net):
    kind = spec["kind"]
    if kind == "flat":
        return Flat
    if kind == "flat_matched":
        share = spec["share"]

        class FlatMatched(Flat):
            def start_day(self, k, u, weekend):
                self.u = u * share
        return FlatMatched
    return lambda: OnTheSpot(spec.get("demand", "nn"), adapt=True, net=net, shape=spec["shape"],
                             accounting=spec.get("accounting", "dose"), cap_x=spec.get("cap_x"))


def run(args):
    which, pi, label, spec, scen, seed = args
    pop = people(which)
    person = pop[pi]
    change = (42, pop[(pi + 7) % len(pop)]) if scen in ("hard_change", "routine_change") else None
    net = BaseNet.load()
    rec = []
    r = simulate(person, make(spec, net), SlowSchedule(), seed=1000 * seed + pi, physiology=SCEN[scen],
                 routine_change=change, stop_at_relapse=False, record=rec)
    days = r["days"][WARMUP_DAYS:]
    bw = r["baseline_w"]
    excess = [d["mean_w"] - bw[d["day"] % 7] for d in days]
    u = {d["day"]: d["u"] for d in r["days"]}
    tp = [x for x in rec if x[0] >= WARMUP_DAYS and u.get(x[0], 0) > 0]
    base = r["days"][:WARMUP_DAYS]
    return dict(set=which, person=person.name, label=label, scenario=scen, seed=seed,
                expected_risk=r["expected_risk"], relapsed=int(r["relapsed"]),
                success=int(not r["relapsed"] and r["zero_day"] is not None),
                mean_excess=float(np.mean(excess)), weeks1_8=float(np.mean(excess[:56])),
                peak_week=float(max(np.mean(excess[i:i + 7]) for i in range(0, len(excess) - 6, 7))),
                delivered_share=float(np.mean([d["delivered"] / max(u_ * sum(b["delivered"] for b in base) / len(base), 1e-9)
                                               for d, u_ in ((d, d["u"]) for d in days) if u_ > 0])),
                empty_puffs=float(np.mean([x[3] == 0 for x in tp])) if tp else 0.0,
                max_dose=float(max((x[3] for x in tp), default=0.0)))


def table(rows, labels, scens, metric="expected_risk"):
    per = {}
    for r in rows:
        per.setdefault((r["label"], r["scenario"], r["person"]), []).append(r)
    names = sorted({r["person"] for r in rows})
    agg = lambda lab, sc, f: np.array([np.mean([x[f] for x in per[(lab, sc, n)]]) for n in names])
    out = {}
    print(f"{'version':24s} " + " ".join(f"{sc + ' risk':>17s}" for sc in scens) + "   excess   deliv.  empty")
    for lab in labels:
        vals = [agg(lab, sc, metric).mean() for sc in scens]
        ex = np.mean([agg(lab, sc, "mean_excess").mean() for sc in scens])
        dv = np.mean([agg(lab, sc, "delivered_share").mean() for sc in scens])
        em = np.mean([agg(lab, sc, "empty_puffs").mean() for sc in scens])
        out[lab] = dict(zip(scens, map(float, vals)), mean=float(np.mean(vals)), excess=float(ex), delivered=float(dv), empty=float(em))
        print(f"{lab:24s} " + " ".join(f"{v:17.4f}" for v in vals) + f"  {ex:+.4f}  {dv:6.3f}  {em:6.4f}")
    return out, agg, names


def paired(agg, a, b, scens, f="expected_risk", seed=0):
    d = np.mean([agg(a, sc, f) - agg(b, sc, f) for sc in scens], axis=0)
    rng = np.random.default_rng(seed)
    bt = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    return dict(diff=float(d.mean()), lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)),
                better=int((d < -0.005).sum()), worse=int((d > 0.005).sum()), n=len(d))


def go(jobs):
    with ProcessPoolExecutor() as ex:
        return list(ex.map(run, jobs, chunksize=4))


def save(rows, name):
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    mode = sys.argv[1]
    choice = json.loads(CHOICE.read_text()) if CHOICE.exists() else {}
    if mode == "fixes":
        scens = ["hard", "hard_long"]
        n = len(people("dev"))
        specs = {k: dict(kind="nn", shape=V4, **v) for k, v in FIXES.items()}
        rows = go([("dev", pi, k, s, sc, 0) for k, s in specs.items() for pi in range(n) for sc in scens])
        save(rows, "hard-people-fixes-dev.csv")
        out, agg, _ = table(rows, list(specs), scens)
        for k in list(specs)[1:]:
            p = paired(agg, k, "v4", scens)
            print(f"  {k:16s} - v4: {p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] better {p['better']} worse {p['worse']}")
        best = min(out, key=lambda k: out[k]["mean"])
        choice.update(fix=best, fix_spec=FIXES[best], fixes_dev=out)
        CHOICE.write_text(json.dumps(choice, indent=1) + "\n")
        print("best fix on the development set:", best)
    elif mode == "retune":
        scens = ["hard", "hard_long"]
        n = len(people("dev"))
        grid = [dict(pk=False, front=f, relief_boost=rb, pregap_cut=pc, pregap_h=ph, relief_gap_h=rg)
                for f, rb, pc, ph, rg in itertools.product((0.0, 0.5), (2.0, 2.5), (0.2, 0.35), (2.0, 3.0), (2.0, 3.0))]
        specs = {f"g{i:02d}": dict(kind="nn", shape=g, **choice["fix_spec"]) for i, g in enumerate(grid)}
        rows = go([("dev", pi, k, s, sc, 0) for k, s in specs.items() for pi in range(n) for sc in scens])
        save(rows, "hard-people-retune-dev.csv")
        out, agg, _ = table(rows, list(specs), scens)
        ranked = sorted(out, key=lambda k: out[k]["mean"])
        base = [k for k, s in specs.items() if all(s["shape"].get(x) == V4.get(x, d) for x, d in
                                                   (("front", 0), ("relief_boost", 0), ("pregap_cut", 0), ("pregap_h", 2.0), ("relief_gap_h", 3.0)))][0]
        for k in ranked[:6]:
            print(k, round(out[k]["mean"], 4), specs[k]["shape"])
        print("v4 settings:", base, round(out[base]["mean"], 4))
        p = paired(agg, ranked[0], base, scens)
        print(f"best - v4 settings: {p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] better {p['better']} worse {p['worse']}")
        choice.update(final_shape=specs[ranked[0]]["shape"], retune_best_vs_v4=p,
                      final_is_v4_shape=p["hi"] >= 0)        # keep v4's settings unless the gain is clear
        CHOICE.write_text(json.dumps(choice, indent=1) + "\n")
    elif mode in ("test", "noharm"):
        final_shape = V4 if choice.get("final_is_v4_shape", True) else choice["final_shape"]
        final = dict(kind="nn", shape=final_shape, **choice["fix_spec"])
        if mode == "test":
            which, scens, seeds = "test", ["hard", "hard_long", "hard_change"], (0, 1, 2)
            specs = {"flat": dict(kind="flat"), "v3": dict(kind="nn", shape=V3), "v4": dict(kind="nn", shape=V4),
                     "v4 + best fix": dict(kind="nn", shape=V4, **choice["fix_spec"]), "final": final,
                     "final, no network": dict(final, demand="habit")}
        else:
            which, scens, seeds = "B", ["stable", "routine_change"], (0, 1, 2)
            specs = {"flat": dict(kind="flat"), "v4": dict(kind="nn", shape=V4), "final": final}
        specs = {k: v for k, v in specs.items() if k == "final" or k == "flat" or v != final}
        n = len(people(which))
        rows = go([(which, pi, k, s, sc, seed) for k, s in specs.items() for pi in range(n) for sc in scens for seed in seeds])
        save(rows, f"hard-people-{mode}.csv")
        out, agg, names = table(rows, list(specs), scens)
        pairs = {}
        for a, b in (("final", "flat"), ("final", "v4"), ("final", "v3"), ("final", "final, no network"), ("v4", "flat")):
            if a in specs and b in specs:
                pairs[f"{a} - {b}"] = paired(agg, a, b, scens)
                p = pairs[f"{a} - {b}"]
                print(f"  {a:8s} - {b:18s} risk {p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] better {p['better']} worse {p['worse']} of {p['n']}")
                s_ = paired(agg, a, b, scens, "success")
                print(f"  {'':8s}   {'':18s} success {s_['diff']:+.3f} [{s_['lo']:+.3f}, {s_['hi']:+.3f}]")
        if mode == "test":
            for nm in ("Lucía", "Karim"):
                print(f"  {nm}: " + "  ".join(f"{k} {np.mean([agg(k, sc, 'expected_risk')[names.index(nm)] for sc in scens]):.3f}" for k in specs))
        (OUT / f"hard-people-{mode}-summary.json").write_text(json.dumps(dict(label="SIMULATION ONLY", table=out, pairs=pairs,
                                                                               final=final), indent=1, ensure_ascii=False) + "\n")
