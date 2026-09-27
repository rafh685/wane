"""Interview-driven starting plans: tuning on development people, one test on untouched people. SIMULATION ONLY.

  python3 experiments/interview_eval.py tune     random search over InterviewPolicy settings (development people)
  python3 experiments/interview_eval.py refine   a second search around the best setting
  python3 experiments/interview_eval.py test     once: v4 at 12 %, v4 at the same average length, interview, weak
                                                 interview answers, and the hidden-difficulty upper bound
People: development = difficult development set (50, irregular habits) + general development population (40,
seed 31); test = difficult test set (52 incl. Lucía, Karim) + population B (65). Answers are fixed per person.
Every run: 3 baseline weeks, up to 52 weeks of taper, 4 weeks after; relapse draws recorded, never stopping a run.
Selection rule, fixed before running: lowest J = mean expected risk (difficult) + 0.5 x mean expected risk
(general), among settings whose mean time to zero over the development people is at most 26 weeks (v4 at 12 %
takes 23). Tuning uses the habit-only demand model for speed; the test uses Felipe's network.
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
sys.path.insert(0, str(ROOT / "experiments"))
from hard_people import HARD, load                                          # noqa: E402
from interview import InterviewPolicy, synthetic_answers, weeks_to_zero    # noqa: E402
from population import population                                           # noqa: E402
from puff_nn import V4, BaseNet, OnTheSpot                                  # noqa: E402
from puffsim import WARMUP_DAYS, SlowSchedule, simulate, test_population   # noqa: E402
from relapse import evaluate                                                # noqa: E402

V2 = os.environ.get("RELAPSE") == "v2"          # relapse model v2 (relapse.py) with a 28-day burn-in
SUFFIX = "-v2" if V2 else ""
OUT = ROOT / "experiments"
CHOICE = OUT / f"interview-choice{SUFFIX}.json"
TAPER_DAYS, FOLLOW_DAYS, MAX_WEEKS = 364, 28, 26


def people(which):
    if which == "dev":
        return [(p, "hard") for p in load("dev")] + [(p, "general") for p in population(40, seed=31)]
    return [(p, "hard") for p in load("test")] + [(p, "general") for p in test_population()]


def answers(which, link_scale=1.0, random_rate=0.15):
    rng = np.random.default_rng({"dev": 5001, "test": 5002}[which] + int(link_scale * 100))
    return [synthetic_answers(p, rng, link_scale=link_scale, random_rate=random_rate) for p, _ in people(which)]


def run(args):
    which, i, label, pol, ans, seed, demand = args
    p, group = people(which)[i]
    plan = pol.plan(ans, p)
    net = BaseNet.load() if demand == "nn" else None
    make = lambda: OnTheSpot(demand, adapt=True, net=net, shape=plan["shape"])
    r = simulate(p, make, SlowSchedule(weekly_cut=plan["cut"]), taper_days=TAPER_DAYS, follow_days=FOLLOW_DAYS,
                 seed=1000 * seed + i, physiology=HARD if group == "hard" else None, stop_at_relapse=False,
                 burn_in_days=28 if V2 else 0)
    if V2:
        ev = evaluate(r, p, seed=1000 * seed + i, n_mc=300 if demand == "habit" else 500)
        r["expected_risk"], r["relapsed"] = ev["expected_risk"], ev["expected_risk"] > 0.5
    days = r["days"][WARMUP_DAYS:]
    bw = r["baseline_w"]
    return dict(set=which, person=p.name, group=group, label=label, seed=seed, cut=plan["cut"], score=plan["score"],
                weeks_to_zero=weeks_to_zero(plan["cut"]), expected_risk=r["expected_risk"], relapsed=int(r["relapsed"]),
                success=(float(1 - r["expected_risk"]) if V2 else int(not r["relapsed"])) * (r["zero_day"] is not None),
                mean_excess=float(np.mean([d["mean_w"] - bw[d["day"] % 7] for d in days])))


def summarise(rows, labels):
    out = {}
    for lab in labels:
        rs = [r for r in rows if r["label"] == lab]
        g = lambda grp, f: float(np.mean([r[f] for r in rs if r["group"] == grp]))
        out[lab] = dict(risk_hard=g("hard", "expected_risk"), risk_general=g("general", "expected_risk"),
                        success_hard=g("hard", "success"), success_general=g("general", "success"),
                        weeks=float(np.mean([r["weeks_to_zero"] for r in rs])),
                        weeks_hard=g("hard", "weeks_to_zero"), weeks_general=g("general", "weeks_to_zero"))
        out[lab]["J"] = out[lab]["risk_hard"] + 0.5 * out[lab]["risk_general"]
    return out


def random_policy(rng, around=None):
    if around is None:
        return InterviewPolicy(base_cut=float(rng.uniform(0.10, 0.20)), k=float(rng.uniform(0.5, 3.5)),
                               w=dict(ttfv=float(rng.uniform(0.3, 1)), night=float(rng.uniform(0, 1)),
                                      history=float(rng.uniform(0, 1)), mood=float(rng.uniform(0, 1))),
                               morning_x=float(rng.choice([0, 0.25, 0.5])), night_relief=bool(rng.random() < 0.5),
                               pregap_social=float(rng.choice([0.2, 0.5, 0.8])))
    b = around
    j = lambda x, s, lo, hi: float(np.clip(x + rng.normal(0, s), lo, hi))
    return InterviewPolicy(base_cut=j(b.base_cut, 0.015, 0.08, 0.25), k=j(b.k, 0.4, 0, 5),
                           w={q: j(v, 0.15, 0, 1) for q, v in b.w.items()},
                           morning_x=float(rng.choice([b.morning_x, 0, 0.25, 0.5])),
                           night_relief=b.night_relief if rng.random() < 0.7 else not b.night_relief,
                           pregap_social=float(rng.choice([b.pregap_social, 0.2, 0.5, 0.8])))


def pol_dict(p):
    return dict(base_cut=p.base_cut, k=p.k, w=p.w, morning_x=p.morning_x, night_relief=p.night_relief,
                pregap_social=p.pregap_social, oracle=p.oracle)


def search(policies, name):
    ans = answers("dev")
    n = len(people("dev"))
    jobs = [("dev", i, lab, pol, ans[i], 0, "habit") for lab, pol in policies.items() for i in range(n)]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(run, jobs, chunksize=4))
    with open(OUT / f"interview-{name}-dev{SUFFIX}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return summarise(rows, list(policies))


def report(summ, policies):
    print(f"{'setting':10s} {'J':>7s} {'risk hard':>9s} {'risk gen':>9s} {'succ hard':>9s} {'weeks':>6s} {'wk hard':>7s} {'wk gen':>6s}")
    for lab in sorted(summ, key=lambda k: summ[k]["J"]):
        s = summ[lab]
        print(f"{lab:10s} {s['J']:7.4f} {s['risk_hard']:9.4f} {s['risk_general']:9.4f} {s['success_hard']:9.3f} "
              f"{s['weeks']:6.1f} {s['weeks_hard']:7.1f} {s['weeks_general']:6.1f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    choice = json.loads(CHOICE.read_text()) if CHOICE.exists() else {}
    if mode in ("tune", "refine"):
        rng = np.random.default_rng(11 if mode == "tune" else 12)
        if mode == "tune":
            policies = {f"uni{int(c * 100):02d}": InterviewPolicy(base_cut=c, k=0.0) for c in (0.08, 0.10, 0.12, 0.15)}
            policies.update({f"oracle{k}": InterviewPolicy(base_cut=0.12, k=k, oracle=True) for k in (1, 2, 3)})
            policies.update({f"r{i:02d}": random_policy(rng) for i in range(40)})
        else:
            best = InterviewPolicy(**choice["best"])
            policies = {"best0": best}
            policies.update({f"n{i:02d}": random_policy(rng, around=best) for i in range(24)})
        summ = search(policies, mode)
        report(summ, policies)
        ok = {k: v for k, v in summ.items() if not k.startswith(("uni", "oracle")) and v["weeks"] <= MAX_WEEKS}
        best = min(ok, key=lambda k: ok[k]["J"])
        prev = choice.get("best_J", 1e9)
        if summ[best]["J"] < prev:
            choice.update(best=pol_dict(policies[best]), best_J=summ[best]["J"], best_summary=summ[best])
        choice.setdefault("rounds", {})[mode] = dict(best=best, summary=summ)
        CHOICE.write_text(json.dumps(choice, indent=1) + "\n")
        print("chosen:", best, round(summ[best]["J"], 4), pol_dict(policies[best]))
    elif mode == "test":
        best = InterviewPolicy(**choice["best"])
        ans = answers("test")
        weak = answers("test", link_scale=0.5, random_rate=0.30)
        n = len(people("test"))
        mean_weeks = float(np.mean([weeks_to_zero(best.plan(a, p)["cut"]) for a, (p, _) in zip(ans, people("test"))]))
        matched = min(np.arange(0.06, 0.20, 0.0025), key=lambda c: abs(weeks_to_zero(c) - mean_weeks))
        oracle_k = choice.get("oracle_k", 2)
        policies = {"v4 12%": InterviewPolicy(base_cut=0.12), f"v4 same length ({matched:.3f})": InterviewPolicy(base_cut=float(matched)),
                    "interview": best, "interview, weak answers": best,
                    "hidden difficulty (upper bound)": InterviewPolicy(base_cut=best.base_cut, k=best.k, oracle=True)}
        jobs = []
        for lab, pol in policies.items():
            src = weak if lab == "interview, weak answers" else ans
            for i in range(n):
                for seed in (0, 1, 2):
                    jobs.append(("test", i, lab, pol, src[i], seed, "nn"))
        with ProcessPoolExecutor() as ex:
            rows = list(ex.map(run, jobs, chunksize=4))
        with open(OUT / f"interview-test{SUFFIX}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summ = summarise(rows, list(policies))
        report(summ, policies)
        pairs = {}
        names = sorted({(r["person"], r["group"]) for r in rows})
        per = lambda lab, f: np.array([np.mean([r[f] for r in rows if r["label"] == lab and r["person"] == n_ and r["group"] == g])
                                       for n_, g in names])
        grp = np.array([g for _, g in names])
        rng = np.random.default_rng(0)
        for a, b in (("interview", "v4 12%"), ("interview", list(policies)[1]), ("interview, weak answers", list(policies)[1]),
                     ("hidden difficulty (upper bound)", "interview")):
            for f in ("expected_risk", "success"):
                for g in ("hard", "general"):
                    d = (per(a, f) - per(b, f))[grp == g]
                    bt = [rng.choice(d, len(d)).mean() for _ in range(4000)]
                    key = f"{a} - {b} | {f} | {g}"
                    pairs[key] = dict(diff=float(d.mean()), lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)),
                                      better=int((d < -0.005).sum() if f == "expected_risk" else (d > 0.005).sum()),
                                      worse=int((d > 0.005).sum() if f == "expected_risk" else (d < -0.005).sum()), n=len(d))
                    p_ = pairs[key]
                    print(f"  {key:80s} {p_['diff']:+.4f} [{p_['lo']:+.4f}, {p_['hi']:+.4f}] better {p_['better']} worse {p_['worse']}")
        for nm in ("Lucía", "Karim"):
            print(f"  {nm}: " + "  ".join(f"{lab}: risk {np.mean([r['expected_risk'] for r in rows if r['label'] == lab and r['person'] == nm]):.3f} "
                                            f"cut {np.mean([r['cut'] for r in rows if r['label'] == lab and r['person'] == nm]):.3f}"
                                            for lab in policies))
        (OUT / f"interview-test-summary{SUFFIX}.json").write_text(json.dumps(dict(label="SIMULATION ONLY", matched_cut=float(matched),
                                                                         mean_weeks_interview=mean_weeks, table=summ, pairs=pairs,
                                                                         policy=pol_dict(best)), indent=1, ensure_ascii=False) + "\n")
