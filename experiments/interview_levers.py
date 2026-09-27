"""The interview's non-pace levers one at a time at the v4 pace (12 %), development people, 3 seeds. SIMULATION ONLY."""
import sys; from pathlib import Path; ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from interview_eval import answers, people, run
from interview import InterviewPolicy
POL = {"v4 12%": InterviewPolicy(), "+ morning relief": InterviewPolicy(morning_x=0.5),
       "+ night relief": InterviewPolicy(night_relief=True), "+ drinking nights": InterviewPolicy(pregap_social=0.5),
       "all three": InterviewPolicy(morning_x=0.5, night_relief=True, pregap_social=0.5)}
if __name__ == "__main__":
    ans = answers("dev"); n = len(people("dev"))
    jobs = [("dev", i, lab, pol, ans[i], s, "habit") for lab, pol in POL.items() for i in range(n) for s in (0, 1, 2)]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(run, jobs, chunksize=4))
    names = sorted({(r["person"], r["group"]) for r in rows})
    per = lambda lab: np.array([np.mean([r["expected_risk"] for r in rows if r["label"] == lab and (r["person"], r["group"]) == k]) for k in names])
    grp = np.array([g for _, g in names]); rng = np.random.default_rng(0)
    # who the lever touches: night wakers / drinkers / early vapers
    touched = {"+ morning relief": [a["ttfv"] <= 1 for a in ans], "+ night relief": [a["night"] >= 2 for a in ans],
               "+ drinking nights": [a.get("alcohol") == 1 for a in ans], "all three": [True] * n}
    order = [(p.name, g) for p, g in people("dev")]
    for lab in list(POL)[1:]:
        d = per(lab) - per("v4 12%")
        for g in ("hard", "general"):
            dd = d[grp == g]; bt = [rng.choice(dd, len(dd)).mean() for _ in range(3000)]
            print(f"{lab:18s} {g:7s} risk vs v4 {dd.mean():+.4f} [{np.percentile(bt, 2.5):+.4f}, {np.percentile(bt, 97.5):+.4f}]")
        mask = np.array([touched[lab][order.index(k)] for k in names])
        dd = d[mask]; bt = [rng.choice(dd, len(dd)).mean() for _ in range(3000)]
        print(f"{'':18s} only people it applies to (n={mask.sum()}): {dd.mean():+.4f} [{np.percentile(bt, 2.5):+.4f}, {np.percentile(bt, 97.5):+.4f}]")
