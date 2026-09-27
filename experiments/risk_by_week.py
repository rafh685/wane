"""Where simulated relapse risk comes from: weekly excess hazard with and without any nicotine cut. SIMULATION ONLY.

Finding (27 Sept 2026): with NO cut at all the flat taper still accumulates about 0.025 excess hazard a week on
difficult people, because day-to-day withdrawal noise (SD about 0.12) is as large as the relapse threshold
(0.08 to 0.17). See docs/interview_priors.md.
"""
import sys; from pathlib import Path; ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from hard_people import HARD, load
from puff_nn import V4, OnTheSpot, Flat
from puffsim import WARMUP_DAYS, SlowSchedule, simulate
PEOPLE = load("dev")
class NoCut(SlowSchedule):
    def level(self, taper_day, last):
        return 1.0
def run(args):
    i, ctrl, sched = args
    make = Flat if ctrl == "flat" else (lambda: OnTheSpot("habit", shape=V4))
    slow = NoCut() if sched == "nocut" else SlowSchedule(weekly_cut=0.12)
    r = simulate(PEOPLE[i], make, slow, taper_days=84, follow_days=0, seed=i, physiology=HARD, stop_at_relapse=False)
    days = r["days"][WARMUP_DAYS:]
    bw = r["baseline_w"]
    wk = [sum(d.get("hazard", 0) - 0.0003 for d in days[7 * w:7 * w + 7]) for w in range(12)]
    ex = [np.mean([d["mean_w"] - bw[d["day"] % 7] for d in days[7 * w:7 * w + 7]]) for w in range(12)]
    sd = np.std([d["mean_w"] for d in r["days"][7:WARMUP_DAYS]])
    return ctrl, sched, wk, ex, sd
if __name__ == "__main__":
    jobs = [(i, c, s) for c in ("flat", "v4") for s in ("nocut", "cut12") for i in range(len(PEOPLE))]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, jobs))
    for c in ("flat", "v4"):
        for s in ("nocut", "cut12"):
            rs = [r for r in res if r[0] == c and r[1] == s]
            print(f"{c:4s} {s:5s} excess hazard per week 1..12: " + " ".join(f"{np.mean([r[2][w] for r in rs]):.3f}" for w in range(12)))
            print(f"{'':10s} excess W per week 1..12:      " + " ".join(f"{np.mean([r[3][w] for r in rs]):+.3f}" for w in range(12)))
    print("day-to-day SD of awake W in baseline weeks 2-3:", round(float(np.mean([r[4] for r in res])), 4))
