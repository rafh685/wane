"""How much does Wane's gain over the traditional taper depend on how the traditional taper is modelled? SIMULATION ONLY.

Same people, seeds and windows as experiments/traditional_taper_eval.py; only the hand-taper assumptions change.
Outcome: nicotine-free at 9 months (not relapsed, on 0 mg/ml on the last day). Wane's runs are reused from
experiments/relapse-v2-runs.csv. Every variant keeps the main assumptions except what its name says.
Run: WANE_MAIN=/path/to/main python3 experiments/traditional_taper_sensitivity.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from puffsim import ManualTaper                                              # noqa: E402
from traditional_taper_eval import FIRST_VERSION, OUT, paired, people, per_person, run_traditional, wane_rows  # noqa: E402

SALT = (1.0, 0.5, 0.25, 0.0)
VARIANTS = {
    "main assumptions": dict(),
    "first version: nobody stuck, zero by week 24": FIRST_VERSION,
    "nobody stuck, no deadline": dict(stall=None, give_up=False),
    "half as many stalls": dict(stall=(0.05, 0.05, 0.05, 0.125)),
    "twice as many stalls": dict(stall=(0.20, 0.20, 0.20, 0.50)),
    "never steps back": dict(back_p=0.0),
    "quicker steps, every 3 to 5 weeks": dict(weeks=(3, 5)),
    "slower steps, every 5 to 8 weeks": dict(weeks=(5, 8)),
    "salt bottles 20 -> 10 -> 5 -> 0, every 5 to 8 weeks": dict(weeks=(5, 8), ladder=SALT, stall=(0.10, 0.10, 0.25)),
}


class Ladder(ManualTaper):
    def __init__(self, ladder=None, **kw):
        super().__init__(**kw)
        if ladder:
            self.LADDER = ladder


def run(args):
    i, seed, name = args
    return dict(run_traditional(i, seed, schedule=Ladder, **VARIANTS[name]), variant=name)


if __name__ == "__main__":
    n = len(people())
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(run, [(i, s, v) for v in VARIANTS for i in range(n) for s in (0, 1, 2)], chunksize=2))
    wane = [r for r in wane_rows() if r["controller"] == "v4"]
    names = sorted({(r["person"], r["group"]) for r in wane})
    grp = np.array([g for _, g in names])
    w = per_person(wane, names, "v4", "free")
    rng = np.random.default_rng(0)
    out = {}
    for v in VARIANTS:
        rs = [r for r in rows if r["variant"] == v]
        t = per_person(rs, names, "traditional", "free")
        res = dict(share_ending_on_a_bottle=float(np.mean([r["final_u"] > 0 for r in rs])))
        for g in ("general", "difficult"):
            res[g] = dict(traditional_free=float(t[grp == g].mean()), wane_free=float(w[grp == g].mean()),
                          traditional_stuck=float(per_person(rs, names, "traditional", "stuck")[grp == g].mean()),
                          wane_minus_traditional=paired(w, t, grp, g, rng))
        out[v] = res
        gp, gd = res["general"], res["difficult"]
        d = gp["wane_minus_traditional"]
        print(f"{v:52s} on a bottle {res['share_ending_on_a_bottle']:.2f} | typical: free {gp['traditional_free']:.3f} "
              f"(stuck {gp['traditional_stuck']:.3f}) vs Wane {gp['wane_free']:.3f}, gain {100 * d['diff']:+.2f} "
              f"[{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}] | difficult gain {100 * gd['wane_minus_traditional']['diff']:+.2f}")
    (OUT / "traditional-taper-sensitivity.json").write_text(json.dumps(dict(label="SIMULATION ONLY", variants=out), indent=1) + "\n")
