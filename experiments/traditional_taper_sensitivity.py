"""How much does Wane's gain over the traditional taper depend on how the traditional taper is modelled? SIMULATION ONLY.

Same people, seeds and windows as experiments/traditional_taper_eval.py; only the hand-taper assumptions change.
Wane's runs are reused from experiments/relapse-v2-runs.csv.
Run: WANE_MAIN=/path/to/main python3 experiments/traditional_taper_sensitivity.py
"""
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from puff_nn import Flat                                                     # noqa: E402
from puffsim import ManualTaper, simulate                                    # noqa: E402
from relapse import evaluate                                                 # noqa: E402
from relapse_v2_eval import HARD, people                                     # noqa: E402

OUT = ROOT / "experiments"
SALT = (1.0, 0.5, 0.25, 0.0)
VARIANTS = {
    "never steps back": dict(back_p=0.0),
    "always steps back when puffing jumps": dict(back_p=1.0),
    "quicker steps, every 3 to 5 weeks": dict(weeks=(3, 5)),
    "slower steps, every 5 to 8 weeks": dict(weeks=(5, 8)),
    "salt bottles 20 -> 10 -> 5 -> 0, every 5 to 8 weeks": dict(weeks=(5, 8), ladder=SALT),
}


class Ladder(ManualTaper):
    def __init__(self, ladder=None, **kw):
        super().__init__(**kw)
        if ladder:
            self.LADDER = ladder


def run(args):
    i, seed, name = args
    p, group = people()[i]
    slow = Ladder(seed=1000 * seed + i + 555, **VARIANTS[name])
    r = simulate(p, Flat, slow, taper_days=168, follow_days=84, seed=1000 * seed + i,
                 physiology=HARD if group == "difficult" else None, stop_at_relapse=False, burn_in_days=28)
    ev = evaluate(r, p, seed=1000 * seed + i, n_mc=500)
    return dict(person=p.name, group=group, variant=name, seed=seed, expected_risk=ev["expected_risk"],
                zero_day=r["zero_day"])


if __name__ == "__main__":
    n = len(people())
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(run, [(i, s, v) for v in VARIANTS for i in range(n) for s in (0, 1, 2)], chunksize=2))
    with open(OUT / "relapse-v2-runs.csv") as fh:
        wane = [r for r in csv.DictReader(fh) if r["controller"] == "v4"]
    with open(OUT / "traditional-taper-runs.csv") as fh:
        base = [dict(r, variant="main assumptions") for r in csv.DictReader(fh)]
    rows = base + rows
    names = sorted({(r["person"], r["group"]) for r in wane})
    grp = np.array([g for _, g in names])
    per = lambda rs: np.array([np.mean([float(r["expected_risk"]) for r in rs if (r["person"], r["group"]) == nm]) for nm in names])
    w = per(wane)
    rng = np.random.default_rng(0)
    out = {}
    for v in ["main assumptions", *VARIANTS]:
        rs = [r for r in rows if r["variant"] == v]
        t = per(rs)
        res = dict(zero_week_mean=float(np.mean([float(r["zero_day"]) for r in rs]) / 7))
        for g in ("general", "difficult"):
            d = (w - t)[grp == g]
            bt = [rng.choice(d, len(d)).mean() for _ in range(4000)]
            res[g] = dict(traditional_still_off=float(1 - t[grp == g].mean()), wane_still_off=float(1 - w[grp == g].mean()),
                          wane_minus_traditional=float(d.mean()), lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)))
        out[v] = res
        print(f"{v:52s} zero wk {res['zero_week_mean']:5.1f}  general: still off {res['general']['traditional_still_off']:.3f} "
              f"vs Wane {res['general']['wane_still_off']:.3f}, relapse diff {res['general']['wane_minus_traditional']:+.4f} "
              f"[{res['general']['lo']:+.4f}, {res['general']['hi']:+.4f}]   difficult diff {res['difficult']['wane_minus_traditional']:+.4f}")
    (OUT / "traditional-taper-sensitivity.json").write_text(json.dumps(dict(label="SIMULATION ONLY", variants=out), indent=1) + "\n")
