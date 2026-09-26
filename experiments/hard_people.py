"""Difficult people with irregular habits, for the fast layer. SIMULATION ONLY.

Rafael (27 Sept 2026): tune and judge the engine on difficult people with irregular habits, like Lucía, not on
people with a stable routine. So:
  - habits: every person's daily amount swings from day to day (their profile noise, x1.5 here), and compensation
    splits 57 % into longer draws, 43 % into more puffs (the LSBU split already in profiles.py)
  - pool: 200 synthetic people per set; "difficult" = the top quarter by expected relapse risk on the flat taper,
    measured with selection seeds (900, 901) that are never used for evaluation, so picking people who were
    merely unlucky cannot flatter the other controllers
  - development set: pool seed 41 (used to choose fixes and settings)
  - test set: pool seed 51 plus Lucía and Karim (used once, at the end)
Run once to write experiments/hard-people.json: python3 experiments/hard_people.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from population import sample_profile                  # noqa: E402
from profiles import PROFILES                           # noqa: E402
from puff_nn import Flat                                # noqa: E402
from puffsim import SlowSchedule, simulate              # noqa: E402

HARD = dict(dur_share=0.57, irregular=True)
POOL_N, SELECT_SEEDS, TOP = 200, (900, 901), 0.25
FILE = ROOT / "experiments" / "hard-people.json"


def pool(seed):
    rng = np.random.default_rng(seed)
    people = []
    for i in range(POOL_N):
        p = sample_profile(rng, f"H{seed}-{i:03d}")
        people.append(replace(p, noise=p.noise * 1.5))
    return people


def named():
    return [replace(p, noise=p.noise * 1.5) for p in PROFILES if p.name in ("Lucía", "Karim")]


def flat_risk(args):
    seed, i = args
    p = pool(seed)[i]
    return i, float(np.mean([simulate(p, Flat, SlowSchedule(), seed=s, physiology=HARD, stop_at_relapse=False)["expected_risk"]
                             for s in SELECT_SEEDS]))


def load(which):
    """'dev' or 'test' -> list of Profiles."""
    sel = json.loads(FILE.read_text())[which]
    people = [pool(sel["pool_seed"])[i] for i in sel["index"]]
    return people + (named() if which == "test" else [])


if __name__ == "__main__":
    out = {}
    with ProcessPoolExecutor() as ex:
        for which, seed in (("dev", 41), ("test", 51)):
            risks = dict(ex.map(flat_risk, [(seed, i) for i in range(POOL_N)]))
            cut = float(np.quantile(list(risks.values()), 1 - TOP))
            idx = sorted(i for i, r in risks.items() if r >= cut)
            out[which] = dict(pool_seed=seed, index=idx, threshold=cut,
                              pool_median_risk=float(np.median(list(risks.values()))),
                              selected_mean_risk=float(np.mean([risks[i] for i in idx])))
            print(f"{which}: {len(idx)} of {POOL_N} selected, flat risk >= {cut:.3f} "
                  f"(pool median {out[which]['pool_median_risk']:.3f}, selected mean {out[which]['selected_mean_risk']:.3f})")
    FILE.write_text(json.dumps(out, indent=1) + "\n")
