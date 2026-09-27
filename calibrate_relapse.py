"""Calibrate relapse model v2 on an abrupt, unaided quit. SIMULATION ONLY.

60 synthetic people (calibration population, seed 61, never used elsewhere) stop nicotine completely on taper day 0
(flat controller, level 0, no nicotine available). The physiology runs once; the lapse process is replayed for
every candidate (alpha_mean, kappa, lam_r). Targets, fit to the published population relapse curve (Herd, Borland
and Hyland 2009, h(t) = 0.40 / (t + 1)^1.05 per day; Garvey 1992, 62 % relapsed by 2 weeks):
    survival 0.36 at day 14, 0.28 at day 30, 0.16 at day 180
Check, not fitted: median lapses before relapse (Kirchner 2012: about 5, in a nicotine-patch trial).
Run: python3 calibrate_relapse.py   (writes relapse_calibration.json)
"""
import itertools
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from population import population
from puff_nn import Flat
from puffsim import SlowSchedule, simulate
from relapse import RelapseParams, daily_logits, replay, save_calibration

TARGET = {14: 0.36, 30: 0.28, 180: 0.16}
PEOPLE = population(60, seed=61)


class Abrupt(SlowSchedule):
    def level(self, taper_day, last_day_puffs):
        return 0.0


def physiology(i):
    return simulate(PEOPLE[i], Flat, Abrupt(), taper_days=182, follow_days=0, seed=6100 + i, burn_in_days=28,
                    stop_at_relapse=False)


def survival(results, params, n_mc=300):
    surv = {t: [] for t in TARGET}
    lap = []
    for i, r in enumerate(results):
        lg, av = daily_logits(r, PEOPLE[i], params, 6100 + i)
        rel, day, lapses = replay(lg, av, params, n_mc=n_mc, seed=i)
        for t in TARGET:
            surv[t].append(1 - np.mean(rel & (day < t)))
        if rel.any():
            lap.append(np.median(lapses[rel]))
    return {t: float(np.mean(v)) for t, v in surv.items()}, float(np.median(lap)) if lap else float("nan")


def loss(s):
    lg = lambda p: math.log(max(p, 1e-4) / max(1 - p, 1e-4))
    return sum((lg(s[t]) - lg(TARGET[t])) ** 2 for t in TARGET)


def score(args):
    params, results = args
    s, lap = survival(results, params)
    return loss(s), s, lap, params


if __name__ == "__main__":
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(physiology, range(len(PEOPLE))))
    base = RelapseParams()
    grid = [replace(base, alpha_mean=a, kappa=k, lam_r=l)
            for a, k, l in itertools.product(np.arange(-9.0, -1.9, 0.5), (1, 2, 3, 5, 8, 12), (0, 0.5, 1, 2, 4))]
    with ProcessPoolExecutor() as ex:
        scored = sorted(ex.map(score, [(p, results) for p in grid], chunksize=8), key=lambda x: x[0])
    for l_, s, lap, p in scored[:8]:
        print(f"loss {l_:.3f}  S14 {s[14]:.2f} S30 {s[30]:.2f} S180 {s[180]:.2f}  lapses before relapse {lap:.1f}  "
              f"alpha {p.alpha_mean:.1f} kappa {p.kappa} lam_r {p.lam_r}")
    best = scored[0][3]
    fine = [replace(best, alpha_mean=a, kappa=k, lam_r=l)
            for a, k, l in itertools.product(best.alpha_mean + np.arange(-0.4, 0.41, 0.2),
                                              best.kappa * np.array([0.7, 0.85, 1, 1.2, 1.4]),
                                              sorted({max(0.0, best.lam_r + d) for d in (-0.5, -0.25, 0, 0.25, 0.5)}))]
    with ProcessPoolExecutor() as ex:
        scored2 = sorted(ex.map(score, [(p, results) for p in fine], chunksize=8), key=lambda x: x[0])
    l_, s, lap, p = scored2[0]
    s_final, lap_final = survival(results, p, n_mc=1000)
    print(f"final: loss {loss(s_final):.3f}  S14 {s_final[14]:.3f} S30 {s_final[30]:.3f} S180 {s_final[180]:.3f}  "
          f"lapses before relapse {lap_final:.1f}  alpha {p.alpha_mean:.2f} kappa {p.kappa:.2f} lam_r {p.lam_r:.2f}")
    save_calibration(p, dict(target=TARGET, fitted={str(k): v for k, v in s_final.items()}, loss=loss(s_final),
                             lapses_before_relapse=lap_final, people=len(PEOPLE), population_seed=61))
