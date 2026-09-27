"""Calibrate relapse model v2 on an abrupt, unaided quit. SIMULATION ONLY.

60 synthetic people (calibration population, seed 61, never used elsewhere) stop nicotine completely on taper day 0
(flat controller, level 0, no nicotine available). The physiology runs once; the lapse process is replayed for
every candidate (alpha_mean, kappa, lam_r). Targets, fit to the published population relapse curve (Herd, Borland
and Hyland 2009, h(t) = 0.40 / (t + 1)^1.05 per day; Garvey 1992, 62 % relapsed by 2 weeks):
    survival 0.36 at day 14, 0.28 at day 30, 0.16 at day 180
Fourth target: the craving tail. Among verified abstainers, urge strength at 26 weeks is about 0.36 of week 1
(1.0 vs 2.8 on a 0 to 5 scale, Ussher et al. 2013); here the ratio of mean craving x in days 176 to 182 to days 1
to 7, among people who have not relapsed.
Checks, not fitted: median lapses before relapse (Kirchner 2012: about 5); survival at 7 days (Hughes 2004: 24 to
51 %) and 90 days (Hughes 2004: 10 to 20 %).
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
CRAVING_RATIO = 0.36
CHECKS = (7, 90)
PEOPLE = population(60, seed=61)


class Abrupt(SlowSchedule):
    def level(self, taper_day, last_day_puffs):
        return 0.0


def physiology(i):
    return simulate(PEOPLE[i], Flat, Abrupt(), taper_days=182, follow_days=0, seed=6100 + i, burn_in_days=28,
                    stop_at_relapse=False)


def survival(results, params, n_mc=300):
    surv = {t: [] for t in list(TARGET) + list(CHECKS)}
    lap = []
    for i, r in enumerate(results):
        lg, av = daily_logits(r, PEOPLE[i], params, 6100 + i)
        rel, day, lapses = replay(lg, av, params, n_mc=n_mc, seed=i)
        for t in surv:
            surv[t].append(1 - np.mean(rel & (day < t)))
        if rel.any():
            lap.append(np.median(lapses[rel]))
    s = {t: float(np.mean(v)) for t, v in surv.items()}
    s["craving_ratio"] = craving_ratio(results, params)
    return s, float(np.median(lap)) if lap else float("nan")


def craving_ratio(results, params):
    from puffsim import WARMUP_DAYS
    early, late = [], []
    for r in results:
        d = r["days"][WARMUP_DAYS:]
        x = [e["mean_w"] + params.lam_r * e.get("mean_r", 0.0) for e in d]
        early.append(np.mean(x[0:7])); late.append(np.mean(x[175:182]))
    return float(np.mean(late) / max(np.mean(early), 1e-9))


def loss(s):
    lg = lambda p: math.log(max(p, 1e-4) / max(1 - p, 1e-4))
    return sum((lg(s[t]) - lg(TARGET[t])) ** 2 for t in TARGET) + (math.log(max(s["craving_ratio"], 1e-4) / CRAVING_RATIO)) ** 2


def score(args):
    params, results = args
    s, lap = survival(results, params)
    return loss(s), s, lap, params


if __name__ == "__main__":
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(physiology, range(len(PEOPLE))))
    base = RelapseParams()
    grid = [replace(base, alpha_mean=a, kappa=k, lam_r=l)
            for a, k, l in itertools.product(np.arange(-9.0, -1.9, 0.5), (1, 2, 3, 5, 8, 12, 18, 27), (0, 0.25, 0.5, 1, 2, 4))]
    with ProcessPoolExecutor() as ex:
        scored = sorted(ex.map(score, [(p, results) for p in grid], chunksize=8), key=lambda x: x[0])
    for l_, s, lap, p in scored[:8]:
        print(f"loss {l_:.3f}  S7 {s[7]:.2f} S14 {s[14]:.2f} S30 {s[30]:.2f} S90 {s[90]:.2f} S180 {s[180]:.2f} ratio {s['craving_ratio']:.2f}  lapses {lap:.1f}  "
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
    print(f"final: loss {loss(s_final):.3f}  S7 {s_final[7]:.3f} S14 {s_final[14]:.3f} S30 {s_final[30]:.3f} S90 {s_final[90]:.3f} "
          f"S180 {s_final[180]:.3f} craving ratio {s_final['craving_ratio']:.3f}  "
          f"lapses before relapse {lap_final:.1f}  alpha {p.alpha_mean:.2f} kappa {p.kappa:.2f} lam_r {p.lam_r:.2f}")
    save_calibration(p, dict(target=TARGET, craving_ratio_target=CRAVING_RATIO, checks=dict(hughes_2004_s7="0.24 to 0.51", hughes_2004_s90="0.10 to 0.20"),
                             fitted={str(k): v for k, v in s_final.items()}, loss=loss(s_final),
                             lapses_before_relapse=lap_final, people=len(PEOPLE), population_seed=61))
