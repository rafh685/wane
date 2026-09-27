"""Calibrate relapse model v2 jointly on an abrupt quit and a plain taper. SIMULATION ONLY.

Calibration population: 60 synthetic people (seed 61, never used elsewhere), realistic vapers (puffsim REALISM).
Two physiology runs per person, simulated once:
  abrupt   nicotine stops on day 0 (flat controller, level 0), 26 weeks
  taper    weaker bottle, 12 %/week to zero (23 weeks), then 12 weeks at zero
The lapse process (relapse.py) is then replayed for every candidate setting, which is cheap.
Targets (logit-scale errors; weights in brackets):
  abrupt   survival 0.36 at 14 days, 0.28 at 30, 0.16 at 180 (Herd 2009 fit; Garvey 1992)            [1 each]
  abrupt   craving at 26 weeks / week 1 = 0.36 (Ussher 2013)                                        [1]
  taper    still off 12 weeks after reaching zero: 0.22 (Lindson 2019 RR 1.01; Lindson-Hawley 2016 15.5 %
           at 6 months; vaping-cessation trials 13 to 20 %)                                           [2]
  taper    still on plan at week 20: 0.57 (Hatsukami 2018 gradual arm, compliance counting dropouts)   [1]
Checks, not fitted: survival at 7 and 90 days (Hughes 2004: 0.24 to 0.51, 0.10 to 0.20); median lapses before
relapse (Kirchner 2012: about 5); share of those reaching zero who relapse within 12 weeks (PATH former vapers
back within a year: 38 % if quit under a month, 31 % if quit 1 to 6 months).
Search: 1,500 random settings, then 300 around the best 5. Writes relapse_calibration.json.
Run: python3 calibrate_relapse.py
"""
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np

from population import population
from puff_nn import Flat
from puffsim import WARMUP_DAYS, SlowSchedule, simulate
from relapse import RelapseParams, daily_logits, replay, save_calibration

PEOPLE = population(60, seed=61)
ABRUPT_T = {14: 0.36, 30: 0.28, 180: 0.16}
CRAVING_RATIO = 0.36
TAPER_SUCCESS, TAPER_WEEK20 = 0.22, 0.57


class Abrupt(SlowSchedule):
    def level(self, taper_day, last_day_puffs):
        return 0.0


def physiology(args):
    i, kind = args
    slow = Abrupt() if kind == "abrupt" else SlowSchedule()
    days = 182 if kind == "abrupt" else 168
    return simulate(PEOPLE[i], Flat, slow, taper_days=days, follow_days=0 if kind == "abrupt" else 84,
                    seed=6100 + i, burn_in_days=28, stop_at_relapse=False)


def craving_ratio(results, params):
    early, late = [], []
    for r in results:
        d = r["days"][WARMUP_DAYS:]
        x = [e["mean_w"] + params.lam_r * e.get("mean_r", 0.0) for e in d]
        early.append(np.mean(x[0:7])); late.append(np.mean(x[175:182]))
    return float(np.mean(late) / max(np.mean(early), 1e-9))


def measure(params, abrupt, taper, n_mc=150):
    s = {}
    surv = {t: [] for t in (7, 14, 30, 90, 180)}
    lap = []
    for i, r in enumerate(abrupt):
        lg, av = daily_logits(r, PEOPLE[i], params, 6100 + i)
        rel, day, lapses = replay(lg, av, params, n_mc=n_mc, seed=i)
        for t in surv:
            surv[t].append(1 - np.mean(rel & (day < t)))
        if rel.any():
            lap.append(np.median(lapses[rel]))
    s.update({f"abrupt_S{t}": float(np.mean(v)) for t, v in surv.items()})
    s["craving_ratio"] = craving_ratio(abrupt, params)
    s["lapses_before_relapse"] = float(np.median(lap)) if lap else float("nan")
    succ, wk20, post = [], [], []
    for i, r in enumerate(taper):
        lg, av = daily_logits(r, PEOPLE[i], params, 6100 + i)
        rel, day, _ = replay(lg, av, params, n_mc=n_mc, seed=100 + i)
        zero = next(j for j, d in enumerate(r["days"][WARMUP_DAYS:]) if d["u"] == 0)
        succ.append(1 - rel.mean())
        wk20.append(1 - np.mean(rel & (day < 140)))
        at_zero = ~(rel & (day < zero))
        post.append(np.mean((rel & (day >= zero))[at_zero]) if at_zero.any() else np.nan)
    s["taper_success"] = float(np.mean(succ))
    s["taper_week20"] = float(np.mean(wk20))
    s["post_zero_relapse_12wk"] = float(np.nanmean(post))
    return s


def loss(s):
    lg = lambda p: math.log(max(p, 1e-4) / max(1 - p, 1e-4))
    L = sum((lg(s[f"abrupt_S{t}"]) - lg(v)) ** 2 for t, v in ABRUPT_T.items())
    L += math.log(max(s["craving_ratio"], 1e-4) / CRAVING_RATIO) ** 2
    L += 2 * (lg(s["taper_success"]) - lg(TAPER_SUCCESS)) ** 2
    L += (lg(s["taper_week20"]) - lg(TAPER_WEEK20)) ** 2
    return L


def score(args):
    params, abrupt, taper = args
    s = measure(params, abrupt, taper)
    return loss(s), s, params


def draw(rng, around=None):
    if around is None:
        return replace(RelapseParams(), alpha_mean=float(rng.uniform(-7, -2)), kappa=float(np.exp(rng.uniform(np.log(1.5), np.log(40)))),
                       lam_r=float(rng.uniform(0, 2)), taper_shift=float(rng.uniform(-2.5, 1.5)),
                       contagion_avail=float(rng.uniform(0.1, 1.5)))
    b = around
    return replace(b, alpha_mean=b.alpha_mean + rng.normal(0, 0.3), kappa=float(b.kappa * np.exp(rng.normal(0, 0.2))),
                   lam_r=float(max(0, b.lam_r + rng.normal(0, 0.15))), taper_shift=b.taper_shift + rng.normal(0, 0.25),
                   contagion_avail=float(np.clip(b.contagion_avail + rng.normal(0, 0.15), 0.0, 2.0)))


if __name__ == "__main__":
    with ProcessPoolExecutor() as ex:
        abrupt = list(ex.map(physiology, [(i, "abrupt") for i in range(len(PEOPLE))]))
        taper = list(ex.map(physiology, [(i, "taper") for i in range(len(PEOPLE))]))
        rng = np.random.default_rng(2027)
        cands = [draw(rng) for _ in range(1500)]
        scored = sorted(ex.map(score, [(c, abrupt, taper) for c in cands], chunksize=10), key=lambda x: x[0])
        top = [x[2] for x in scored[:5]]
        cands2 = [draw(rng, around=top[j % 5]) for j in range(300)]
        scored2 = sorted(ex.map(score, [(c, abrupt, taper) for c in cands2], chunksize=10), key=lambda x: x[0])
    best = min(scored[:1] + scored2[:1], key=lambda x: x[0])[2]
    for l_, s, p in (scored[:3] + scored2[:3]):
        print(f"loss {l_:.3f}  abrupt S14 {s['abrupt_S14']:.2f} S30 {s['abrupt_S30']:.2f} S180 {s['abrupt_S180']:.2f} ratio {s['craving_ratio']:.2f}"
              f"  taper success {s['taper_success']:.2f} week20 {s['taper_week20']:.2f}  | alpha {p.alpha_mean:.2f} kappa {p.kappa:.1f}"
              f" lam_r {p.lam_r:.2f} shift {p.taper_shift:.2f} c_avail {p.contagion_avail:.2f}")
    final = measure(best, abrupt, taper, n_mc=1000)
    print("final:", {k: round(v, 3) for k, v in final.items()}, "loss", round(loss(final), 3))
    print("params:", {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(best).items()})
    save_calibration(best, dict(targets=dict(abrupt=ABRUPT_T, craving_ratio=CRAVING_RATIO, taper_success=TAPER_SUCCESS,
                                             taper_week20=TAPER_WEEK20),
                                checks=dict(hughes_2004_S7="0.24 to 0.51", hughes_2004_S90="0.10 to 0.20",
                                            kirchner_2012_lapses="about 5",
                                            path_former_vapers_back_within_year="0.38 if quit < 1 month, 0.31 if 1 to 6 months"),
                                fitted=final, loss=loss(final), people=len(PEOPLE), population_seed=61, realism=True))
