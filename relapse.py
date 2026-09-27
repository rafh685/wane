"""Relapse model v2, rebuilt from the research report (docs/research/nicotine_dependence_parameters.md). SIMULATION ONLY.

Why: in v1 (inside puffsim.simulate) the daily relapse hazard was a near step (+x7.4 odds per 0.03 of excess
withdrawal) above a threshold of 0.08 to 0.17, while ordinary day-to-day withdrawal swings have an SD of 0.12. So
relapse was mostly random bad days, and no plan could change it (docs/interview_priors.md).

v2 works on the daily records of a finished run (puffsim.simulate with burn_in_days=28), so it can be calibrated
without re-simulating:
  craving   x = awake withdrawal W (fast clock, days) + lam_r x slow craving R (receptor clock, weeks)
  standard  z = the 3-day mean of x against this person's own baseline, per weekday, in units of kappa x the SD of
            their own 3-day mean at baseline. A normal bad day is about z = 1, not a cliff.
  lapse     daily probability, logit = alpha_i + beta x z + cue terms + contagion x recent lapses
            beta 0.64 per SD: pooled nicotine odds ratio 1.87 per standardised unit of craving (Vafaie & Kober 2022,
            237 studies, 51,788 people). No absolute threshold (Allen 2008).
            alcohol days x3 (heavy drinkers about x4 lapse risk on drinking days), stress days x1.5 (assumption,
            crude OR 2.0 in Businelle 2016)
            alpha_i: person's baseline lapse propensity, higher with a lower relapse threshold, random spread
  contagion each lapse raises the next day's odds, decaying with a 5-day half-life; more so without nicotine
            available (abrupt quit, or at zero dose) than during a taper (nicotine replacement slows lapse to
            relapse: Shiffman 2006, Kirchner 2012). Sizes are ASSUMPTIONS.
  relapse   5 lapse days within 14 days: daily use resumes after a median of about 5 lapses over about 13 days
            (Kirchner, Shiffman and Wileyto 2012)
  risk      the lapse process is replayed n_mc times on the same physiology: expected risk = share relapsed
Calibrated jointly (calibrate_relapse.py, 27 Sept evening): alpha_mean, kappa, lam_r, taper_shift and
contagion_avail, so that
  an abrupt unaided quit reproduces the published relapse curve: survival 0.36 at 14 days, 0.28 at 30, 0.16 at 180
  (fit to Herd, Borland and Hyland 2009; Garvey 1992), and the craving tail of Ussher 2013 (0.36 at 26 weeks), and
  a plain 12 %/week taper matches the trials: about 0.22 still off 12 weeks after reaching zero (gradual reduction
  no better than abrupt quitting with support, RR 1.01, Lindson 2019; 15.5 % abstinent at 6 months in the gradual arm
  of Lindson-Hawley 2016; 13 to 20 % in vaping-cessation trials), and about 0.57 still on plan at week 20 (gradual
  arm of Hatsukami 2018, compliance counting dropouts).
Checks, not fitted: Hughes 2004 ranges at 7 and 90 days, lapses before relapse (Kirchner 2012), and PATH's return
rate among former vapers.
"""
import json
import math
import pathlib
import zlib
from dataclasses import asdict, dataclass

import numpy as np

from puffsim import WARMUP_DAYS

CALIBRATION_FILE = pathlib.Path(__file__).with_name("relapse_calibration.json")


@dataclass
class RelapseParams:
    beta: float = 0.64
    alpha_mean: float = -6.0
    alpha_sd: float = 0.8                 # between-person spread of baseline propensity (assumption)
    alpha_thr: float = 0.5                # logit per SD of lower relapse threshold (the profile trait)
    kappa: float = 3.0
    lam_r: float = 1.0
    alcohol: float = math.log(3.0)
    stress: float = math.log(1.5)
    contagion_avail: float = 0.5
    taper_shift: float = 0.0              # logit added on days nicotine is still in the plan (taper phase, calibrated)
    contagion_none: float = 1.2
    contagion_half_life: float = 5.0
    relapse_lapses: int = 5
    relapse_window: int = 14

    @classmethod
    def calibrated(cls):
        if CALIBRATION_FILE.exists():
            return cls(**json.loads(CALIBRATION_FILE.read_text())["params"])
        return cls()


def person_alpha(profile, params, seed=None):
    """A person's baseline tendency to slip: a fixed personal trait. Fixed 27 Sept: it used to be drawn from the
    run's seed, so a batch run with one shared seed gave everyone the same random offset."""
    z_thr = (profile.relapse_threshold - 7.0) / 0.866            # threshold ~ U(5.5, 8.5) in population.py
    rng = np.random.default_rng(zlib.crc32(profile.name.encode()) + 31337)
    return params.alpha_mean - params.alpha_thr * z_thr + params.alpha_sd * rng.normal()


def daily_logits(result, profile, params, seed):
    """Per taper/follow-up day: base logit (without contagion) and whether nicotine was available."""
    days = result["days"]
    x = np.array([d["mean_w"] + params.lam_r * d.get("mean_r", 0.0) for d in days])
    dow = np.array([d["day"] % 7 for d in days])
    base = [i for i in range(7, WARMUP_DAYS) if i < len(days)]
    mu = {w: float(np.mean([x[i] for i in base if dow[i] == w])) for w in range(7)}
    resid = np.array([x[i] - mu[dow[i]] for i in base])
    sd_day = float(np.std(resid) * math.sqrt(2)) or 1e-3             # two samples per weekday: undo the shrinkage
    z = np.array([(x[i] - mu[dow[i]]) / sd_day for i in range(len(days))])
    z3 = {i: float(np.mean(z[max(WARMUP_DAYS, i - 2):i + 1]) * math.sqrt(3)) for i in range(WARMUP_DAYS, len(days))}
    alpha = person_alpha(profile, params, seed)
    out_logit, avail = [], []
    for i in range(WARMUP_DAYS, len(days)):
        d = days[i]
        lg = alpha + params.beta * z3[i] / params.kappa
        lg += params.alcohol * d.get("alcohol", False) + params.stress * d.get("stress", False)
        if d["u"] > 0:
            lg += params.taper_shift
        out_logit.append(lg)
        avail.append(d["u"] > 0)
    return np.array(out_logit), np.array(avail, dtype=bool)


def replay(logits, avail, params, n_mc=1000, seed=0):
    """Replay the lapse process n_mc times. Returns (relapsed mask, relapse day or -1, lapses per replay)."""
    rng = np.random.default_rng(seed)
    n = len(logits)
    decay = 0.5 ** (1 / params.contagion_half_life)
    c = np.zeros(n_mc)
    window = np.zeros((n_mc, params.relapse_window), dtype=np.int8)
    relapsed = np.zeros(n_mc, dtype=bool)
    day = np.full(n_mc, -1)
    lapses = np.zeros(n_mc, dtype=int)
    for t in range(n):
        live = ~relapsed
        lg = logits[t] + (params.contagion_avail if avail[t] else params.contagion_none) * c
        p = 1 / (1 + np.exp(-lg))
        lapse = (rng.random(n_mc) < p) & live
        lapses += lapse
        c = c * decay + lapse
        window[:, t % params.relapse_window] = lapse
        new = live & (window.sum(axis=1) >= params.relapse_lapses)
        relapsed |= new
        day[new] = t
    return relapsed, day, lapses


def evaluate(result, profile, params=None, seed=0, n_mc=1000):
    """Expected relapse risk under model v2 for one finished run, plus survival at a few days."""
    params = params or RelapseParams.calibrated()
    logits, avail = daily_logits(result, profile, params, seed)
    relapsed, day, lapses = replay(logits, avail, params, n_mc=n_mc, seed=seed)
    out = dict(expected_risk=float(relapsed.mean()))
    for t in (14, 30, 90, 140, 180):
        out[f"survival_{t}"] = float(1 - np.mean(relapsed & (day < t))) if t <= len(logits) else float("nan")
    zero = next((j for j, d in enumerate(result["days"][WARMUP_DAYS:]) if d["u"] == 0), None)
    out["relapse_before_zero"] = float(np.mean(relapsed & (day < zero))) if zero is not None else float(relapsed.mean())
    out["reached_zero_then_relapsed"] = (float(np.mean((relapsed & (day >= zero))[~(relapsed & (day < zero))]))
                                         if zero is not None and (~(relapsed & (day < zero))).any() else float("nan"))
    out["median_relapse_day"] = float(np.median(day[relapsed])) if relapsed.any() else float("nan")
    out["lapses_before_relapse"] = float(np.median(lapses[relapsed])) if relapsed.any() else float("nan")
    return out


def save_calibration(params, report):
    CALIBRATION_FILE.write_text(json.dumps(dict(params=asdict(params), report=report), indent=1) + "\n")
