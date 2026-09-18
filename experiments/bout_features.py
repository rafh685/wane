"""Does bout structure make the hard compensator visible sooner than daily counts do?

Puffs in bouts (bouts.py, built on Ilian's generator) give features daily counts cannot: bouts per day, puffs per
bout, the gap between puffs inside a bout, and per-puff flow (pressure sensor). For each feature we measure the
two-week response after the first cuts, Lucía (elasticity 0.7, sensitivity 7) against Ana (0.15, 2.5), 20 seeds
each, and report how separable the two are (AUC: 0.5 = indistinguishable, 1.0 = perfectly separable).
Run: .venv/bin/python experiments/bout_features.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, profiles
from profiles import PROFILES, Vaper
profiles.BOUT_MODE = True

BOUT_GAP_H = 10 / 60


def bout_stats(times, flows):
    """Cluster puffs into bouts (gap > 10 min starts a new one)."""
    if len(times) < 2:
        return dict(bouts=len(times), per_bout=len(times), ipi=np.nan, flow=np.mean(flows) if flows else np.nan)
    t = np.array(times); gaps = np.diff(t)
    breaks = np.where(gaps > BOUT_GAP_H)[0]
    n_bouts = len(breaks) + 1
    intra = gaps[gaps <= BOUT_GAP_H] * 3600
    return dict(bouts=n_bouts, per_bout=len(t) / n_bouts, ipi=float(np.median(intra)) if len(intra) else np.nan, flow=float(np.mean(flows)))


def week_feats(rows):
    f = {}
    f["puffs"] = np.mean([r["puffs"] for r in rows])
    f["puff_dur"] = np.mean([r["puff_dur"] for r in rows])
    f["intake"] = f["puffs"] * f["puff_dur"]
    f["night_share"] = np.mean([r["night_puffs"] for r in rows]) / max(f["puffs"], 1)
    f["ttfc"] = np.mean([r["ttfc_min"] for r in rows])
    bs = [bout_stats(r["times"], r.get("flows", [20.0])) for r in rows]
    for k in ("bouts", "per_bout", "ipi", "flow"):
        f[k] = np.nanmean([b[k] for b in bs])
    f["volume"] = f["puffs"] * f["puff_dur"] * f["flow"]
    return f


def response(profile, seed, cuts=(20.0, 18.8, 16.5, 14.5)):
    """Week 1 at 20 mg is the baseline; weeks 3-4 after two full cuts are the response. Ratio minus 1."""
    v = Vaper(profile, np.random.default_rng(seed))
    weeks = [[v.day(d) for _ in range(7)] for d in cuts]
    base, resp = week_feats(weeks[0]), week_feats(weeks[2] + weeks[3])
    return {k: resp[k] / base[k] - 1 if base[k] else np.nan for k in base}


def auc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b]))


if __name__ == "__main__":
    luc = next(p for p in PROFILES if p.name == "Lucía"); ana = next(p for p in PROFILES if p.name == "Ana")
    N = 20
    L = [response(luc, s) for s in range(N)]; A = [response(ana, s) for s in range(N)]
    print(f"two-week response after the first cuts, Lucía vs Ana, {N} seeds each (AUC 0.5 = same, 1.0 = separable)\n")
    print(f"{'feature':<12}{'Lucía median':>14}{'Ana median':>12}{'AUC':>7}")
    for k in ("puffs", "puff_dur", "intake", "flow", "volume", "bouts", "per_bout", "ipi", "night_share", "ttfc"):
        l = [r[k] for r in L]; a = [r[k] for r in A]
        sign = -1 if k in ("ipi", "ttfc") else 1          # shorter gaps / earlier first puff = more dependent
        print(f"{k:<12}{np.nanmedian(l):>+14.2f}{np.nanmedian(a):>+12.2f}{auc(sign*np.array(l), sign*np.array(a)):>7.2f}")
