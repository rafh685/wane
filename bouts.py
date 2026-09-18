"""Bout-level day generator: puffs come in bouts, bouts follow the person's routine.

Built on Ilian's day generator (18 Sept 2026): his intra-bout structure (3-15 puffs, 16-69 s apart, shorter with
craving) and per-puff duration / flow / volume = duration x flow. Changed so it plugs into profiles.Vaper:
  - craving, wake, bed, puffs_per_day and an rng are INPUTS, nothing is drawn inside
  - bout start times are drawn from the person's hourly routine (work bans, evening-only), not uniformly
  - night bouts exist: withdrawal wakes people up, more so as craving rises
  - hours are floats from midnight, no datetime
Output per day: sorted list of puffs, each (t_hours, duration_s, flow_ml_s).
"""
import numpy as np

PUFF_DUR_S, FLOW_ML_S = 3.4, 20.0


def puff(rng, craving, dur_mult=1.0, flow_mult=1.0):
    dur = float(np.clip(rng.lognormal(np.log(PUFF_DUR_S * (1 + 0.03 * (craving - 2)) * dur_mult), 0.25), 1.2, 7.0))
    flow = float(np.clip(rng.lognormal(np.log(FLOW_ML_S * (1 + 0.02 * (craving - 2)) * flow_mult), 0.25), 8.0, 40.0))
    return dur, flow


def bout(rng, start_h, n, craving, dur_mult=1.0, flow_mult=1.0):
    """Ilian's bout: n puffs, gaps 16..(69 - 2.5*craving) s, first at start_h."""
    t = start_h; out = []
    hi = max(17, int(69 - craving * 2.5))
    for i in range(n):
        if i:
            t += rng.integers(16, hi + 1) / 3600
        d, f = puff(rng, craving, dur_mult, flow_mult)
        out.append((t, d, f))
    return out


def day(rng, craving, wake, bed, puffs_target, routine, cue=None, dur_mult=1.0, flow_mult=1.0):
    """One day of puffs in bouts.
    routine: 24 weights, where in the day this person vapes (0 = never at that hour)
    cue: 24 boosts to craving by hour (coffee, alcohol...), optional
    """
    cue = np.zeros(24) if cue is None else cue
    n_total = max(3, int(rng.normal(puffs_target, puffs_target * 0.08)))
    n_bouts = int(np.clip(rng.normal(np.clip(n_total / 9, 8, 30), 3), 6, 40))
    # split puffs over bouts, 3 minimum each
    sizes = rng.multinomial(max(0, n_total - 3 * n_bouts), np.ones(n_bouts) / n_bouts) + 3
    # candidate start hours: sample from the routine restricted to waking hours, with cue boost
    hours = np.arange(24)
    awake = np.array([(wake <= h < bed) or (bed > 24 and h < bed - 24) for h in hours], dtype=float)
    w = routine * awake * (1 + 0.06 * cue)
    if w.sum() <= 0:
        w = awake
    starts = np.sort(rng.choice(hours, size=n_bouts, p=w / w.sum()) + rng.uniform(0, 1, n_bouts))
    # first bout: time to first puff after waking, Fagerström logic, shorter with craving
    ttfc_h = max(1.0, rng.exponential(30 * np.exp(-0.25 * (craving - 2)))) / 60
    starts[0] = wake + ttfc_h
    starts = np.sort(starts)
    # enforce sequential bouts with a minimum gap of 10 minutes
    puffs = []; last_end = -1
    for s, n in zip(starts, sizes):
        s = max(s, last_end + 10 / 60)
        c = float(np.clip(craving + cue[int(s) % 24] * 0.5 + rng.uniform(-0.5, 0.5), 0, 10))
        b = bout(rng, s, int(n), c, dur_mult, flow_mult)
        puffs.extend(b); last_end = b[-1][0]
    # night bouts: withdrawal waking, inside the sleep window, more as craving rises
    n_night = rng.poisson(0.15 * (1 + max(0.0, craving - 4)) * puffs_target / 100)
    sleep_len = (wake + 24 - bed) if bed > wake else (wake - bed)
    for _ in range(n_night):
        s = (bed + rng.uniform(0.5, max(0.6, sleep_len - 0.5))) % 24
        puffs.extend(bout(rng, s, int(rng.integers(2, 5)), craving, dur_mult, flow_mult))
    puffs.sort()
    return puffs
