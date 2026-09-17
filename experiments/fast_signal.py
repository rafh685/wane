"""Experiment: does a 2-day "fast" signal let daily decisions beat weekly ones, and can it tell
withdrawal from a person's normal weekend?

Variants of the adaptive engine, all at 12 %/week, decisions daily:
  A  Wane as committed                  (7-day features only)
  B  + fast hold vs last 5 days         (punishing: a fast hold also lowers the personal rate)
  D  + fast hold vs OWN weekday baseline (punishing)
  E  D + alarm scaled to the person's own noise (2 sigma) + punishment once per episode
Run: .venv/bin/python experiments/fast_signal.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from simulate import run_one
from profiles import PROFILES
from engine import FixedTaper, AdaptiveTaper

FAST_TREND, FAST_NIGHT = 0.15, 0.03


class FastVsRecent(AdaptiveTaper):
    name = "B fast vs last 5 days"
    def alarm(self, window):
        if len(window) < 7:
            return False
        p = np.array([r["puffs"] for r in window]); n = np.array([r["night_puffs"] for r in window])
        trend = p[-2:].mean() / max(p[:-2].mean(), 1) - 1
        night = n[-2:].mean() / max(p[-2:].mean(), 1) - n[:-2].mean() / max(p[:-2].mean(), 1)
        return trend > FAST_TREND or night > FAST_NIGHT
    def next_dose(self, dose, window, prev_window):
        if self.alarm(window):
            self.personal_cut = max(self.min_cut, self.personal_cut * self.down)
            return dose
        return super().next_dose(dose, window, prev_window)


class FastVsBaseline(FastVsRecent):
    """The engine keeps a per-weekday baseline of this person's puffs and night share (exponential moving average),
    learned from the days it has seen. The alarm compares the last 2 days to THEIR OWN weekday baseline."""
    name = "D fast vs own weekday baseline"
    ALPHA = 0.3                      # baseline update speed
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.base_puffs = {}         # dow -> puffs
        self.base_night = {}         # dow -> night share
        self.seen = 0                # last day index folded into the baseline
    def update_baseline(self, window):
        for r in window:
            if r["day"] <= self.seen:
                continue
            d = r["dow"]; ns = r["night_puffs"] / max(r["puffs"], 1)
            self.base_puffs[d] = r["puffs"] if d not in self.base_puffs else (1 - self.ALPHA) * self.base_puffs[d] + self.ALPHA * r["puffs"]
            self.base_night[d] = ns if d not in self.base_night else (1 - self.ALPHA) * self.base_night[d] + self.ALPHA * ns
            self.seen = r["day"]
    def alarm(self, window):
        last2 = window[-2:]
        if any(r["dow"] not in self.base_puffs for r in last2):
            return False                                   # no baseline for that weekday yet
        trend = np.mean([r["puffs"] / max(self.base_puffs[r["dow"]], 1) for r in last2]) - 1
        night = np.mean([r["night_puffs"] / max(r["puffs"], 1) - self.base_night[r["dow"]] for r in last2])
        return trend > FAST_TREND or night > FAST_NIGHT
    def next_dose(self, dose, window, prev_window):
        # decide on the days already in the baseline, THEN fold today's days in (otherwise today is its own baseline)
        fired = self.alarm(window)
        self.update_baseline(window)
        if fired:
            self.personal_cut = max(self.min_cut, self.personal_cut * self.down)
            return dose
        return AdaptiveTaper.next_dose(self, dose, window, prev_window)


class FastSigma(FastVsBaseline):
    """E: two fixes on D.
    1. The alarm is scaled to the person's own noise: it fires when the last 2 days sit more than Z standard
       deviations above this person's baseline, using the spread of THEIR residuals (puffs / baseline - 1).
    2. Punishment per episode, not per day: one alarm slows the rate once and cannot fire again until the
       person has had a clean day. The hold itself still happens on every alarm day."""
    name = "E sigma alarm, per-episode"
    Z = 2.0
    MIN_SIGMA = 0.08                 # floor so a very regular person still gets a usable threshold
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.resid = []              # history of (puffs / baseline - 1) on seen days
        self.in_episode = False
    def update_baseline(self, window):
        for r in window:
            if r["day"] <= self.seen:
                continue
            d = r["dow"]
            if d in self.base_puffs:
                self.resid.append(r["puffs"] / max(self.base_puffs[d], 1) - 1)
        super().update_baseline(window)
    def sigma(self):
        return max(self.MIN_SIGMA, float(np.std(self.resid))) if len(self.resid) >= 5 else None
    def alarm(self, window):
        last2 = window[-2:]
        sg = self.sigma()
        if sg is None or any(r["dow"] not in self.base_puffs for r in last2):
            return False
        trend = np.mean([r["puffs"] / max(self.base_puffs[r["dow"]], 1) for r in last2]) - 1
        night = np.mean([r["night_puffs"] / max(r["puffs"], 1) - self.base_night[r["dow"]] for r in last2])
        return trend > self.Z * sg or night > FAST_NIGHT
    def next_dose(self, dose, window, prev_window):
        fired = self.alarm(window)
        self.update_baseline(window)
        if fired:
            if not self.in_episode:                        # punish once per episode
                self.personal_cut = max(self.min_cut, self.personal_cut * self.down)
                self.in_episode = True
            return dose
        self.in_episode = False                            # a clean day closes the episode
        return AdaptiveTaper.next_dose(self, dose, window, prev_window)


def stats(E, period, weeks, n=30, cut=0.12):
    rows = []
    for p in PROFILES:
        for s in range(n):
            df, _ = run_one(p, E, weeks, seed=s, weekly_cut=cut, period=period)
            last = df.iloc[-1]
            rows.append(dict(engine=E.name, period=period, weeks=weeks, profile=p.name,
                             relapsed=bool(last.relapsed), final=np.nan if last.relapsed else float(last.dose)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    frames = []
    for weeks in (14, 26):
        for E, period in ((FixedTaper, 7), (AdaptiveTaper, 7), (AdaptiveTaper, 1), (FastVsRecent, 1), (FastVsBaseline, 1), (FastSigma, 1)):
            frames.append(stats(E, period, weeks))
    df = pd.concat(frames)
    df["engine"] = df["engine"] + " @" + df["period"].astype(str) + "d"
    df.to_csv(pathlib.Path(__file__).with_name("fast_signal_results.csv"), index=False)

    print("Overall (5 profiles x 30 runs)")
    g = df.groupby(["weeks", "engine"], sort=False).agg(relapse=("relapsed", "mean"), final_mg=("final", "median"))
    print(g.to_string(formatters={"relapse": "{:.0%}".format, "final_mg": "{:.1f}".format}), "\n")
    for weeks in (14, 26):
        print(f"Per profile at {weeks} weeks: relapse rate | median final dose of survivors (mg/ml)")
        sub = df[df.weeks == weeks]
        pr = sub.pivot_table(index="profile", columns="engine", values="relapsed", aggfunc="mean", sort=False)
        pf = sub.pivot_table(index="profile", columns="engine", values="final", aggfunc="median", sort=False)
        out = pr.copy().astype(object)
        for c in pr.columns:
            out[c] = [f"{a:.0%} | {b:4.1f}" for a, b in zip(pr[c], pf[c])]
        print(out.to_string(), "\n")
