"""Run the same people through both engines. Many times, with noise, to get bands not lines."""
import numpy as np
import pandas as pd
from profiles import PROFILES, Vaper, START_MG
from engine import FixedTaper, AdaptiveTaper


def run_one(profile, engine_cls, weeks=14, seed=0, weekly_cut=0.12):
    rng = np.random.default_rng(seed)
    v = Vaper(profile, rng)
    eng = engine_cls(weekly_cut)
    dose = START_MG
    rows, prev_week = [], []
    for w in range(weeks):
        week = []
        for d in range(7):
            r = v.day(dose)
            r["week"] = w + 1
            week.append(r)
            rows.append(r)
            if r["relapsed"]:
                break
        if v.relapsed:
            # relapse = back to 20 mg disposables; record the remaining weeks as failed
            for w2 in range(w + 1, weeks):
                rows.append(dict(day=v.t + 7 * (w2 - w), week=w2 + 1, dose=START_MG, puffs=profile.puffs_per_day,
                                 craving=v.craving, night_puffs=0, ttfc_min=30, evening_share=0, relapsed=True))
            break
        dose = eng.next_dose(dose, week, prev_week)
        prev_week = week
    df = pd.DataFrame(rows)
    df["profile"] = profile.name
    df["engine"] = eng.name
    df["seed"] = seed
    log = getattr(eng, "log", None)
    return df, log


def run_many(n_runs=100, weeks=14, weekly_cut=0.12, engines=(FixedTaper, AdaptiveTaper)):
    frames = []
    for p in PROFILES:
        for E in engines:
            for s in range(n_runs):
                df, _ = run_one(p, E, weeks, seed=s, weekly_cut=weekly_cut)
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarise(all_runs):
    """Per profile x engine: relapse rate and final-week dose (median)."""
    last = all_runs.sort_values("day").groupby(["profile", "engine", "seed"]).tail(1)
    out = last.groupby(["profile", "engine"]).agg(
        relapse_rate=("relapsed", "mean"),
        final_dose_median=("dose", "median"),
    ).reset_index()
    return out


if __name__ == "__main__":
    runs = run_many(n_runs=100)
    s = summarise(runs)
    s["relapse_rate"] = (s["relapse_rate"] * 100).round(0).astype(int).astype(str) + " %"
    s["final_dose_median"] = s["final_dose_median"].round(1)
    print(s.pivot(index="profile", columns="engine", values=["relapse_rate", "final_dose_median"]).to_string())
