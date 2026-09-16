"""Run the same people through both engines. Many times, with noise, to get bands not lines."""
import numpy as np
import pandas as pd
from profiles import PROFILES, Vaper, START_MG
from engine import FixedTaper, AdaptiveTaper


def run_one(profile, engine_cls, weeks=14, seed=0, weekly_cut=0.12, period=7):
    """Simulate one person for `weeks` weeks.

    period = days between engine decisions.
      7   manual weekly refill (today's product)
      2-3 pre-mixed kit
      1   hardware that meters the dose
    Whatever the period, the engine sees a rolling 7-day window and the 7 days before it,
    so weekend effects stay visible and the features stay comparable across periods.
    """
    rng = np.random.default_rng(seed)
    v = Vaper(profile, rng)
    eng = engine_cls(weekly_cut, period)
    dose = START_MG
    rows = []
    total_days = weeks * 7
    while v.t < total_days and not v.relapsed:
        for _ in range(period):
            r = v.day(dose)
            r["week"] = (r["day"] - 1) // 7 + 1
            rows.append(r)
            if v.relapsed or v.t >= total_days:
                break
        if v.relapsed:
            # relapse = back to 20 mg disposables; record the remaining days as failed
            for d in range(v.t + 1, total_days + 1):
                rows.append(dict(day=d, week=(d - 1) // 7 + 1, dose=START_MG, puffs=profile.puffs_per_day,
                                 craving=v.craving, night_puffs=0, ttfc_min=30, evening_share=0, relapsed=True))
            break
        if v.t >= total_days:
            break
        window = rows[-7:]                                  # last 7 days, whatever the period
        prev = rows[-14:-7] if len(rows) >= 14 else []      # the 7 before that; empty on the first decisions
        dose = eng.next_dose(dose, window, prev)
    df = pd.DataFrame(rows)
    df["profile"] = profile.name
    df["engine"] = eng.name
    df["seed"] = seed
    log = getattr(eng, "log", None)
    return df, log


def run_many(n_runs=100, weeks=14, weekly_cut=0.12, period=7, engines=(FixedTaper, AdaptiveTaper)):
    frames = []
    for p in PROFILES:
        for E in engines:
            for s in range(n_runs):
                df, _ = run_one(p, E, weeks, seed=s, weekly_cut=weekly_cut, period=period)
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
    import sys
    period = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    runs = run_many(n_runs=100, period=period)
    s = summarise(runs)
    s["relapse_rate"] = (s["relapse_rate"] * 100).round(0).astype(int).astype(str) + " %"
    s["final_dose_median"] = s["final_dose_median"].round(1)
    print(f"decision every {period} day(s)")
    print(s.pivot(index="profile", columns="engine", values=["relapse_rate", "final_dose_median"]).to_string())
