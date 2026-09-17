"""Run the same people through both engines. Many times, with noise, to get bands not lines."""
import numpy as np
import pandas as pd
from profiles import PROFILES, Vaper, START_MG
from engine import FixedTaper, AdaptiveTaper

CLEAN_DAYS = 14     # two weeks at 0 mg without relapse counts as "off nicotine"; the run stops there


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
    clean_at_zero = 0                 # days at 0 mg without relapse; CLEAN_DAYS of them = off nicotine
    while v.t < total_days and not v.relapsed:
        for _ in range(period):
            r = v.day(dose)
            r["week"] = (r["day"] - 1) // 7 + 1
            rows.append(r)
            if dose == 0.0 and not v.relapsed:
                clean_at_zero += 1
            if v.relapsed or v.t >= total_days or clean_at_zero >= CLEAN_DAYS:
                break
        if clean_at_zero >= CLEAN_DAYS:
            break                     # off nicotine: nothing left to decide
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


def per_run(all_runs):
    """One row per run: relapsed, reached zero, week it reached zero, final dose."""
    g = all_runs.sort_values("day").groupby(["profile", "engine", "seed"])
    last = g.tail(1).set_index(["profile", "engine", "seed"])
    zero_day = all_runs[all_runs.dose == 0.0].groupby(["profile", "engine", "seed"]).day.min()
    out = pd.DataFrame({"relapsed": last.relapsed, "final_dose": last.dose})
    out["zero_day"] = zero_day
    out["reached_zero"] = out.zero_day.notna() & ~out.relapsed
    out["weeks_to_zero"] = out.zero_day / 7
    return out.reset_index()


def summarise(all_runs):
    """Per profile x engine: relapse rate, share off nicotine, median weeks to zero (of those who got there), final dose."""
    r = per_run(all_runs)
    out = r.groupby(["profile", "engine"]).agg(
        relapse_rate=("relapsed", "mean"),
        off_rate=("reached_zero", "mean"),
        weeks_to_zero_median=("weeks_to_zero", lambda x: x[r.loc[x.index, "reached_zero"]].median()),
        final_dose_median=("final_dose", "median"),
    ).reset_index()
    return out


if __name__ == "__main__":
    import sys
    period = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    WEEKS = 30
    runs = run_many(n_runs=100, weeks=WEEKS, period=period)
    s = summarise(runs)
    for c in ("relapse_rate", "off_rate"):
        s[c] = (s[c] * 100).round(0).astype(int).astype(str) + " %"
    s["weeks_to_zero_median"] = s["weeks_to_zero_median"].round(1)
    print(f"decision every {period} day(s), up to {WEEKS} weeks")
    print(s.pivot(index="profile", columns="engine", values=["relapse_rate", "off_rate", "weeks_to_zero_median"]).to_string())
