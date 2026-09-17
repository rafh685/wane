"""Run the same people through both engines. Many times, with noise, to get bands not lines."""
import numpy as np
import pandas as pd
from profiles import PROFILES, Vaper, START_MG
from engine import FixedTaper, AdaptiveTaper

CLEAN_DAYS = 14     # two weeks at 0 mg without relapse counts as "off nicotine"; the run stops there
STALL_WEEKS = 8     # this many consecutive skipped steps = the person has stalled on the ladder (dropped out of the taper)


def step_taken(rng, craving, week, adherence):
    """Manual product: the user has to mix or buy a weaker liquid and switch. Do they, this week?
    adherence = None or 1.0 -> automatic (hardware meters the dose): always.
    Otherwise a base probability, lower when craving is up (why make it harder now?) and drifting down
    over the weeks (motivation fatigue). NRT literature: only about a third to a half complete a schedule."""
    if adherence is None or adherence >= 1.0:
        return True
    p = adherence * np.exp(-0.25 * max(0.0, craving - 3.0)) * 0.99 ** week
    return rng.random() < p


def run_one(profile, engine_cls, weeks=14, seed=0, weekly_cut=0.12, period=7, adherence=None):
    """Simulate one person for `weeks` weeks.

    adherence = None: every dose change the engine decides is applied (hardware, or a perfectly compliant user).
    adherence = 0.85: manual product; each step is applied with about that probability, less when craving is up.

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
    skipped = 0                       # consecutive engine decisions the user did not apply
    stalled = False
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
                rows.append(dict(day=d, week=(d - 1) // 7 + 1, dose=START_MG, puffs=profile.puffs_per_day, puff_dur=3.4,
                                 craving=v.craving, night_puffs=0, ttfc_min=30, relapsed=True))
            break
        if v.t >= total_days:
            break
        window = rows[-7:]                                  # last 7 days, whatever the period
        prev = rows[-14:-7] if len(rows) >= 14 else []      # the 7 before that; empty on the first decisions
        proposed = eng.next_dose(dose, window, prev)
        if proposed < dose and not step_taken(rng, v.craving, v.t / 7, adherence):
            skipped += 1                                    # the user did not switch to the weaker liquid
            if skipped * period >= STALL_WEEKS * 7:
                stalled = True
                break
        else:
            dose = proposed
            skipped = 0
    df = pd.DataFrame(rows)
    df["stalled"] = stalled
    df["profile"] = profile.name
    df["engine"] = eng.name
    df["seed"] = seed
    log = getattr(eng, "log", None)
    return df, log


def run_many(n_runs=100, weeks=14, weekly_cut=0.12, period=7, engines=(FixedTaper, AdaptiveTaper), adherence=None):
    frames = []
    for p in PROFILES:
        for E in engines:
            for s in range(n_runs):
                df, _ = run_one(p, E, weeks, seed=s, weekly_cut=weekly_cut, period=period, adherence=adherence)
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def per_run(all_runs):
    """One row per run: relapsed, reached zero, week it reached zero, final dose."""
    g = all_runs.sort_values("day").groupby(["profile", "engine", "seed"])
    last = g.tail(1).set_index(["profile", "engine", "seed"])
    zero_day = all_runs[all_runs.dose == 0.0].groupby(["profile", "engine", "seed"]).day.min()
    out = pd.DataFrame({"relapsed": last.relapsed, "final_dose": last.dose, "stalled": last.get("stalled", False)})
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
        stall_rate=("stalled", "mean"),
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
