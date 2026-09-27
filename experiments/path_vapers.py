"""Real adult vapers in the PATH Study (ICPSR 36498 v25, public use): quitting, relapse, time to first vape.

Data: Waves 1 (2013-14), 2 (2014-15) and 4 (2016-18) adult questionnaire files, downloaded by Codex into the
git-ignored data/external/path/ (see docs/path_data_handoff_for_claude.md) and cached as pickles by the loader
below. Respondent-level data never leave that folder; only aggregate estimates are written.

Definitions (PATH derived variables, value "(1) 1 = Yes"):
  current established vaper  R0xR_A_CUR_ESTD_ECIG  (ever fairly regular use and now every day or some days)
  daily vaper                R0xR_A_EDY_ECIG
  former established vaper   R0xR_A_FMR_ESTD_ECIG
  past-30-day use            R0xR_A_P30D_ECIG
  minutes to first vape      R01R_A_MINFIRST_ECIG (current users, minutes after waking)
  days since quitting        R01R_A_DAYSQUIT_ECIG (former users)
  current established smoker R01R_A_CUR_ESTD_CIGS
Outcomes one wave later (about 1 year): "stopped" = not a current established vaper and no past-30-day use;
"back" = current established vaper or any past-30-day use. Missing at the next wave = not counted (people who
dropped out are excluded, never counted as relapsed).
Weights: Wave 1 person weight R01_A_PWGT with its 100 Fay replicate weights (rho 0.3) for 95 % intervals. The
longitudinal all-wave weights are not in these files, so transition estimates use Wave 1 weights among people
re-interviewed; they ignore attrition bias. Unweighted counts are shown beside every estimate.
Run: ~/Projects/wane/.venv/bin/python experiments/path_vapers.py  (needs rdata or the cached pickles)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH_DIR = Path.home() / "Projects" / "wane" / "data" / "external" / "path" / "extracted" / "ICPSR_36498"
OUT = ROOT / "experiments" / "path-vapers-summary.json"
YES = "(1) 1 = Yes"
NO = "(2) 2 = No"


def load(ds):
    pkl = PATH_DIR / f"{ds}.pkl"
    if not pkl.exists():
        import rdata
        d = rdata.read_rda(PATH_DIR / ds / f"36498-{ds[2:]}-Data.rda", default_encoding="latin1")
        next(iter(d.values())).to_pickle(pkl)
    return pd.read_pickle(pkl)


def yes(s):
    return s == YES


def weighted(mask_outcome, mask_group, df, wcol="R01_A_PWGT", reps=100, rho=0.3):
    """Weighted proportion with Fay BRR 95 % interval, plus unweighted n and events."""
    g = mask_group.values
    y = mask_outcome.values & g
    def prop(w):
        return float(w[y].sum() / w[g].sum()) if w[g].sum() > 0 else float("nan")
    est = prop(df[wcol].values)
    rep = np.array([prop(df[f"{wcol}{r}"].values) for r in range(1, reps + 1)])
    se = math.sqrt(np.sum((rep - est) ** 2) / (reps * (1 - rho) ** 2))
    return dict(estimate=est, lo=max(0.0, est - 1.96 * se), hi=min(1.0, est + 1.96 * se),
                n=int(g.sum()), events=int(y.sum()), unweighted=float(y.sum() / g.sum()) if g.sum() else float("nan"))


def fmt(r):
    return f"{r['estimate']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]  (n = {r['n']}, events = {r['events']}, unweighted {r['unweighted']:.3f})"


def main():
    w1, w2, w4 = load("DS1001"), load("DS2001"), load("DS4001")
    w2i = w2.set_index("PERSONID")
    out = {}

    # --- time to first vape after waking, current established vapers at Wave 1
    cur1 = yes(w1["R01R_A_CUR_ESTD_ECIG"])
    daily1 = yes(w1["R01R_A_EDY_ECIG"])
    mf = w1["R01R_A_MINFIRST_ECIG"]
    cats = {"within 5 min": mf <= 5, "6 to 30 min": (mf > 5) & (mf <= 30), "31 to 60 min": (mf > 30) & (mf <= 60),
            "after 60 min": mf > 60}
    out["ttfv_daily_w1"] = {k: weighted(v & daily1, daily1 & mf.notna(), w1) for k, v in cats.items()}
    print("Wave 1 daily vapers, time to first vape after waking (weighted share):")
    for k, r in out["ttfv_daily_w1"].items():
        print(f"  {k:13s} {fmt(r)}")

    # --- current vapers at Wave 1: stopped by Wave 2?
    linked = w1["PERSONID"].isin(w2i.index)
    nxt = w2i.reindex(w1["PERSONID"])
    cur2 = pd.Series(yes(nxt["R02R_A_CUR_ESTD_ECIG"]).values, index=w1.index)
    p30_2 = pd.Series(yes(nxt["R02R_A_P30D_ECIG"]).values, index=w1.index)
    known2 = pd.Series((nxt["R02R_A_CUR_ESTD_ECIG"].notna() & nxt["R02R_A_P30D_ECIG"].notna()).values, index=w1.index)
    stopped = ~cur2 & ~p30_2
    smoker1 = yes(w1["R01R_A_CUR_ESTD_CIGS"])
    base = cur1 & linked & known2
    groups = {"all current vapers": base, "daily vapers": base & daily1, "some-days vapers": base & ~daily1,
              "exclusive vapers (not smoking)": base & ~smoker1, "dual users (also smoking)": base & smoker1}
    for k, v in cats.items():
        groups[f"daily vapers, first vape {k}"] = base & daily1 & v
    out["stopped_by_w2"] = {k: weighted(stopped, g, w1) for k, g in groups.items()}
    print("\nWave 1 current vapers who had stopped by Wave 2 (about 1 year):")
    for k, r in out["stopped_by_w2"].items():
        print(f"  {k:40s} {fmt(r)}")

    # --- former vapers at Wave 1: back by Wave 2, by days since quitting
    fmr1 = yes(w1["R01R_A_FMR_ESTD_ECIG"]) & linked & known2
    back = cur2 | p30_2
    dq = w1["R01R_A_DAYSQUIT_ECIG"]
    fgroups = {"all former vapers": fmr1, "quit < 30 days": fmr1 & (dq < 30), "quit 30 to 182 days": fmr1 & (dq >= 30) & (dq < 183),
               "quit 183 to 365 days": fmr1 & (dq >= 183) & (dq <= 365), "quit > 1 year": fmr1 & (dq > 365),
               "days since quit unknown": fmr1 & dq.isna()}
    out["back_by_w2"] = {k: weighted(back, g, w1) for k, g in fgroups.items()}
    print("\nWave 1 former vapers who were vaping again by Wave 2:")
    for k, r in out["back_by_w2"].items():
        print(f"  {k:30s} {fmt(r)}")

    # --- odds ratio of stopping, first vape within 30 min vs after 30 min (daily vapers), weighted
    early = base & daily1 & (mf <= 30)
    late = base & daily1 & (mf > 30)
    def odds(g, w):
        s = w[(stopped & g).values].sum(); n = w[g.values].sum()
        return s / max(n - s, 1e-9)
    w = w1["R01_A_PWGT"].values
    or_est = odds(late, w) / odds(early, w)
    reps = np.array([odds(late, w1[f"R01_A_PWGT{r}"].values) / odds(early, w1[f"R01_A_PWGT{r}"].values) for r in range(1, 101)])
    se = math.sqrt(np.sum((np.log(reps) - math.log(or_est)) ** 2) / (100 * 0.49))
    out["or_stop_late_vs_early_first_vape"] = dict(estimate=float(or_est), lo=float(math.exp(math.log(or_est) - 1.96 * se)),
                                                   hi=float(math.exp(math.log(or_est) + 1.96 * se)),
                                                   n_early=int(early.sum()), n_late=int(late.sum()))
    r = out["or_stop_late_vs_early_first_vape"]
    print(f"\nDaily vapers: odds of having stopped by Wave 2, first vape after 30 min vs within 30 min: "
          f"OR {r['estimate']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] (n {r['n_early']} early, {r['n_late']} late)")

    # --- Wave 2 -> Wave 4 (about 2 years), unweighted (Wave 4 has no weights in these files)
    w4i = w4.set_index("PERSONID")
    n4 = w4i.reindex(w2["PERSONID"])
    cur4 = yes(n4["R04R_A_CUR_ESTD_EPRODS"]).values
    p30_4 = yes(n4["R04R_A_P30D_EPRODS"]).values
    known4 = (n4["R04R_A_CUR_ESTD_EPRODS"].notna() & n4["R04R_A_P30D_EPRODS"].notna()).values
    cur2b = yes(w2["R02R_A_CUR_ESTD_ECIG"]).values & known4
    fmr2 = yes(w2["R02R_A_FMR_ESTD_ECIG"]).values & known4
    out["w2_to_w4_unweighted"] = dict(
        current_stopped=dict(n=int(cur2b.sum()), events=int((cur2b & ~cur4 & ~p30_4).sum())),
        former_back=dict(n=int(fmr2.sum()), events=int((fmr2 & (cur4 | p30_4)).sum())))
    for k, v in out["w2_to_w4_unweighted"].items():
        print(f"Wave 2 -> Wave 4 (about 2 years, unweighted) {k}: {v['events']} of {v['n']} = {v['events'] / max(v['n'], 1):.3f}")

    OUT.write_text(json.dumps(dict(source="PATH Study ICPSR 36498 v25 public use, Waves 1, 2, 4; aggregate estimates only",
                                   note="Wave 1 weights with Fay BRR (100 replicates, rho 0.3); transitions ignore attrition",
                                   results=out), indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
