"""Do the simulated vapers look like real ones? Compares the realistic generator with published values. SIMULATION.

60 synthetic people (calibration population, seed 61), 3 baseline weeks at full strength, then 2 weeks held at one
third of the dose (the LSBU 18 -> 6 mg/ml contrast, Dawkins 2018: puffs x1.21, puff length x1.24, total puff time
x1.50 after a week at fixed power).
Run: python3 experiments/realism_checks.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from population import population                     # noqa: E402
from puff_nn import Flat                              # noqa: E402
from puffsim import WARMUP_DAYS, Person, SlowSchedule, simulate   # noqa: E402

PEOPLE = population(60, seed=61)


class HoldThird(SlowSchedule):
    def level(self, taper_day, last):
        return 1 / 3


def run(i):
    p = PEOPLE[i]
    rec = []
    r = simulate(p, Flat, HoldThird(), taper_days=14, follow_days=0, seed=6100 + i, burn_in_days=28,
                 stop_at_relapse=False, record=rec)
    base = [x for x in rec if x[0] < WARMUP_DAYS]
    t = np.array([x[1] for x in base]); dur = np.array([x[2] for x in base])
    gaps = np.diff(t) * 3600
    split = np.r_[0, np.where(gaps > 60)[0] + 1]
    sizes = np.diff(np.r_[split, len(t)])
    ipi = gaps[gaps <= 60]
    per_day = np.array([d["puffs"] for d in r["days"][:WARMUP_DAYS]])
    dows = np.array([d["day"] % 7 for d in r["days"][:WARMUP_DAYS]])
    sess = np.diff(np.r_[0, np.where(gaps > 5 * 60)[0] + 1, len(t)])
    night = [x for x in base if (x[1] % 24) < 5 or (x[1] % 24) >= 23.99]
    hold = [x for x in rec if x[0] >= WARMUP_DAYS + 7]              # second week at one third
    wk3 = [x for x in base if x[0] >= WARMUP_DAYS - 7]
    pers = Person(p, np.random.default_rng(6100 + i), slow_rng=np.random.default_rng(6100 + i + 104729))
    return dict(
        puffs_per_day=float(per_day.mean()), target=p.puffs_per_day,
        bout_mean=float(sizes.mean()), single=float(np.mean(sizes == 1)), ipi_median=float(np.median(ipi)) if len(ipi) else np.nan,
        sessions_per_day=len(sess) / WARMUP_DAYS, session_mean=float(sess.mean()),
        dur_median=float(np.median(dur)), dur_cv=float(np.std(dur) / np.mean(dur)), tank=pers.tank,
        day_cv=float(per_day.std() / per_day.mean()),
        weekend_ratio=float(per_day[np.isin(dows, [5, 6])].mean() / per_day[~np.isin(dows, [5, 6])].mean()),
        night_waker=pers.night_waker,
        puffs_ratio=len(hold) / 7 / (len(wk3) / 7),
        dur_ratio=float(np.mean([x[2] for x in hold]) / np.mean([x[2] for x in wk3])),
        total_ratio=float(np.sum([x[2] for x in hold]) / np.sum([x[2] for x in wk3])),
    )


if __name__ == "__main__":
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(run, range(len(PEOPLE))))
    m = lambda k: float(np.nanmedian([r[k] for r in rows]))
    tank = [r for r in rows if r["tank"]]; pod = [r for r in rows if not r["tank"]]
    table = [
        ("puffs per day (median of people)", m("puffs_per_day"), "median about 130 to 290 (Dautzenberg, Kosmider, LSBU)"),
        ("puffs per bout, 60 s rule (median of people)", m("bout_mean"), "4.0 refillable, 2.4 pods"),
        ("single-puff bouts", m("single"), "0.37 to 0.47"),
        ("gap between puffs in a bout, s", m("ipi_median"), "median 13 s"),
        ("sessions per day, 5 min rule", m("sessions_per_day"), "15.3 (Kosmider)"),
        ("puffs per session, 5 min rule", m("session_mean"), "10.2 (Kosmider)"),
        ("puff length, tank users, s", float(np.median([r["dur_median"] for r in tank])), "3.1 to 3.4"),
        ("puff length, pod users, s", float(np.median([r["dur_median"] for r in pod])), "about 2.2"),
        ("puff-to-puff CV of length", m("dur_cv"), "about 0.5"),
        ("day-to-day CV of puffs", m("day_cv"), "0.2 to 0.6, median about 0.35"),
        ("weekend / weekday puffs", m("weekend_ratio"), "0.75 to 0.95"),
        ("share of night wakers", float(np.mean([r["night_waker"] for r in rows])), "0.07 to 0.10"),
        ("after a 3x cut: puffs per day ratio", m("puffs_ratio"), "x1.18 to 1.21 (LSBU)"),
        ("after a 3x cut: puff length ratio", m("dur_ratio"), "x1.24 to 1.26 (LSBU)"),
        ("after a 3x cut: total puff time ratio", m("total_ratio"), "x1.47 to 1.50, IQR 1.34 to 1.68 (LSBU)"),
    ]
    for name, v, ref in table:
        print(f"{name:46s} {v:8.2f}   published: {ref}")
    (ROOT / "experiments" / "realism-checks.json").write_text(json.dumps([dict(check=n, model=v, published=r) for n, v, r in table], indent=1) + "\n")
