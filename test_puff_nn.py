import math
import unittest

import numpy as np

from population import population
from puff_nn import V4, AdaptiveHead, BaseNet, OnTheSpot, gap_starts
from puffsim import SlowSchedule, simulate

BASELINE = {
    "puffs_per_day": 100.0, "puffs_wd": 100.0, "puffs_we": 100.0,
    "hourly": {"wd": np.r_[np.zeros(8), np.full(16, 100 / 16)], "we": np.r_[np.zeros(8), np.full(16, 100 / 16)]},
    "dur": 2.1, "longest_gap_h": 8.0, "puffs": [],
}


def fresh(adapt=True):
    c = OnTheSpot("nn", adapt=adapt)
    c.begin(BASELINE)
    return c


class OnTheSpotSafety(unittest.TestCase):
    def test_daily_budget_and_per_puff_ceiling_hold_under_chain_puffing(self):
        c = fresh()
        c.start_day(0, 0.6, False)
        total = 0.0
        for i in range(400):                                   # 400 puffs 10 s apart: far above the 100 habit
            t = 5 + 9 + i * 10 / 3600
            d = c.dose(t)
            self.assertLessEqual(d, min(1.0, 2 * 0.6) + 1e-12)
            c.observe(t, 2.1, d)
            total += d
        self.assertLessEqual(total, 0.6 * 100 + 1e-9)

    def test_dose_is_chosen_before_the_puff_so_pulling_harder_buys_nothing(self):
        a, b = fresh(), fresh()
        for c in (a, b):
            c.start_day(0, 0.8, False)
        for i, t in enumerate([9.0, 9.01, 9.02]):
            da, db = a.dose(t), b.dose(t)
            self.assertAlmostEqual(da, db)
            a.observe(t, 1.5, da)
            b.observe(t, 1.5 if i < 2 else 6.0, db)             # b's third puff is a long, hard pull
        self.assertLessEqual(b.dose(9.03), a.dose(9.03) + 1e-12)

    def test_base_is_frozen_and_only_the_head_moves(self):
        c = fresh()
        W1 = c.net.W1.copy()
        with self.assertRaises(ValueError):
            c.net.W1[0, 0] = 1.0
        for day in range(3):
            c.start_day(day, 0.8, False)
            for i in range(80):
                t = day * 24 + 5 + 4 + i * 0.15
                c.observe(t, 2.1, c.dose(t))
        c.start_day(3, 0.8, False)
        self.assertTrue(np.array_equal(W1, c.net.W1))
        self.assertGreater(c.head.updates, 0)
        self.assertGreater(np.linalg.norm(c.head.theta - c.head.theta0), 0)

    def test_head_drift_is_bounded(self):
        h = AdaptiveHead(np.zeros(17), max_drift=2.0)
        rng = np.random.default_rng(0)
        for _ in range(5000):
            h.update(rng.uniform(-1, 1, 16), 50.0)
        self.assertLessEqual(np.linalg.norm(h.theta - h.theta0), 2.0 + 1e-9)
        self.assertTrue(np.all(np.isfinite(h.P)))

    def test_network_failure_falls_back_to_habit(self):
        c = fresh()
        c.start_day(0, 0.5, False)
        c.head.theta[:] = float("nan")
        d = c.dose(10.0)
        self.assertTrue(math.isfinite(d) and 0 <= d <= 0.5 * 2)

    def test_closed_loop_run_on_a_synthetic_person(self):
        person = population(1, seed=99)[0]
        r = simulate(person, lambda: OnTheSpot("nn"), SlowSchedule(), taper_days=21, follow_days=0, seed=1)
        self.assertGreater(len(r["days"]), 21)
        for d in r["days"][21:]:
            self.assertLessEqual(d["dose_sum"], d["u"] * max(r["baseline_puffs"] * 3, 1))


class V4Shaping(unittest.TestCase):
    def evening_baseline(self):
        """Three weeks of an evening vaper: bouts of 10 puffs every 20 min from 19:00 to 01:40, then nothing."""
        puffs = []
        for day in range(21):
            for b in range(21):
                start = day * 24 + 19 + b / 3
                puffs += [(start + i * 10 / 3600, 2.1) for i in range(10)]
        return dict(BASELINE, puffs=puffs, puffs_per_day=210.0, puffs_wd=210.0, puffs_we=210.0,
                    hourly={"wd": np.r_[np.full(2, 30), np.zeros(17), np.full(5, 30)],
                            "we": np.r_[np.full(2, 30), np.zeros(17), np.full(5, 30)]})

    def test_gap_start_is_learned_from_device_data(self):
        c = OnTheSpot("habit", shape=V4)
        c.begin(self.evening_baseline())
        c.start_day(21, 0.8, False)
        self.assertAlmostEqual(c.gap_today, (1 + 40 / 60 + 9 * 10 / 3600) - 5 + 24, places=2)   # about 01:42 = 20.7 h after 05:00
        self.assertEqual(c.front_k, 3)

    def test_relief_bout_gets_more_and_the_pre_gap_hours_get_less(self):
        c = OnTheSpot("habit", shape=V4)
        c.begin(self.evening_baseline())
        c.start_day(21, 0.8, False)
        doses = []
        for b in range(21):
            start = 21 * 24 + 19 + b / 3
            for i in range(10):
                t = start + i * 10 / 3600
                d = c.dose(t)
                c.observe(t, 2.1, d)
                doses.append((b, i, d))
        first = np.mean([d for b, i, d in doses if b == 0 and i < 5])
        middle = np.mean([d for b, i, d in doses if 3 <= b <= 12])
        late = np.mean([d for b, i, d in doses if b >= 19])          # after 23:20, inside the 2 h before the gap
        self.assertGreater(first, middle)
        self.assertLess(late, middle)
        self.assertLessEqual(sum(d for _, _, d in doses), 0.8 * 210 + 1e-9)
        self.assertLessEqual(max(d for _, _, d in doses), min(1.0, 2 * 0.8) + 1e-12)

    def test_gap_starts_uses_the_longest_gap_of_each_device_day(self):
        times = [24 + 8, 24 + 9, 24 + 12, 24 + 16, 24 + 23, 48 + 8]       # device day 1 (from 05:00): longest gap 23:00 -> 08:00
        self.assertAlmostEqual(gap_starts(times)[1], 23 - 5)


if __name__ == "__main__":
    unittest.main()
