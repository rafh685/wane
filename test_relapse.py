import unittest

import numpy as np

from population import population
from puff_nn import Flat
from puffsim import END_PLAN, ManualTaper, SlowSchedule, simulate
from relapse import RelapseParams, daily_logits, evaluate, replay


class RelapseModelV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = population(1, seed=77)[0]
        cls.res = simulate(cls.p, Flat, SlowSchedule(), taper_days=28, follow_days=0, seed=3, burn_in_days=28,
                           stop_at_relapse=False)

    def test_burn_in_leaves_recorded_days_starting_at_zero(self):
        self.assertEqual(self.res["days"][0]["day"], 0)
        self.assertTrue(all("mean_r" in d and "alcohol" in d for d in self.res["days"]))

    def test_an_ordinary_bad_day_is_not_a_cliff(self):
        params = RelapseParams.calibrated()
        lg, _ = daily_logits(self.res, self.p, params, 3)
        # one baseline SD of 3-day craving moves the daily odds by exp(beta / kappa), a few percent, not x7
        self.assertLess(np.exp(params.beta / params.kappa), 1.2)
        self.assertTrue(np.all(np.isfinite(lg)))

    def test_more_craving_means_more_risk(self):
        params = RelapseParams.calibrated()
        lg = np.full(120, params.alpha_mean)
        avail = np.ones(120, dtype=bool)
        low = replay(lg, avail, params, n_mc=2000, seed=1)[0].mean()
        high = replay(lg + 1.0, avail, params, n_mc=2000, seed=1)[0].mean()
        self.assertGreater(high, low)

    def test_nicotine_still_in_the_plan_lowers_relapse_overall(self):
        # calibrated 27 Sept: fewer slips while nicotine is in the plan (taper_shift), even though each slip snowballs
        # about as much; overall relapse must still be lower than with no nicotine left
        params = RelapseParams.calibrated()
        lg = np.full(120, params.alpha_mean)
        with_nic = replay(lg + params.taper_shift, np.ones(120, dtype=bool), params, n_mc=2000, seed=2)[0].mean()
        without = replay(lg, np.zeros(120, dtype=bool), params, n_mc=2000, seed=2)[0].mean()
        self.assertLess(with_nic, without)

    def test_evaluate_returns_probabilities(self):
        ev = evaluate(self.res, self.p, n_mc=200)
        self.assertTrue(0 <= ev["expected_risk"] <= 1)


class TraditionalTaper(unittest.TestCase):
    def levels(self, sched, puffs):
        return [sched.level(d, puffs(d)) for d in range(252)]

    def test_first_version_only_shop_strengths_and_zero_by_week_24(self):
        for seed in range(20):
            lv = self.levels(ManualTaper(seed=seed, stall=None, give_up=False, end_day=168), lambda d: 100)
            self.assertTrue(set(lv) <= set(ManualTaper.LADDER))
            self.assertEqual(lv[168:], [0.0] * 84)
            steps = [d for d in range(1, 168) if lv[d] < lv[d - 1]]           # the week-24 deadline may come sooner
            self.assertTrue(all(b - a >= 28 for a, b in zip(steps, steps[1:])))

    def test_goes_back_to_the_stronger_bottle_when_puffing_jumps(self):
        backs = 0
        for seed in range(40):
            s = ManualTaper(seed=seed, stall=None, give_up=False, end_day=168)
            lv = []
            for d in range(252):
                prev = lv[-1] if lv else 1.0
                lv.append(s.level(d, 100 if prev == 1.0 else 150))      # 50 % more puffs on every weaker bottle
            backs += s.step_backs
            self.assertLessEqual(s.step_backs, 4)
        self.assertTrue(10 < backs < 150)

    def test_some_people_get_stuck_on_a_bottle_and_never_reach_zero(self):
        stuck = 0
        for seed in range(200):
            s = ManualTaper(seed=seed)
            lv = self.levels(s, lambda d: 100)
            self.assertTrue(set(lv) <= set(ManualTaper.LADDER))
            if s.stuck:
                stuck += 1
                self.assertGreater(lv[-1], 0.0)
                last_change = max(d for d in range(1, 252) if lv[d] != lv[d - 1]) if len(set(lv)) > 1 else 0
                self.assertTrue(all(x == lv[-1] for x in lv[last_change:]))
            else:
                self.assertEqual(lv[-1], 0.0)
        # per-step stall chances 0.10, 0.10, 0.10, 0.25: about 1 - 0.9^3 x 0.75 = 45 % stuck
        self.assertTrue(70 < stuck < 110)

    def test_stalls_do_not_change_step_timing_before_the_stall(self):
        for seed in range(30):
            a = self.levels(ManualTaper(seed=seed, stall=None, give_up=False), lambda d: 100)
            s = ManualTaper(seed=seed)
            b = self.levels(s, lambda d: 100)
            first_diff = next((d for d in range(252) if a[d] != b[d]), None)
            if first_diff is not None:
                self.assertTrue(s.stuck)
                self.assertGreater(b[first_diff], a[first_diff])


class EndPlan(unittest.TestCase):
    def weekly(self, **kw):
        s = SlowSchedule(**kw)
        s.begin({"puffs_per_day": 100})
        return [s.level(d, 100) for d in range(0, 30 * 7, 7)]

    def test_front_loaded_plan_ends_in_week_23_like_the_current_one(self):
        cur, new = self.weekly(), self.weekly(**END_PLAN)
        self.assertEqual(cur.index(0.0), 23)
        self.assertEqual(new.index(0.0), 23)
        self.assertAlmostEqual(new[9], 0.15)
        steps = [a - b for a, b in zip(new[9:23], new[10:24])]
        self.assertTrue(all(abs(x - 0.15 / 14) < 1e-9 for x in steps))
        self.assertLess(sum(new[:23]), sum(cur[:23]))            # less nicotine overall, gentler bottom


if __name__ == "__main__":
    unittest.main()
