import unittest

import numpy as np

from population import population
from puff_nn import Flat
from puffsim import SlowSchedule, simulate
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

    def test_nicotine_availability_slows_lapse_to_relapse(self):
        params = RelapseParams.calibrated()
        lg = np.full(120, params.alpha_mean)
        with_nic = replay(lg, np.ones(120, dtype=bool), params, n_mc=2000, seed=2)[0].mean()
        without = replay(lg, np.zeros(120, dtype=bool), params, n_mc=2000, seed=2)[0].mean()
        self.assertLess(with_nic, without)

    def test_evaluate_returns_probabilities(self):
        ev = evaluate(self.res, self.p, n_mc=200)
        self.assertTrue(0 <= ev["expected_risk"] <= 1)


if __name__ == "__main__":
    unittest.main()
