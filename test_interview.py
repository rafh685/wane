import unittest

import numpy as np

from interview import InterviewPolicy, answer_levels, synthetic_answers, true_difficulty, weeks_to_zero
from population import population
from puff_nn import V4, OnTheSpot
from test_puff_nn import BASELINE

EASY = dict(ttfv=2, night=0, history=0, mood=0, alcohol=0)    # a mid-range person, pace not at a limit
POL = InterviewPolicy(base_cut=0.14, k=2.0, morning_x=0.5, night_relief=True, pregap_social=0.6)


class InterviewAnswers(unittest.TestCase):
    def test_every_simulated_answer_changes_the_plan(self):
        base = POL.plan(EASY)
        for q, v in (("ttfv", 0), ("night", 3), ("history", 3), ("mood", 1), ("alcohol", 1)):
            other = POL.plan(dict(EASY, **{q: v}))
            self.assertTrue(other["cut"] != base["cut"] or other["shape"] != base["shape"], q)

    def test_harder_answers_mean_a_slower_taper(self):
        hard = dict(ttfv=0, night=3, history=3, mood=1, alcohol=1)
        self.assertLess(POL.plan(hard)["cut"], POL.plan(EASY)["cut"])
        self.assertGreaterEqual(POL.plan(hard)["cut"], POL.min_cut)
        self.assertLessEqual(weeks_to_zero(POL.plan(hard)["cut"]), 52)

    def test_answers_track_hidden_difficulty_and_shares(self):
        pop = population(600, seed=3)
        rng = np.random.default_rng(0)
        ans = [synthetic_answers(p, rng) for p in pop]
        d = np.array([true_difficulty(p) for p in pop])
        ttfv = np.array([answer_levels(a)["ttfv"] for a in ans])
        self.assertGreater(np.corrcoef(ttfv, d)[0, 1], 0.2)
        night_wakers = np.mean([a["night"] >= 2 for a in ans])
        self.assertTrue(0.04 < night_wakers < 0.16, night_wakers)


class AnswerBehaviours(unittest.TestCase):
    def controller(self, shape):
        c = OnTheSpot("habit", shape=shape)
        c.begin(dict(BASELINE, puffs=[(d * 24 + 8 + h, 2.1) for d in range(21) for h in range(0, 15)]))
        c.start_day(21, 0.8, False)
        return c

    def test_morning_boost_raises_the_first_puff_after_the_night(self):
        plain, boosted = self.controller(V4), self.controller(dict(V4, morning_x=0.5))
        self.assertGreater(boosted.shape_weight(21 * 24 + 8.0), plain.shape_weight(21 * 24 + 8.0))

    def test_night_relief_spares_puffs_inside_the_long_gap(self):
        plain, spared = self.controller(V4), self.controller(dict(V4, night_relief=True))
        t = 21 * 24 + 5 + plain.gap_today + 2.0          # two hours into the usual night gap
        self.assertGreater(spared.shape_weight(t), plain.shape_weight(t))

    def test_drinking_nights_soften_the_pre_gap_cut(self):
        c = self.controller(dict(V4, social_dows=(21 % 7,), pregap_social=0.6))
        plain = self.controller(V4)
        t = 21 * 24 + 5 + plain.gap_today - 0.5
        self.assertGreater(c.shape_weight(t), plain.shape_weight(t))


if __name__ == "__main__":
    unittest.main()
