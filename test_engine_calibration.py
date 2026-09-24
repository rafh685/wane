import unittest

from engine import FittedTaper, features, make_baseline
from fit_real_data import build_calibration


def week(day, dose, puffs, duration):
    return [
        {
            "day": day + index,
            "dose": dose,
            "puffs": puffs,
            "puff_dur": duration,
            "night_puffs": 2,
            "ttfc_min": 30,
        }
        for index in range(7)
    ]


class RealDataCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.baseline = week(1, 20.0, 100, 3.4)

    def test_lsbu_calibration_is_reproducible(self):
        calibration = build_calibration()
        self.assertEqual(calibration["source"]["participants"], 20)
        self.assertEqual(calibration["source"]["paired_puff_measurements"], 19)
        self.assertAlmostEqual(calibration["total_puffing"]["median_ratio"], 1.468, places=3)
        self.assertAlmostEqual(calibration["total_puffing"]["p75_ratio"], 1.682, places=3)

    def test_extreme_response_reduces_rate_but_moderate_response_does_not(self):
        extreme = week(8, 18.0, 220, 8.5)
        engine = FittedTaper(weekly_cut=0.12, period=7)
        features_after_cut = features(extreme, self.baseline, dose=18.0, baseline=make_baseline(self.baseline))
        original_rate = engine.personal_cut
        engine.update_personal_rate(features_after_cut, 0.0, 2.0, 18.0, self.baseline, 1.0, 0.5)
        self.assertGreaterEqual(engine.response_pressure(features_after_cut, 18.0, self.baseline), 3.0)
        self.assertLess(engine.personal_cut, original_rate)

        moderate = week(8, 18.0, 101, 3.5)
        engine = FittedTaper(weekly_cut=0.12, period=7)
        features_after_cut = features(moderate, self.baseline, dose=18.0, baseline=make_baseline(self.baseline))
        original_rate = engine.personal_cut
        engine.update_personal_rate(features_after_cut, 0.0, 2.0, 18.0, self.baseline, 1.0, 0.5)
        self.assertLess(engine.response_pressure(features_after_cut, 18.0, self.baseline), 1.0)
        self.assertEqual(engine.personal_cut, original_rate)


if __name__ == "__main__":
    unittest.main()
