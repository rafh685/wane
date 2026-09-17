"""A synthetic population: random people drawn from ranges around the five named profiles.

Used to FIT the engine's weights on one population (A) and TEST them on another (B) that it never saw.
Ranges are the same sourced ranges as profiles.py (see docs/model_assumptions.md); routines are drawn
from a few templates so the population has office workers, evening-only vapers, shift patterns, students.
"""
import numpy as np
from profiles import Profile, Cue, _slots

ROUTINES = {
    "all_day":  (_slots((7, 9, 3), (9, 13, 1.5), (13, 14, 3), (14, 18, 1.5), (18, 24, 3)), _slots((9, 12, 2), (12, 20, 2), (20, 25, 3))),
    "evening":  (_slots((19, 26, 3)), _slots((14, 20, 1), (20, 28, 4))),
    "work_ban": (_slots((7, 9, 4), (13, 14, 4), (18, 24, 3)), _slots((11, 18, 2), (18, 27, 4))),
    "three_moments": (_slots((6, 8, 3), (12, 13, 3), (17, 22, 2)), _slots((8, 23, 1.5))),
    "early_shift": (_slots((5, 7, 4), (10, 11, 2), (14, 16, 3), (19, 22, 2)), _slots((8, 22, 2))),
}
WAKE = {"all_day": (7, 9), "evening": (9, 11), "work_ban": (7.5, 11), "three_moments": (6.5, 8), "early_shift": (4.5, 8)}
BED = {"all_day": (23.5, 25), "evening": (26, 28), "work_ban": (24, 27), "three_moments": (22.5, 23.5), "early_shift": (21.5, 23.5)}


def sample_profile(rng, name):
    kind = rng.choice(list(ROUTINES))
    wd, we = ROUTINES[kind]
    cues = []
    if rng.random() < 0.5:                       # coffee / morning cue on weekdays
        cues.append(Cue((0, 1, 2, 3, 4), 7, 9, rng.uniform(0.5, 1.2)))
    if rng.random() < 0.6:                       # alcohol / social Friday and/or Saturday
        days = tuple(sorted(rng.choice([4, 5], size=rng.integers(1, 3), replace=False)))
        cues.append(Cue(days, 20, 27, rng.uniform(1.0, 2.5), alcohol=True))
    if rng.random() < 0.3:                       # stress day
        cues.append(Cue((int(rng.integers(0, 5)),), 8, 18, rng.uniform(0.5, 1.5)))
    return Profile(
        name, f"synthetic {kind}",
        puffs_per_day=float(np.exp(rng.uniform(np.log(30), np.log(300)))),
        elasticity=float(rng.uniform(0.10, 0.80)),
        craving_sensitivity=float(rng.uniform(2.0, 8.0)),
        craving_decay=float(rng.uniform(0.08, 0.25)),
        relapse_threshold=float(rng.uniform(5.5, 8.5)),
        noise=float(rng.uniform(0.08, 0.25)),
        wake={"wd": WAKE[kind][0], "we": WAKE[kind][1]}, bed={"wd": BED[kind][0], "we": BED[kind][1]},
        weekday_routine=wd, weekend_routine=we,
        weekend_factor=float(rng.uniform(1.0, 2.2)),
        cues=cues,
    )


def population(n, seed):
    rng = np.random.default_rng(seed)
    return [sample_profile(rng, f"P{seed}-{i:03d}") for i in range(n)]
