# Day generator: what the Wane engine needs from it
### Spec for Ilian · 18 September 2026 · one function, one output format, four rules, three checks

Your session simulator is the right idea. This turns it into the piece that replaces the hourly block inside `profiles.py`, so the engine can be tested on realistic bout structure. Everything the engine reads is listed here; nothing else is needed.

## 1. The function

```
simulate_day(craving, wake, bed, puffs_per_day, rng) -> list of puffs
```

| input | meaning | range |
|---|---|---|
| `craving` | the person's craving today, our scale | 0 to 10, resting is 2 |
| `wake`, `bed` | hours from midnight, floats | e.g. 7.5 and 23.5; bed may exceed 24 |
| `puffs_per_day` | this person's usual daily count at full strength | 50 to 450 |
| `rng` | `numpy.random.default_rng(seed)` | so runs repeat |

Each puff in the output is a dict: `t` (hours from midnight, float, 12.53 = 12:31:48), `dur` (seconds), `flow` (ml/s). Volume is `dur × flow`, never stored separately. Sorted by `t`.

The engine also needs, per day, `night_puffs` (puffs with `t` inside the sleep window) and `ttfc_min` (minutes from `wake` to the first puff). Compute both from the list; do not draw them.

## 2. Two timescales, not one

A day is a sequence of **bouts**. Inside a bout, puffs come 20 to 90 seconds apart, 3 to 15 of them, over 3 to 10 minutes. Between bouts, 20 minutes to 2 hours. Bouts never overlap: the next one starts after the previous one ends plus the gap.

Generate the day like this:
1. First bout starts at `wake + ttfc`.
2. Draw a gap between bouts (log-normal, median 45 minutes at resting craving), place the next bout, repeat until `bed`.
3. Inside each bout draw the puff count and the inter-puff intervals (log-normal, median 40 s).
4. Per puff draw `dur` and `flow`.
5. If the total for the day is far from `puffs_per_day × (1 + 0.06 × (craving − 2))`, scale the bout gap and regenerate. Puff count is the anchor, bouts are how it is spread.

Weekday routine (work bans, evening-only) is applied on top by `profiles.py`, not here. Your job is a plain day.

## 3. The four rules that link craving to behaviour

| what moves | at resting craving 2 | rule | source |
|---|---|---|---|
| time to first puff | 30 min | `30 × exp(−0.25 × (craving − 2))` minutes, floor 1 | Heatherton 1991: time to first use is the strongest dependence item |
| gap between bouts | median 45 min | `45 × exp(−0.12 × (craving − 2))` | bouts come closer as craving rises; direction from puff-topography field studies, size is a guess |
| puff duration | median 3.4 s | `3.4 × (1 + 0.03 × (craving − 2))`, log-normal spread 0.25 | 3.44 s mean from the 61-user PR-ENDS field study; rise with compensation from Dawkins 2018 |
| flow | median 20 ml/s | `20 × (1 + 0.02 × (craving − 2))`, spread 0.25, clip 8 to 40 | e-cigarette flow rates 15 to 35 ml/s in topography studies |

Night puffs are not a separate rule: they appear when a bout gap places a bout inside the sleep window, and that happens more as gaps shrink. Check that at craving 2 fewer than 3 % of puffs land in the sleep window, and at craving 7 about 5 to 8 %.

## 4. Realistic ranges, so a jury does not catch us

| quantity | realistic | your last run |
|---|---|---|
| puffs per day | 50 to 450, median about 200 | 22 |
| bouts per day | 10 to 30 | 3 |
| puffs per bout | 3 to 15 | 6 to 14 |
| inter-puff interval inside a bout | 20 to 90 s | 38 to 452 s |
| gap between bouts | 20 min to 2 h | overlapping |
| first puff after waking | 5 to 30 min for a dependent user | 4.5 h |
| puff duration | 1.5 to 6 s, mean 3.4 | fine |
| flow | 10 to 35 ml/s | 6.6 to 21, too low |

## 5. Three checks the code must pass

1. **Physics.** For every puff, `abs(volume − dur × flow) < 0.01`. Do not store volume at all and this cannot fail.
2. **Order.** `t` strictly increasing; no bout starts before the previous one ended.
3. **Response.** Run the same person at craving 2 and at craving 7, 20 days each. At 7: first puff earlier, more bouts, longer puffs, more puffs in the sleep window. If any of those does not move, a rule is not wired.

## 6. Output the engine reads, exactly

One dict per day, the same keys `profiles.py` produces today:

```
day, dow, dose, puffs, puff_dur, craving, night_puffs, ttfc_min, hours (24 counts), times (list of t), relapsed
```

Add `flows` (list, one per puff) and `volumes` computed on the fly. Nothing else changes: `engine.features()` reads `puffs`, `puff_dur`, `night_puffs`, `ttfc_min` and will get a new `bout` feature set once your generator is in.

## 7. Why this matters

Daily puff counts identify Lucía, the hard compensator, too late: her weekly response is the size of the noise. Bout structure changes faster and on two channels at once, duration and flow. If your generator shows that a compensator's bouts separate from a normal person's within two weeks, that is the strongest algorithm result we would have, and the first one that argues for the pressure sensor from the software side.
