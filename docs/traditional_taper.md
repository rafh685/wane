# Wane against the traditional taper people do by hand

SIMULATION ONLY. Synthetic vapers, relapse model v2, realistic vapers (27 Sept 2026, late).

The sections before "Update" describe the first version (nobody stuck, zero by week 24). The committed runs and
summaries (`experiments/traditional-taper-*`) are from the update, where people can get stuck on the ladder.

## What was asked (Rafael)

"i need in the road map the traditional taper, and replace the name by a name everybody understands, and what does
the percentages represent, you need to make that clear" and "we need to see the efficacity of wane".

## Why a new control was needed

Every earlier result compared Wane with a control called "weaker bottle" (`puff_nn.Flat` on `SlowSchedule`). That
control is Wane's own smooth weekly plan (12 % less nicotine per puff each week, zero in week 23) with every puff at
the same strength. It isolates what per-puff dosing adds, but it is not what vapers do by hand: nobody buys a new
strength every week. It is now called **"Wane's weekly plan only"**.

## The traditional taper (`puffsim.ManualTaper`)

Settings fixed before running; nothing was tuned on results.

- Shop strengths 18, 12, 6, 3, 0 mg/ml (1, 2/3, 1/3, 1/6, 0 of the starting strength), every puff at the bottle's strength.
- A new bottle every 4 to 7 weeks (drawn per step). Zero came in week 22.6 on average (range 18 to 24), against week 23 for Wane.
- Stepping back (assumption): if the first week on a new bottle brings 15 % more puffs than the week before the step,
  half of people go back to the stronger bottle for 2 weeks, then try again; each step is undone at most once.
  Result: 0.62 step-backs per person, and 50 % of runs had at least one.
- Nicotine-free by week 24 at the latest, the same window as Wane.

## Results (`experiments/traditional_taper_eval.py`, same 117 people, seeds and windows as `relapse_v2_eval.py`)

Everyone is followed for 36 weeks from the first taper day (24 weeks of taper, 12 after). Numbers are people out of 100.

| Typical vapers (65) | Relapse | During taper | After zero | Still off at 9 months |
|---|---:|---:|---:|---:|
| Stop all at once | 86.8 | 0 | 86.8 | 13.2 |
| Traditional taper by hand | 77.7 | 38.3 | 39.3 | 22.3 |
| Wane's weekly plan only | 73.3 | 42.7 | 30.6 | 26.7 |
| Wane (v4) | 71.6 | 42.3 | 29.3 | 28.4 |

| Difficult people (52) | Relapse | Still off |
|---|---:|---:|
| Stop all at once | 99.9 | 0.1 |
| Traditional taper by hand | 99.5 | 0.5 |
| Wane's weekly plan only | 99.1 | 0.9 |
| Wane (v4) | 98.8 | 1.2 |

Paired differences in relapse per 100 people (95 % bootstrap interval over people):

- Wane vs traditional: typical -6.05 [-7.81, -4.45], better for 39 of 65, worse for none; difficult -0.66 [-1.40, -0.07].
- Wane's weekly plan only vs traditional: typical -4.39 [-5.69, -3.16]; difficult -0.35.
- Wane vs its weekly plan only (the per-puff part): typical -1.66 [-2.14, -1.23]; difficult -0.32.
- Traditional vs stopping at once: typical -9.11 [-11.95, -6.43].
- Lucía: stop at once 100.0, traditional 99.0, weekly plan only 98.9, Wane 98.4. Karim: 98.4, 86.3, 80.4, 77.3.

So of Wane's 6 extra people in 100, about 4 come from small weekly drops instead of big bottle jumps and about 2 from
per-puff dosing.

Why the hand taper loses: it keeps people on stronger bottles for longer early on, so slightly fewer relapse during
the taper (38 against 42), but its last step, 3 mg/ml to nothing, is a cliff: 39 relapse after zero against 29.

## Sensitivity (`experiments/traditional_taper_sensitivity.py`)

| Hand taper modelled as | Zero at week | Traditional still off | Wane minus traditional, relapse per 100 |
|---|---:|---:|---:|
| Main assumptions | 22.6 | 22.3 | -6.05 [-7.81, -4.45] |
| Never steps back | 21.8 | 22.0 | -6.38 [-8.13, -4.68] |
| Always steps back when puffing jumps | 23.1 | 22.4 | -5.96 [-7.77, -4.37] |
| Quicker steps, every 3 to 5 weeks | 17.4 | 19.8 | -8.63 [-11.01, -6.36] |
| Slower steps, every 5 to 8 weeks | 23.9 | 22.6 | -5.79 [-7.51, -4.12] |
| Salt bottles 20, 10, 5, 0, every 5 to 8 weeks | 20.4 | 20.2 | -8.16 [-10.46, -6.00] |

Wane stays ahead by 6 to 9 people in 100 for typical vapers under every variant; difficult people gain 0.6 to 0.9.

## What it means

- Against what vapers can do today without a device, Wane leaves about 6 more people in 100 nicotine-free at 9 months
  in this simulation. The traditional taper lands at 22 in 100, the top of the real taper-trial range (15 to 22 in 100,
  smokers cutting down gradually), which is a useful sanity check but not a validation.
- Most of the gain comes from the smooth weekly plan, which a manual taper cannot follow with shop strengths. The
  per-puff engine adds about 2 in 100 on top.
- Difficult people relapse almost whatever the method; dose timing alone does not rescue them.
- The stepping-back rule and the step timing are assumptions, not measurements; the sensitivity table shows they
  change the size of the gain but not its direction.

## Update, 27 Sept (night): people stuck on the ladder

Rafael: the roadmap did not show people who get stuck on the ladder. The first version forced everyone to zero by
week 24, which hid them. `ManualTaper` now allows it (settings fixed before running, both assumptions):

- **Stall:** at each step, a chance of never buying the next bottle and staying on the current strength for good:
  10 % for the steps to 12, 6 and 3 mg/ml, 25 % for the last step to no nicotine at all.
- **Give up:** failing the same step twice (puffing jumps again on the retry) sends half of those people back to the
  stronger bottle for good.
- No deadline: a slow ladder finishes late instead of being forced to zero.

No study measures stalling in self-tapers. The closest data: in the ITC Four Country survey (2016 to 2018), 85 % of
vapers kept the same strength over two years, and 35 % of those on 0 mg/ml went back up to nicotine.

The main outcome is now **nicotine-free at 9 months**: not relapsed, and on 0 mg/ml on the last day. Someone still
vaping a low-strength bottle without having relapsed is **stuck**: neither relapsed nor free (the slip model protects
them while nicotine is in their plan, so they relapse less).

| Typical vapers (65), per 100 | Relapse during taper | Stuck on a bottle | Relapse after zero | Nicotine-free |
|---|---:|---:|---:|---:|
| Stop all at once | 0 | 0 | 86.8 | 13.2 |
| Traditional taper by hand | 38.5 | 35.4 | 18.0 | 8.2 |
| Wane's weekly plan only | 42.7 | 0 | 30.6 | 26.7 |
| Wane (v4) | 42.3 | 0 | 29.3 | 28.4 |

57 % of hand-taper runs ended on a bottle (including those who relapsed); 26 in 100 reached zero at all.
Difficult people: nicotine-free 0.1 / 0.5 / 0.9 / 1.2, stuck 4.4 on the hand taper.

Nicotine-free, Wane minus traditional: typical +20.2 per 100 [+14.1, +26.9], better for 41 of 65, worse for none;
difficult +0.7 [+0.0, +1.7].

How much it depends on the stuck assumptions (`experiments/traditional_taper_sensitivity.py`, nicotine-free per 100,
typical vapers):

| Hand taper modelled as | Ends on a bottle | Hand taper free | Wane minus hand taper |
|---|---:|---:|---:|
| Nobody stuck, zero by week 24 (first version) | 0 % | 22.3 | +6.1 [+4.4, +7.7] |
| Nobody stuck, no deadline | 0 % | 23.4 | +5.0 [+3.6, +6.7] |
| Half as many stalls | 40 % | 13.3 | +15.1 [+10.5, +20.3] |
| Main assumptions | 57 % | 8.2 | +20.2 [+14.1, +26.9] |
| Twice as many stalls | 76 % | 3.0 | +25.4 [+17.8, +33.9] |
| Never steps back | 50 % | 10.2 | +18.2 |
| Quicker steps, every 3 to 5 weeks | 54 % | 7.9 | +20.5 |
| Slower steps, every 5 to 8 weeks | 57 % | 11.6 | +16.8 |
| Salt bottles 20, 10, 5, 0, every 5 to 8 weeks | 50 % | 9.9 | +18.6 |

Where Wane's 20 extra people come from, under the main assumptions: 8.2 (hand taper) +15.2 (nobody gets stuck,
because the device lowers the strength itself) +3.3 (small weekly drops instead of big bottle jumps) +1.7 (per-puff
dosing) = 28.4.

Why fewer people relapse during the hand taper (38.5 against 42.3): it keeps about 50 % more nicotine in each puff
over the plan (on average about 53 % of the starting strength against 34 % for Wane's plan), and many stop on a low
bottle instead of facing zero. It postpones the problem rather than solving it.

Caveats: the stall chances are assumptions, so the size of the gain is uncertain (5 to 25 in 100); its direction is
not. Wane users cannot get stuck in the simulation because the device lowers the strength itself; a real user could
stop using the device, which would count as a relapse here, and no dropout is modelled yet.
