# Wane — adaptive nicotine tapering

Every quit product tapers how often you vape. Wane tapers how much nicotine is in each puff — continuously, from measured behaviour.

This repo is the **taper engine** and a simulation demo: five synthetic vapers run through a traditional fixed taper and through the Wane engine, side by side.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas streamlit plotly
.venv/bin/streamlit run app.py
```

Or the numbers only: `.venv/bin/python simulate.py`

## Files

| File | What |
|---|---|
| `profiles.py` | One behavioural model, five parameter sets. Compensation elasticity, craving sensitivity, withdrawal curve, relapse threshold, noise. Nothing scripted per week. |
| `engine.py` | `FixedTaper` (cut x % every week) and `AdaptiveTaper` (the JITAI loop: features from puff data → risk + derived craving → cut / half / hold; per-person rate that learns). Dose arithmetic is rules and monotonic — the engine can only lower or hold. |
| `simulate.py` | Same people, both engines, N runs with noise. |
| `app.py` | Streamlit demo: week slider, fixed vs Wane with 10–90 % bands, zoom on one person with every decision and the features behind it. |

## Honest status

Profiles are calibrated to published population ranges (puff topography, compensation in the light-cigarette literature, withdrawal time-course). They are not validated individuals. A supervised pilot (adaptive arm vs fixed arm) is what tests the thesis.

Built for IE University's Tech Venture Bootcamp, Sept–Oct 2026.
