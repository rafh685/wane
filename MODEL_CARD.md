# Wane adaptive taper engine

## Version

`0.2.0-data-calibrated` · 24 September 2026

## GPT version used for this release

**GPT-5 (Codex)** was used to implement the data-calibration pipeline, the runtime response guardrail and this documentation. The model did not generate participant data, labels or clinical outcome claims. Weight calculation is deterministic Python in `fit_real_data.py`.

## What changed

The engine already adapted its taper rate from simulated risk and inferred craving. This release adds a separate, real-data-calibrated response signal:

1. After each dose reduction, it measures the person's change in total puffing (`puff count × puff duration`).
2. It scales that response by the size of the reduction.
3. If the scaled response stays at or above the upper-quartile response observed in the paired adult study for two consecutive decision windows, it slows the next taper rate. One extreme response triggers the same brake.
4. It only speeds up after two calm, low-compensation decision windows.

The policy remains monotonic: it can cut, half cut or hold. It never increases nicotine concentration.

## Training and calibration data included in this repository

`data/lsbu/lsbu_compensation.csv` is the open dataset associated with Dawkins et al. (2018), *Real-world compensatory behaviour with low nicotine concentration e-liquid*, DOI `10.18744/LSBU.002952`.

- 20 experienced adult vapers
- 19 valid pairs for the fixed-power puff-count and puff-duration comparison
- 18 mg/ml versus 6 mg/ml conditions
- Data used here: puff count, puff duration, urge and withdrawal summaries

The generated `engine_calibration.json` records the exact medians and quartiles. Rebuild it with:

```bash
.venv/bin/python fit_real_data.py
```

## Evidence boundary

The real dataset measures short-term compensatory behaviour. It does not include a taper programme, long-term adherence, relapse outcomes, Wane hardware or a representative user population. Therefore:

- the response calibration is a conservative behavioural guardrail;
- the fitted relapse model remains synthetic and is marked as such in the code;
- a prospective study with consenting adult Wane users is required before using outcome predictions as product decisions.

## Intended use

Research and product-prototype simulation for adult nicotine-use reduction. This is not a clinical decision system or medical advice.
