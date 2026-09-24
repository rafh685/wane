"""Build the reproducible response calibration from the checked-in LSBU study data.

This script does not train a clinical predictor. The study has 20 experienced
adult vapers and no relapse labels. It calibrates a conservative guardrail for
how strongly Wane should react when a person's puff count and puff duration rise
after a nicotine reduction.

Run: .venv/bin/python fit_real_data.py
"""
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "lsbu" / "lsbu_compensation.csv"
OUT = ROOT / "engine_calibration.json"


def numeric(row, field):
    try:
        value = float(row[field])
        return value if np.isfinite(value) and value > 0 else np.nan
    except (KeyError, TypeError, ValueError):
        return np.nan


def summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "n": int(values.size),
        "median_ratio": round(float(np.median(values)), 6),
        "p25_ratio": round(float(np.quantile(values, 0.25)), 6),
        "p75_ratio": round(float(np.quantile(values, 0.75)), 6),
    }


def build_calibration(source=SOURCE):
    with Path(source).open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    puff_ratio, duration_ratio, total_ratio = [], [], []
    urge_delta, withdrawal_delta = [], []
    for row in rows:
        puff_6, puff_18 = numeric(row, "Puff_number_6F"), numeric(row, "Puff_number_18F")
        dur_6, dur_18 = numeric(row, "Puff_Duration_6F"), numeric(row, "Puff_Duration_18F")
        if all(np.isfinite(v) for v in (puff_6, puff_18, dur_6, dur_18)):
            puff_ratio.append(puff_6 / puff_18)
            duration_ratio.append(dur_6 / dur_18)
            total_ratio.append((puff_6 * dur_6) / (puff_18 * dur_18))
        urge_6, urge_18 = numeric(row, "UTVurges_6F"), numeric(row, "UTVurges_18f")
        mpss_6, mpss_18 = numeric(row, "MPSStotal_6F"), numeric(row, "MPSStotal_18f")
        if np.isfinite(urge_6) and np.isfinite(urge_18):
            urge_delta.append(urge_6 - urge_18)
        if np.isfinite(mpss_6) and np.isfinite(mpss_18):
            withdrawal_delta.append(mpss_6 - mpss_18)

    calibration = {
        "engine_version": "0.2.0-data-calibrated",
        "schema_version": 1,
        "source": {
            "dataset": "Dawkins et al. 2018 LSBU open data",
            "doi": "10.18744/LSBU.002952",
            "condition": "6 mg/ml versus 18 mg/ml, fixed-power condition",
            "participants": len(rows),
            "paired_puff_measurements": len(total_ratio),
        },
        "reference_dose_reduction": round(1 - 6 / 18, 6),
        "puff_count": summary(puff_ratio),
        "puff_duration": summary(duration_ratio),
        "total_puffing": summary(total_ratio),
        "self_report": {
            "urge_delta_median": round(float(np.median(urge_delta)), 6),
            "withdrawal_delta_median": round(float(np.median(withdrawal_delta)), 6),
        },
        "policy": {
            "high_pressure": 1.0,
            "immediate_pressure": 3.0,
            "calm_pressure": 0.35,
            "description": "A pressure of 1.0 equals the study's upper-quartile total-puffing response after scaling for the preceding dose reduction. The engine slows after two consecutive high responses or one response at 3.0.",
        },
        "limitations": [
            "The study is small and does not contain relapse outcomes.",
            "The response range is a runtime guardrail, not a clinical threshold or a treatment recommendation.",
            "The engine still needs a prospective adult Wane pilot with timestamped sessions to fit outcome risk from real use.",
        ],
    }
    return calibration


if __name__ == "__main__":
    result = build_calibration()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {OUT.name}: {result['source']['paired_puff_measurements']} paired puff measurements")
    print(f"Median total puffing ratio: {result['total_puffing']['median_ratio']:.3f}")
    print(f"Upper-quartile total puffing ratio: {result['total_puffing']['p75_ratio']:.3f}")
