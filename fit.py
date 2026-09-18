"""Fit the engine's weights on synthetic population A, choose thresholds on A, evaluate on population B.

Two models, both from features the engine can measure (puff timestamps only):
  craving model : linear regression   features -> the person's true 7-day mean craving (the self-report replacement)
  risk model    : logistic regression features -> relapse within the next 7 days
Then the decision thresholds (HOLD / HALF CUT) are chosen on A to maximise the share of people OFF nicotine
at 30 weeks, which prices speed and safety in one number. Weights go to engine_weights.json.

Run: .venv/bin/python fit.py
"""
import json, sys, time
import numpy as np, pandas as pd
from profiles import Vaper, START_MG
from population import population
from engine import FixedTaper, AdaptiveTaper, FittedTaper, features, FEATURE_NAMES, feature_vector, make_baseline
from simulate import run_one, per_run

WEEKS = 30          # training horizon
WEEKS_EVAL = 40     # evaluation horizon: long enough for a slow but safe taper to finish
N_A, N_B = 120, 120


# ---------------- 1. collect training rows under a data-collection policy ----------------

def collect(profiles, rates=(0.08, 0.12, 0.16), seeds=(0, 1)):
    """Run each person under fixed tapers at several rates. At every weekly decision point record the engine's
    features and the labels: true mean craving over the window, and whether they relapse in the next 7 days."""
    X, y_crave, y_relapse = [], [], []
    for p in profiles:
        for rate in rates:
            for s in seeds:
                rng = np.random.default_rng(s); v = Vaper(p, rng); eng = FixedTaper(rate, 7)
                dose, rows, base = START_MG, [], None
                while v.t < WEEKS * 7 and not v.relapsed and dose > 0:
                    for _ in range(7):
                        rows.append(v.day(dose))
                        if v.relapsed: break
                    if v.relapsed: break
                    window, prev = rows[-7:], (rows[-14:-7] if len(rows) >= 14 else [])
                    if base is None: base = make_baseline(window)
                    f = features(window, prev, dose=dose, baseline=base)
                    # label: what happens in the NEXT 7 days at the dose the engine is about to set
                    nxt = eng.next_dose(dose, window, prev)
                    probe_v = _clone(v); rel = False
                    for _ in range(7):
                        r = probe_v.day(nxt)
                        if r["relapsed"]: rel = True; break
                    X.append(feature_vector(f)); y_crave.append(np.mean([r["craving"] for r in window])); y_relapse.append(rel)
                    dose = nxt
    return np.array(X), np.array(y_crave), np.array(y_relapse, dtype=float)


def _clone(v):
    import copy
    c = copy.copy(v); c.rng = np.random.default_rng(int(v.rng.integers(1 << 30))); return c


# ---------------- 2. fit ----------------

def fit_linear(X, y, l2=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    A = Z.T @ Z + l2 * np.eye(Z.shape[1]); A[0, 0] -= l2
    w = np.linalg.solve(A, Z.T @ y)
    return dict(mu=mu.tolist(), sd=sd.tolist(), w=w.tolist())


def fit_logistic(X, y, l2=1.0, iters=50):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):                     # Newton / IRLS
        p = 1 / (1 + np.exp(-Z @ w)); W = p * (1 - p)
        g = Z.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (Z * W[:, None]).T @ Z + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H, g)
    return dict(mu=mu.tolist(), sd=sd.tolist(), w=w.tolist())


# ---------------- 3. evaluate ----------------

def evaluate(profiles, engine_factory, seeds=(0, 1), weeks=WEEKS_EVAL):
    frames = []
    for p in profiles:
        for s in seeds:
            df, _ = run_one(p, engine_factory, weeks, seed=s, weekly_cut=0.12, period=7); frames.append(df)
    r = per_run(pd.concat(frames))
    off = r.reached_zero.mean(); rel = r.relapsed.mean()
    wtz = r.weeks_to_zero[r.reached_zero].median()
    return dict(relapse=rel, off=off, weeks_to_zero=wtz, n=len(r))


if __name__ == "__main__":
    t0 = time.time()
    A, B = population(N_A, seed=1), population(N_B, seed=2)
    X, yc, yr = collect(A)
    print(f"training rows: {len(X)}  (relapse-next-week rate {yr.mean():.1%})  {time.time()-t0:.0f}s")
    crave_m, risk_m = fit_linear(X, yc), fit_logistic(X, yr)
    # in-sample fit quality
    Zc = (X - crave_m["mu"]) / crave_m["sd"]; pred_c = crave_m["w"][0] + Zc @ np.array(crave_m["w"][1:])
    print(f"craving model: R^2 = {1 - ((yc - pred_c) ** 2).sum() / ((yc - yc.mean()) ** 2).sum():.2f}")
    Zr = (X - risk_m["mu"]) / risk_m["sd"]; pr = 1 / (1 + np.exp(-(risk_m["w"][0] + Zr @ np.array(risk_m["w"][1:]))))
    order = np.argsort(pr); ranks = np.empty(len(pr)); ranks[order] = np.arange(len(pr))
    auc = (ranks[yr == 1].sum() - (yr == 1).sum() * ((yr == 1).sum() - 1) / 2) / ((yr == 1).sum() * (yr == 0).sum())
    print(f"risk model:    AUC = {auc:.2f}")
    print("\nlearned weights (standardised features):")
    for name, wc, wr in zip(FEATURE_NAMES, crave_m["w"][1:], risk_m["w"][1:]):
        print(f"  {name:14} craving {wc:+6.2f}   risk {wr:+6.2f}")

    # ---- risk budget chosen on A: maximise off - relapse at 40 weeks ----
    best = None
    print(f"\nrisk-budget search on population A, {WEEKS_EVAL} weeks. Objective = off - relapse "
          "(take the largest cut whose predicted risk next week stays under the budget):")
    for budget in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
        weights = dict(craving=crave_m, risk=risk_m, budget=budget)
        res = evaluate(A, lambda cut, period: FittedTaper(cut, period, weights=weights), seeds=(3,))
        score = res["off"] - res["relapse"]
        print(f"  budget {budget:.3f}/week  ->  relapse {res['relapse']:.0%}  off {res['off']:.0%}  weeks to zero {res['weeks_to_zero']:.0f}   score {score:+.2f}")
        if best is None or score > best[1]:
            best = (weights, score)
    weights = best[0]
    json.dump(weights, open("engine_weights.json", "w"), indent=1)
    print(f"\nchosen: budget {weights['budget']:.3f} per week  -> engine_weights.json")

    # ---- the honest test: population B, never seen ----
    print(f"\npopulation B ({N_B} people x 2 seeds, {WEEKS_EVAL} weeks, 12 %/week):")
    for label, fac in (("Fixed taper", FixedTaper), ("Wane, hand-set weights", AdaptiveTaper),
                       ("Wane, fitted weights", lambda cut, period: FittedTaper(cut, period, weights=weights))):
        res = evaluate(B, fac)
        print(f"  {label:24} relapse {res['relapse']:4.0%}   off nicotine {res['off']:4.0%}   median weeks to zero {res['weeks_to_zero']:.0f}")

    # ---- and the five named people the demo shows ----
    from profiles import PROFILES
    print(f"\nthe five named profiles (40 seeds each, {WEEKS_EVAL} weeks, 12 %/week):")
    for label, fac in (("Fixed taper", FixedTaper), ("Wane, hand-set weights", AdaptiveTaper),
                       ("Wane, fitted weights", lambda cut, period: FittedTaper(cut, period, weights=weights))):
        res = evaluate(PROFILES, fac, seeds=tuple(range(40)))
        per = {}
        for pr in PROFILES:
            rr = evaluate([pr], fac, seeds=tuple(range(40)))
            per[pr.name] = f"{rr['relapse']:.0%}/{rr['off']:.0%}"
        print(f"  {label:24} relapse {res['relapse']:4.0%}   off {res['off']:4.0%}   weeks to zero {res['weeks_to_zero']:.0f}   " + "  ".join(f"{k[:3]} {v}" for k, v in per.items()))
    print(f"\n{time.time()-t0:.0f}s")
