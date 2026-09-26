"""Train the FROZEN base of the per-puff network on synthetic population A. SIMULATION ONLY.

Data: puffsim people from population A (seed 21, never used for testing), tapered with the shared slow layer,
while an exploring fast policy varies the dose per puff (flat, capped, and randomly scaled days) so the network
sees many dose patterns. At each puff we store the device-only features and, at the end of the device day,
the real answer: how many puffs followed (log1p of puffs from this one to 05:00).

Output: puff_nn_base.json (base layers + the population head). Base layers are frozen from then on; only the
head adapts per user, online (puff_nn.AdaptiveHead).
Run: python3 train_puff_nn.py
"""
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from population import population
from puff_nn import FEATURES, WEIGHTS_FILE, DeviceState, baseline_with_level
from puffsim import SlowSchedule, simulate

N_PEOPLE, SEED, SUBSAMPLE = 60, 21, 4


class Explorer:
    """Behaviour policy for data collection; records features and labels through the device's own state."""
    name = "explorer"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.rows, self.day_rows = [], []

    def begin(self, baseline):
        self.state = DeviceState(baseline_with_level(baseline))
        for t, dur in baseline["puffs"]:
            self.state.observe(t, dur, 1.0)

    def start_day(self, k, u, weekend):
        n = len(self.state.day_times)
        for i, x in self.day_rows:
            self.rows.append((x, math.log1p(n - i), math.log1p(self._habit_rest[i])))
        self.day_rows, self._habit_rest = [], {}
        self.mode = self.rng.choice(["flat", "capped", "scaled"], p=[0.4, 0.3, 0.3])
        self.state.start_day(k, u, weekend)

    def dose(self, t):
        s = self.state
        s.advance(t)
        i = len(s.day_times)
        if i % SUBSAMPLE == 0:
            self.day_rows.append((i, s.features(t)))
            self._habit_rest[i] = s.habit(t)[1]
        left = max(0.0, s.budget - s.used)
        if self.mode == "flat":
            return s.u
        d = s.u if self.mode == "capped" else min(1.0, s.u * self.rng.uniform(0.4, 1.8))
        return min(d, left)

    def observe(self, t, dur, dose):
        self.state.observe(t, dur, dose)


def collect(args):
    i, profile = args
    ex = Explorer(1000 + i)
    ex._habit_rest = {}
    simulate(profile, lambda: ex, SlowSchedule(adaptive=bool(i % 2)), taper_days=154, follow_days=0, seed=500 + i)
    return i, ex.rows


def train(X, y, Xv, yv, hidden=(32, 16), epochs=12, lr=2e-3, batch=1024, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z, Zv = (X - mu) / sd, (Xv - mu) / sd
    sizes = [X.shape[1], *hidden, 1]
    P = []
    for a, b in zip(sizes, sizes[1:]):
        P += [rng.normal(0, 1 / math.sqrt(a), (a, b)), np.zeros(b)]
    M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    step = 0

    def forward(Z):
        h1 = np.tanh(Z @ P[0] + P[1]); h2 = np.tanh(h1 @ P[2] + P[3])
        return h1, h2, (h2 @ P[4] + P[5])[:, 0]

    for ep in range(epochs):
        order = rng.permutation(len(Z))
        for j in range(0, len(Z), batch):
            idx = order[j:j + batch]
            z, t = Z[idx], y[idx]
            h1, h2, out = forward(z)
            g = 2 * (out - t)[:, None] / len(idx)
            gW3, gb3 = h2.T @ g, g.sum(0)
            d2 = (g @ P[4].T) * (1 - h2 ** 2)
            gW2, gb2 = h1.T @ d2, d2.sum(0)
            d1 = (d2 @ P[2].T) * (1 - h1 ** 2)
            gW1, gb1 = z.T @ d1, d1.sum(0)
            step += 1
            for k, gk in enumerate([gW1, gb1, gW2, gb2, gW3, gb3]):
                M[k] = 0.9 * M[k] + 0.1 * gk; V[k] = 0.999 * V[k] + 0.001 * gk ** 2
                P[k] -= lr * (M[k] / (1 - 0.9 ** step)) / (np.sqrt(V[k] / (1 - 0.999 ** step)) + 1e-8)
        mse = float(np.mean((forward(Zv)[2] - yv) ** 2))
        print(f"epoch {ep + 1}: held-out MSE {mse:.4f}")
    return dict(mu=mu.tolist(), sd=sd.tolist(), W1=P[0].tolist(), b1=P[1].tolist(), W2=P[2].tolist(),
                b2=P[3].tolist(), head=np.append(P[4][:, 0], P[5]).tolist()), forward(Zv)[2]


if __name__ == "__main__":
    people = population(N_PEOPLE, SEED)
    with ProcessPoolExecutor() as pool:
        results = dict(pool.map(collect, list(enumerate(people))))
    val_ids = set(range(0, N_PEOPLE, 5))                       # every 5th person held out
    split = lambda ids: [r for i in ids for r in results[i]]
    tr, va = split(i for i in results if i not in val_ids), split(sorted(val_ids))
    X, y = np.array([r[0] for r in tr]), np.array([r[1] for r in tr])
    Xv, yv, hv = np.array([r[0] for r in va]), np.array([r[1] for r in va]), np.array([r[2] for r in va])
    print(f"{len(X)} training puffs, {len(Xv)} held-out puffs, {len(FEATURES)} features")
    params, pred = train(X, y, Xv, yv)
    var = float(np.var(yv))
    report = dict(
        data="synthetic population A (puffsim), seed 21; held-out = every 5th person",
        n_train=len(X), n_heldout=len(Xv), features=FEATURES,
        heldout_r2_network=1 - float(np.mean((pred - yv) ** 2)) / var,
        heldout_r2_habit_only=1 - float(np.mean((hv - yv) ** 2)) / var,
    )
    params.update(feature_names=FEATURES, report=report)
    WEIGHTS_FILE.write_text(json.dumps(params) + "\n")
    print(json.dumps(report, indent=2))
