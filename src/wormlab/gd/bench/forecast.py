"""A2: neural next-state forecasting on held-out animals.

Score per test animal: pooled R^2 over valid (t, node) targets, 1 - SSE / SST with SST around each node's mean in
that animal, and skill vs persistence (1 - SSE / SSE_persistence) on the identical entries. Aggregate over animals
with an animal bootstrap; model comparisons are paired by animal.
"""
from __future__ import annotations

import numpy as np


def score_animal(X, M, P, Pp, h):
    T = len(X)
    y, yh, yp = X[h:], P[: T - h], Pp[: T - h]
    ok = M[h:] & np.isfinite(yh) & np.isfinite(yp)
    if ok.sum() < 50:
        return None
    mu = np.array([y[ok[:, k], k].mean() if ok[:, k].any() else 0.0 for k in range(y.shape[1])])
    sse = float(((y - yh) ** 2)[ok].sum())
    sst = float(((y - mu) ** 2)[ok].sum())
    ssp = float(((y - yp) ** 2)[ok].sum())
    return dict(r2=1 - sse / sst, skill_vs_persistence=1 - sse / ssp, n=int(ok.sum()))


def boot_mean(vals, n=2000, seed=0):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    rng = np.random.default_rng(seed)
    b = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return dict(mean=float(v.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], n_animals=len(v))
