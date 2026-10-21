"""Versioned, label-free preprocessing. Every transform is recorded on the Recording's provenance.

PREPROCESS_VERSION pins the whole chain used by the benchmarks; changing any step must bump it.
"""
from __future__ import annotations

import copy

import numpy as np

from .schema import Recording

PREPROCESS_VERSION = "gd-preprocess/1.0"


def standardize(rec: Recording) -> Recording:
    """Per-neuron z-score using that recording's own valid samples (label-free, transductive within the animal)."""
    r = copy.copy(rec)
    X = np.where(rec.mask, rec.neural, np.nan)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    r.mask = rec.mask & np.isfinite(Z)
    r.neural = np.where(r.mask, Z, 0.0)
    r.provenance = dict(rec.provenance, preprocess=rec.provenance.get("preprocess", []) + ["standardize"])
    return r


def resample(rec: Recording, dt: float) -> Recording:
    """Linear interpolation of neural and continuous behaviour onto a uniform grid; categorical channels (names
    ending in _code or _events) use nearest-sample. A sample is valid only if both bracketing samples were valid."""
    t = np.arange(rec.t[0], rec.t[-1] + 1e-9, dt)
    j = np.clip(np.searchsorted(rec.t, t), 1, len(rec.t) - 1)
    w = (t - rec.t[j - 1]) / (rec.t[j] - rec.t[j - 1])
    X = rec.neural[j - 1] * (1 - w[:, None]) + rec.neural[j] * w[:, None]
    M = rec.mask[j - 1] & rec.mask[j]
    near = np.where(w < 0.5, j - 1, j)
    beh = {}
    for k, v in rec.behavior.items():
        beh[k] = v[near] if (k.endswith("_code") or k.endswith("_events")) else v[j - 1] * (1 - w) + v[j] * w
    r = copy.copy(rec)
    r.t, r.neural, r.mask, r.behavior = t, np.where(M, X, 0.0), M, beh
    r.provenance = dict(rec.provenance, preprocess=rec.provenance.get("preprocess", []) + [f"resample(dt={dt})"])
    return r


def node_space(recs: list[Recording], min_frac: float = 0.5) -> list[str]:
    """Canonical neurons identified in at least ``min_frac`` of recordings (fixed node set for cross-animal models)."""
    cnt: dict[str, int] = {}
    for r in recs:
        for n in r.canonical_index():
            cnt[n] = cnt.get(n, 0) + 1
    return sorted(n for n, c in cnt.items() if c >= min_frac * len(recs))


def to_nodes(rec: Recording, nodes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(T, K) activity and validity mask in the shared node order; absent neurons are all-invalid."""
    ci = rec.canonical_index()
    T = rec.neural.shape[0]
    X = np.zeros((T, len(nodes)))
    M = np.zeros((T, len(nodes)), bool)
    for k, n in enumerate(nodes):
        if n in ci:
            X[:, k] = rec.neural[:, ci[n]]
            M[:, k] = rec.mask[:, ci[n]]
    return X, M
