"""Direct h-step linear predictors over a shared node space.

For node i, the prediction of x_i(t+h) is a ridge regression on lags 0..p-1 of the nodes in its source set S_i.
Coefficients are node-specific and shared across animals. Inputs from neurons absent in an animal are zero (the
z-scored mean); rows are used only when the target and the node's own history are valid.
"""
from __future__ import annotations

import numpy as np


def lagged(X: np.ndarray, M: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """(T, K, p) lag stack (lag 0 = current) and (T, K) validity of the full own history."""
    T, K = X.shape
    L = np.zeros((T, K, p)); V = np.ones((T, K), bool)
    for l in range(p):
        L[l:, :, l] = X[: T - l]
        V[:l] = False
        V[l:] &= M[: T - l]
    return L, V


class SourceSetRidge:
    """Node-specific ridge on lags of a source set S_i. Exact and fast: every S_i design is a column subset of the full
    (all nodes x lags + intercept) design, so per (animal, valid-row pattern) one full Gram is computed and sliced.
    ``fit`` stores per-node normal equations; ``solve(alpha)`` re-solves without touching the data."""

    def __init__(self, sources: list[np.ndarray], p: int = 5, alpha: float = 10.0, name: str = "ridge"):
        self.sources, self.p, self.alpha, self.name = sources, p, alpha, name
        self.coef: list[np.ndarray] = []

    def _cols(self, i):
        S = self.sources[i]
        return np.r_[(S[:, None] * self.p + np.arange(self.p)[None, :]).ravel(), -1]

    def _design(self, L, i):
        return np.concatenate([L[:, self.sources[i], :].reshape(len(L), -1), np.ones((len(L), 1))], 1)

    def fit(self, data: list[tuple[np.ndarray, np.ndarray]], h: int) -> "SourceSetRidge":
        K = data[0][0].shape[1]
        self.normal = []
        full = {}
        for a, (X, M) in enumerate(data):
            L, V = lagged(X, M, self.p)
            T = len(X)
            Dall = np.concatenate([L[: T - h].reshape(T - h, -1), np.ones((T - h, 1))], 1)
            Y = X[h:]
            pats = {}
            for i in range(K):
                ok = V[: T - h, i] & M[h:, i]
                if ok.any():
                    pats.setdefault(ok.tobytes(), (ok, []))[1].append(i)
            for ok, nodes in pats.values():
                D = Dall[ok]
                G = D.T @ D
                B = D.T @ Y[ok][:, nodes]
                for c, i in enumerate(nodes):
                    full.setdefault(i, []).append((G, B[:, c]))
        for i in range(K):
            if i not in full:
                self.normal.append(None); continue
            cols = self._cols(i)
            G = sum(g[np.ix_(cols, cols)] for g, _ in full[i])
            b = sum(bb[cols] for _, bb in full[i])
            self.normal.append((G, b))
        self.h = h
        return self.solve(self.alpha)

    def solve(self, alpha: float) -> "SourceSetRidge":
        self.alpha = alpha
        self.coef = []
        for nb in self.normal:
            if nb is None:
                self.coef.append(None); continue
            G, b = nb
            R = alpha * np.eye(G.shape[0]); R[-1, -1] = 0.0
            self.coef.append(np.linalg.solve(G + R, b))
        return self

    def predict(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """(T, K): row t holds the prediction for t + h (NaN where the own history is invalid or no model)."""
        L, V = lagged(X, M, self.p)
        P = np.full(X.shape, np.nan)
        for i, c in enumerate(self.coef):
            if c is None:
                continue
            ok = V[:, i]
            if ok.any():
                P[ok, i] = self._design(L[ok], i) @ c
        return P


class Persistence:
    name = "persistence"

    def fit(self, data, h):
        self.h = h
        return self

    def predict(self, X, M):
        return np.where(M, X, np.nan)
