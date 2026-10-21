"""Typed connectome graph (from the installed ``wormneuroatlas`` package) and topology-preserving nulls.

Convention: ``chem[i, j]`` = synapse count from j to i (the package convention); ``gap`` is symmetrised.
The default atlas connectome sums White 1986 and Witvliet 2020 (as shipped); recorded in ``source``.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import h5py
import numpy as np

from .identity import canonical_names


@dataclass(frozen=True)
class Connectome:
    names: tuple[str, ...]
    chem: np.ndarray        # (300, 300) counts, row = postsynaptic
    gap: np.ndarray         # (300, 300) symmetric counts
    source: str

    def sub(self, nodes: list[str]) -> "Connectome":
        ix = [self.names.index(n) for n in nodes]
        return Connectome(tuple(nodes), self.chem[np.ix_(ix, ix)], self.gap[np.ix_(ix, ix)], self.source)

    def neighbors(self, kind: str = "any") -> list[np.ndarray]:
        """Per node i, indices j with an edge j->i (chem), j-i (gap) or either; self excluded."""
        A = {"chem_in": self.chem > 0, "gap": self.gap > 0}.get(kind)
        if A is None:
            A = (self.chem > 0) | (self.chem.T > 0) | (self.gap > 0)
        A = A.copy(); np.fill_diagonal(A, False)
        return [np.where(A[i])[0] for i in range(A.shape[0])]


def load_connectome() -> Connectome:
    base = resources.files("wormneuroatlas") / "data"
    with resources.as_file(base / "aconnectome_default.h5") as p, h5py.File(p) as h:
        chem, gap = h["chem"][:].astype(float), h["gap"][:].astype(float)
    gap = np.maximum(gap, gap.T)
    np.fill_diagonal(chem, 0.0)
    np.fill_diagonal(gap, 0.0)
    assert chem.shape == (len(canonical_names()),) * 2
    return Connectome(canonical_names(), chem, gap, "wormneuroatlas aconnectome_default (White 1986 + Witvliet 2020, as shipped; autapses/self-gap entries zeroed)")


def shuffle_degree_preserving(C: Connectome, seed: int, n_swaps_per_edge: int = 10) -> Connectome:
    """Directed double-edge swaps on chem (preserves every in- and out-degree) and undirected swaps on gap (preserves
    degree); edge weights travel with their edges. Self-loops and duplicate edges are rejected."""
    rng = np.random.default_rng(seed)

    def swap_directed(W):
        W = W.copy()
        E = np.argwhere(W > 0)                    # (post, pre)
        w = W[E[:, 0], E[:, 1]].copy()
        A = W > 0
        for _ in range(n_swaps_per_edge * len(E)):
            a, b = rng.integers(len(E), size=2)
            (i1, j1), (i2, j2) = E[a], E[b]
            if len({i1, j1, i2, j2}) < 4 or A[i2, j1] or A[i1, j2]:
                continue
            A[i1, j1] = A[i2, j2] = False; A[i2, j1] = A[i1, j2] = True
            E[a], E[b] = (i2, j1), (i1, j2)
        out = np.zeros_like(W); out[E[:, 0], E[:, 1]] = w
        return out

    def swap_undirected(W):
        U = np.triu(W, 1)
        E = np.argwhere(U > 0); w = U[E[:, 0], E[:, 1]].copy()
        A = (W > 0)
        for _ in range(n_swaps_per_edge * len(E)):
            a, b = rng.integers(len(E), size=2)
            (i1, j1), (i2, j2) = E[a], E[b]
            if rng.random() < 0.5:
                i2, j2 = j2, i2
            if len({i1, j1, i2, j2}) < 4 or A[i1, j2] or A[i2, j1]:
                continue
            A[i1, j1] = A[j1, i1] = A[i2, j2] = A[j2, i2] = False
            A[i1, j2] = A[j2, i1] = A[i2, j1] = A[j1, i2] = True
            E[a], E[b] = sorted((i1, j2)), sorted((i2, j1))
        out = np.zeros_like(W); out[E[:, 0], E[:, 1]] = w
        return np.maximum(out, out.T)

    return Connectome(C.names, swap_directed(C.chem), swap_undirected(C.gap), C.source + f" | degree-preserving shuffle seed={seed}")
