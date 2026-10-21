"""Unified interventions for Graph Dynamics models.

Every intervention declares an estimand, validates against the node space and graph before execution (invalid
requests raise ``InvalidIntervention``), and has a content hash. Two application points:
  * graph-level: ``apply_graph(chem, gap) -> (chem, gap)`` on the typed adjacency a model is built from;
  * state-level: ``apply_state(x) -> x`` on node activity at every simulated / predicted step.
Conventions follow ``gd.graph``: chem[i, j] = synapses j -> i; gap symmetric.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from . import graph as G


class InvalidIntervention(ValueError):
    pass


@dataclass(frozen=True)
class Intervention:
    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps({"type": type(self).__name__, **asdict(self)}, sort_keys=True, default=str).encode()).hexdigest()

    estimand: str = field(default="", init=False)

    def validate(self, nodes: list[str], chem=None, gap=None) -> "Intervention":
        return self

    def apply_graph(self, chem: np.ndarray, gap: np.ndarray, nodes: list[str]):
        return chem, gap

    def apply_state(self, x: np.ndarray, nodes: list[str]) -> np.ndarray:
        return x


def _idx(nodes, names):
    missing = [n for n in names if n not in nodes]
    if missing:
        raise InvalidIntervention(f"neurons not in node space: {missing}")
    return [nodes.index(n) for n in names]


@dataclass(frozen=True)
class NodeClamp(Intervention):
    """Hold neurons at a fixed (z-scored) value: removes their fluctuations, keeps their edges."""
    neurons: tuple[str, ...]
    value: float = 0.0
    estimand = "effect on other neurons of removing the clamped neurons' fluctuations (edges intact)"

    def validate(self, nodes, chem=None, gap=None):
        _idx(nodes, self.neurons); return self

    def apply_state(self, x, nodes):
        x = x.copy(); x[..., _idx(nodes, self.neurons)] = self.value; return x


@dataclass(frozen=True)
class NodeAblate(Intervention):
    """Remove neurons: delete all their edges and clamp their activity to the mean."""
    neurons: tuple[str, ...]
    estimand = "effect of deleting the neurons from the network (edges and activity)"

    def validate(self, nodes, chem=None, gap=None):
        _idx(nodes, self.neurons); return self

    def apply_graph(self, chem, gap, nodes):
        ix = _idx(nodes, self.neurons); chem, gap = chem.copy(), gap.copy()
        chem[ix, :] = chem[:, ix] = 0.0; gap[ix, :] = gap[:, ix] = 0.0
        return chem, gap

    def apply_state(self, x, nodes):
        x = x.copy(); x[..., _idx(nodes, self.neurons)] = 0.0; return x


@dataclass(frozen=True)
class EdgeDelete(Intervention):
    """Delete specific edges: (pre, post, 'chem') or (a, b, 'gap')."""
    edges: tuple[tuple[str, str, str], ...]
    estimand = "effect of deleting the listed synapses"

    def validate(self, nodes, chem=None, gap=None):
        for pre, post, kind in self.edges:
            j, i = _idx(nodes, [pre, post])
            if kind not in ("chem", "gap"):
                raise InvalidIntervention(f"edge type {kind!r}")
            W = chem if kind == "chem" else gap
            if W is not None and W[i, j] <= 0:
                raise InvalidIntervention(f"no {kind} edge {pre} -> {post} to delete")
        return self

    def apply_graph(self, chem, gap, nodes):
        chem, gap = chem.copy(), gap.copy()
        for pre, post, kind in self.edges:
            j, i = _idx(nodes, [pre, post])
            if kind == "chem":
                chem[i, j] = 0.0
            else:
                gap[i, j] = gap[j, i] = 0.0
        return chem, gap


@dataclass(frozen=True)
class EdgeScale(Intervention):
    """Scale all edges of one type (e.g. 'gap' x 0 = innexin loss in the model)."""
    kind: str
    factor: float
    estimand = "effect of scaling every edge of one synapse type"

    def validate(self, nodes, chem=None, gap=None):
        if self.kind not in ("chem", "gap") or self.factor < 0:
            raise InvalidIntervention(f"bad EdgeScale({self.kind}, {self.factor})")
        return self

    def apply_graph(self, chem, gap, nodes):
        return (chem * self.factor, gap) if self.kind == "chem" else (chem, gap * self.factor)


@dataclass(frozen=True)
class DegreePreservingShuffle(Intervention):
    """Null graph: degree-preserving rewiring of the full 300-neuron connectome, then restriction to the node space."""
    seed: int
    estimand = "null: connectome replaced by a degree-preserving random graph"

    def apply_graph(self, chem, gap, nodes):
        S = G.shuffle_degree_preserving(G.load_connectome(), self.seed).sub(nodes)
        return S.chem, S.gap
