"""Versioned evidence graph with dependence-aware evidence-family collapsing.

Nodes: dataset, partition, animal-set, observation, test_outcome, claim, hypothesis, control, synthetic_null,
capsule. Edges use a fixed relation vocabulary. Two outcomes belong to the same EVIDENCE FAMILY when they are
not independent: they share a statistic family on overlapping units in the same data stream, or are linked by
derived_from / shares_data_with / not_independent_of / replicates-on-same-units. Families are computed by
union-find, so "8 analyses support H5" can be reported as "8 analyses, 3 independent families".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .spec import Call, PredictionKind, TestOutcome, canonical_sha

RELATIONS = frozenset({
    "supports", "contradicts", "tests", "depends_on", "derived_from", "alternative_explanation_for",
    "replicates", "shares_data_with", "shares_feature_family_with", "not_independent_of",
    "supersedes_interpretation", "requires", "unresolved_between",
})
DEPENDENCE_RELATIONS = frozenset({"derived_from", "shares_data_with", "not_independent_of", "shares_feature_family_with"})
NODE_KINDS = frozenset({"dataset", "partition", "animals", "observation", "test_outcome", "claim", "hypothesis",
                        "control", "synthetic_null", "capsule", "test_spec"})


@dataclass
class EvidenceGraph:
    version: str = "evidence_graph/1.0"
    nodes: dict = field(default_factory=dict)     # id -> {"kind":..., **attrs}
    edges: list = field(default_factory=list)     # {"src","rel","dst", **attrs}

    def add_node(self, nid: str, kind: str, **attrs) -> str:
        if kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind {kind!r}")
        if nid in self.nodes and self.nodes[nid]["kind"] != kind:
            raise ValueError(f"node {nid} kind conflict")
        self.nodes.setdefault(nid, {"kind": kind})
        self.nodes[nid].update(attrs)
        return nid

    def add_edge(self, src: str, rel: str, dst: str, **attrs) -> None:
        if rel not in RELATIONS:
            raise ValueError(f"unknown relation {rel!r}")
        for n in (src, dst):
            if n not in self.nodes:
                raise ValueError(f"edge endpoint {n} not a node")
        e = {"src": src, "rel": rel, "dst": dst, **attrs}
        if e not in self.edges:
            self.edges.append(e)

    # ---- outcomes --------------------------------------------------------------------------------
    def add_outcome(self, o: TestOutcome, *, data_stream: str) -> str:
        """Register an outcome + its data dependencies. ``data_stream`` names the measurement pipeline
        (e.g. 'raw_traces/1.0', 'dandi_base_signals', 'synthetic:S1')."""
        o.validate()
        oid = self.add_node(o.outcome_id, "test_outcome", test_id=o.test_id, stage=o.stage, partition=o.partition,
                            status=o.status, call=o.call, estimate=o.estimate, ci=o.ci, n_units=o.n_units,
                            feature_family=o.feature_family, contamination_grade=o.contamination_grade,
                            data_stream=data_stream, units=sorted(o.unit_ids))
        ds = self.add_node(f"stream:{data_stream}", "dataset")
        pt = self.add_node(f"partition:{o.partition}", "partition")
        self.add_edge(oid, "depends_on", ds)
        self.add_edge(oid, "depends_on", pt)
        for d in o.derived_from:
            if d in self.nodes:
                self.add_edge(oid, "derived_from", d)
        return oid

    def relate_hypothesis(self, outcome_id: str, hypothesis_id: str, relation: str, *, kind: str, grade: str) -> None:
        self.add_node(hypothesis_id, "hypothesis")
        self.add_edge(outcome_id, relation, hypothesis_id, prediction_kind=kind, grade=grade)

    # ---- families ------------------------------------------------------------------------------------
    def families(self, strict: bool = False) -> dict[str, str]:
        """outcome_id -> family representative id (union-find over dependence).
        strict (policy 1.3): additionally unite ANY two outcomes on the same partition and data stream with overlapping
        units, whatever their statistic family (they share animals, events and response variables)."""
        outs = [n for n, a in self.nodes.items() if a["kind"] == "test_outcome"]
        parent = {n: n for n in outs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for i, a in enumerate(outs):
            na = self.nodes[a]
            for b in outs[i + 1:]:
                nb = self.nodes[b]
                same_stream = na.get("data_stream") == nb.get("data_stream")
                overlap = bool(set(na.get("units", [])) & set(nb.get("units", [])))
                if na.get("feature_family") == nb.get("feature_family") and overlap:
                    if strict and "HISTORICAL" in (na.get("stage"), nb.get("stage")) and na.get("partition") != nb.get("partition"):
                        continue      # strict: an all-animal historical look must not bridge disjoint held-out partitions
                    union(a, b)
                elif same_stream and overlap and na.get("partition") == nb.get("partition") and na.get("test_id") == nb.get("test_id"):
                    union(a, b)
                elif strict and same_stream and overlap and na.get("partition") == nb.get("partition"):
                    union(a, b)
        for e in self.edges:
            if e["rel"] in DEPENDENCE_RELATIONS and e["src"] in parent and e["dst"] in parent:
                union(e["src"], e["dst"])
        return {o: find(o) for o in outs}

    def support_summary(self) -> dict:
        fam = self.families()
        out = {}
        for e in self.edges:
            if e["rel"] not in ("supports", "contradicts"):
                continue
            h = e["dst"]
            rec = out.setdefault(h, {"supports": {"analyses": 0, "families": set()}, "contradicts": {"analyses": 0, "families": set()}})
            rec[e["rel"]]["analyses"] += 1
            rec[e["rel"]]["families"].add(fam.get(e["src"], e["src"]))
        return {h: {r: {"analyses": v["analyses"], "independent_families": len(v["families"])} for r, v in rec.items()} for h, rec in out.items()}

    def content(self) -> dict:
        return {"version": self.version, "nodes": {k: self.nodes[k] for k in sorted(self.nodes)},
                "edges": sorted(self.edges, key=lambda e: (e["src"], e["rel"], e["dst"])), "families": self.families()}

    @property
    def graph_hash(self) -> str:
        return canonical_sha(self.content())


def relation_for(prediction, observed_call: str) -> str | None:
    """Map (prediction, observed call) to supports / contradicts / None."""
    if prediction is None or observed_call not in (Call.POS, Call.NULL, Call.NEG):
        return None
    if prediction.kind == PredictionKind.NON_DISCRIMINATING_OBSERVATION or prediction.expected == Call.ANY:
        return None
    if observed_call in prediction.counts_against_when():
        return "contradicts"
    if prediction.kind == PredictionKind.FALSIFYING_OBSERVATION:
        return None
    if observed_call == prediction.expected:
        return "supports"
    return None
