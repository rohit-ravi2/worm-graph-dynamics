"""Regression tests for the Deep Mechanism Campaign hostile adjudicator audit (policy 1.3).

Each test pins one weakness found in policies 1.0-1.2 (docs/mechanism_engine/deep_campaign/ADJUDICATOR_AUDIT.md) and
checks that the older policies still behave as they did (verdicts are never rewritten; they are reinterpreted)."""
from __future__ import annotations

import random

from wormlab.mechanism import adjudicator as A
from wormlab.mechanism.evidence_graph import EvidenceGraph
from wormlab.mechanism.spec import Call, HypothesisSpec, OutcomeStatus, Prediction, Stage, TestOutcome

N, S, ND = "NECESSARY_PREDICTION", "SUPPORTIVE_PREDICTION", "NON_DISCRIMINATING_OBSERVATION"
PART = ["DISCOVERY", "CONFIRMATION", "RESERVE", "SYNTHETIC", "HISTORICAL"]


def H(hid, cls, preds):
    return HypothesisSpec(hypothesis_id=hid, generation=0, claim=hid, mechanism_class=cls, causal_entities=["x"], causal_relations=["x"],
                          required_observables=sorted({p.test_id for p in preds}), predicted_signatures=["x"], predicted_absences=["x"],
                          falsifiers=["x"], predictions=preds, known_confounds=[], alternative_hypotheses=[], identifiability_conditions=[],
                          required_controls=[], allowed_data_partitions=PART, forbidden_data_partitions=[])


def O(test, call, part, units, family=None, stage=None, grade="UNTOUCHED", details=None, stream="s1"):
    o = TestOutcome(test, "h", stage or (Stage.DISCOVERY if part == "DISCOVERY" else Stage.CONFIRMATION), part, OutcomeStatus.DECIDED, call,
                    0.1, [0.05, 0.15], 0.03, len(units), 100, list(units), family or test, grade, details=dict(details or {}, data_stream=stream)).finalize()
    return o


def graph(outs):
    g = EvidenceGraph()
    for o in outs:
        g.add_outcome(o, data_stream=o.details["data_stream"])
    return g


CONF_UNITS = [f"c{i}" for i in range(10)]
RES_UNITS = [f"r{i}" for i in range(10)]
DISC_UNITS = [f"d{i}" for i in range(10)]


def test_discovery_is_exploratory_under_13():
    o = O("T1", Call.POS, "DISCOVERY", DISC_UNITS)
    assert A.evidence_grade(o, policy="1.2") == A.PART
    assert A.evidence_grade(o, policy="1.3") == A.EXPL


def test_strict_families_collapse_same_partition_same_stream():
    a = O("T1", Call.POS, "CONFIRMATION", CONF_UNITS, family="f1")
    b = O("T2", Call.POS, "CONFIRMATION", CONF_UNITS, family="f2")
    c = O("T1", Call.POS, "RESERVE", RES_UNITS, family="f1")
    g = graph([a, b, c])
    loose, strict = g.families(), g.families(strict=True)
    assert loose[a.outcome_id] != loose[b.outcome_id]
    assert strict[a.outcome_id] == strict[b.outcome_id]
    assert strict[a.outcome_id] != strict[c.outcome_id]


def test_supported_requires_replication_across_partitions_under_13():
    h = H("H1.g0", "STATIC_MEASUREMENT_CONTAMINATION", [Prediction("T1", "POS", N), Prediction("T2", "POS", N)])
    outs = [O("T1", Call.POS, "CONFIRMATION", CONF_UNITS), O("T2", Call.POS, "CONFIRMATION", CONF_UNITS)]
    g = graph(outs)
    assert A.assess(h, outs, g.families(), policy="1.2").state == "SUPPORTED"
    assert A.assess(h, outs, g.families(strict=True), policy="1.3").state == "WEAKLY_SUPPORTED"
    outs2 = outs + [O("T1", Call.POS, "RESERVE", RES_UNITS)]
    g2 = graph(outs2)
    assert A.assess(h, outs2, g2.families(strict=True), policy="1.3").state == "SUPPORTED"


def test_invalid_necessary_prediction_is_weak_under_13():
    """The H2/T6 case: an over-strong NECESSARY prediction falsifies under 1.2 but not when the synthetic audit invalidates it."""
    h2 = H("H2.g0", "DIRECT_OFF_TARGET_PHOTOSTIMULATION", [Prediction("T6", "POS", N), Prediction("T0", "POS", N)])
    outs = [O("T6", Call.NULL, "CONFIRMATION", CONF_UNITS), O("T0", Call.POS, "RESERVE", RES_UNITS)]
    g = graph(outs)
    assert A.assess(h2, outs, g.families(), policy="1.2").state == "FALSIFIED_WITHIN_MODEL"
    st = A.assess(h2, outs, g.families(strict=True), policy="1.3", invalid_predictions=frozenset({("H2", "T6")})).state
    assert st == "WEAKENED"


def test_non_explanations_are_not_paired_under_13():
    h1 = H("H1.g0", "STATIC_MEASUREMENT_CONTAMINATION", [Prediction("T0", "POS", N), Prediction("T17", "NULL", N)])
    h3 = H("H3.g0", "GAP_JUNCTION_OR_LOCAL_ELECTRICAL_COUPLING", [Prediction("T0", "POS", S), Prediction("T17", "POS", N)])
    h4 = H("H4.g0", "DIRECT_CHEMICAL_SYNAPTIC_PROPAGATION", [Prediction("T0", "ANY", ND), Prediction("T17", "ANY", ND), Prediction("T3", "POS", N)])
    outs = [O("T0", Call.POS, "CONFIRMATION", CONF_UNITS), O("T17", Call.NULL, "CONFIRMATION", CONF_UNITS, stream="s2"),
            O("T17", Call.NULL, "RESERVE", RES_UNITS, stream="s2"), O("T0", Call.POS, "RESERVE", RES_UNITS)]
    g = graph(outs)
    r = A.adjudicate([h1, h3, h4], outs, g, phenomenon_test="T0", policy="1.3", phenomenon_family_tests=("T17",),
                     discrimination_report={"status": "NO_DISCRIMINATING_TEST", "candidates": []})
    assert "H4.g0" not in {x for p in r["non_identifiable_pairs"] for x in p}
    assert r["hypotheses"]["H3.g0"]["state"] == "FALSIFIED_WITHIN_MODEL"
    assert any("not a rival explanation" in n for n in r["hypotheses"]["H4.g0"]["notes"])


def test_quantitative_mixture_overlay_under_13():
    h1 = H("H1.g0", "STATIC_MEASUREMENT_CONTAMINATION", [Prediction("T0", "POS", N), Prediction("T17", "NULL", N)])
    h3 = H("H3.g0", "GAP_JUNCTION_OR_LOCAL_ELECTRICAL_COUPLING", [Prediction("T0", "POS", S), Prediction("T17", "POS", N)])
    det = {"explained_fraction": 0.7, "explained_fraction_ci": [0.55, 0.85]}
    outs = [O("T0", Call.POS, "CONFIRMATION", CONF_UNITS), O("T17", Call.POS, "CONFIRMATION", CONF_UNITS, stream="s2", details=det),
            O("T17", Call.POS, "RESERVE", RES_UNITS, stream="s2", details=det)]
    g = graph(outs)
    rep = {"status": "NO_DISCRIMINATING_TEST", "candidates": []}
    r12 = A.adjudicate([h1, h3], outs, g, phenomenon_test="T0", policy="1.2", discrimination_report=rep)
    r13 = A.adjudicate([h1, h3], outs, g, phenomenon_test="T0", policy="1.3", discrimination_report=rep, phenomenon_family_tests=("T17",),
                       mixture_share_tests=("T17",))
    r13_undeclared = A.adjudicate([h1, h3], outs, g, phenomenon_test="T0", policy="1.3", discrimination_report=rep, phenomenon_family_tests=("T17",))
    assert r12["verdict"] != "MULTIPLE_MECHANISMS_REQUIRED"
    assert r13["verdict"] == "MULTIPLE_MECHANISMS_REQUIRED"
    assert r13_undeclared["verdict"] != "MULTIPLE_MECHANISMS_REQUIRED"   # partial-model shares (e.g. static kernel T13) never trigger it


def test_adjudication_is_order_invariant():
    h1 = H("H1.g0", "STATIC_MEASUREMENT_CONTAMINATION", [Prediction("T0", "POS", N), Prediction("T17", "NULL", N), Prediction("T16", "ANY", ND)])
    h3 = H("H3.g0", "GAP_JUNCTION_OR_LOCAL_ELECTRICAL_COUPLING", [Prediction("T0", "POS", S), Prediction("T17", "POS", N), Prediction("T16", "POS", S)])
    outs = [O("T0", Call.POS, "CONFIRMATION", CONF_UNITS), O("T17", Call.NULL, "CONFIRMATION", CONF_UNITS, stream="s2"),
            O("T16", Call.POS, "CONFIRMATION", CONF_UNITS, stream="s3"), O("T17", Call.NULL, "RESERVE", RES_UNITS, stream="s2")]
    rep = {"status": "NO_DISCRIMINATING_TEST", "candidates": []}
    verdicts = set()
    for seed in range(6):
        random.Random(seed).shuffle(outs)
        r = A.adjudicate([h1, h3], list(outs), graph(outs), phenomenon_test="T0", policy="1.3", discrimination_report=rep, phenomenon_family_tests=("T17",))
        verdicts.add((r["verdict"], tuple(sorted((k, v["state"]) for k, v in r["hypotheses"].items()))))
    assert len(verdicts) == 1


def test_policy14_mixed_cluster_keeps_decisive_contradiction():
    """Audit finding A9: under 1.3 a strict cluster with a strong contradiction (T10) and a support from another test (T1) was
    'unresolved' and discarded; 1.4 keeps both, so the strong CONFIRMATION-grade contradiction falsifies."""
    h7 = H("H7.g0", "GLOBAL_OR_SHARED_STATE", [Prediction("T10", "POS", N), Prediction("T1", "NULL", S), Prediction("T0", "NULL", N)])
    outs = [O("T10", Call.NULL, "RESERVE", RES_UNITS), O("T1", Call.NULL, "RESERVE", RES_UNITS)]
    g = graph(outs)
    assert A.assess(h7, outs, g.families(strict=True), policy="1.3").state == "NEUTRAL"
    assert A.assess(h7, outs, g.families(strict=True), policy="1.4").state == "FALSIFIED_WITHIN_MODEL"
    same_test_conflict = [O("T10", Call.NULL, "RESERVE", RES_UNITS), O("T10", Call.POS, "RESERVE", RES_UNITS[:5])]
    g2 = graph(same_test_conflict)
    a = A.assess(h7, same_test_conflict, g2.families(strict=True), policy="1.4")
    assert a.unresolved_families and a.unresolved_families[0]["conflicted_tests"] == ["T10"]


def test_policy_versions_include_all_and_old_semantics_unchanged():
    assert A.POLICY_VERSIONS == ("1.0", "1.1", "1.2", "1.3", "1.4")
    o = O("T1", Call.POS, "DISCOVERY", DISC_UNITS)
    assert A.evidence_grade(o) == A.PART          # default (1.0) unchanged


def test_strict_families_historical_look_does_not_bridge_partitions():
    hist = O("T0", Call.POS, "HISTORICAL", CONF_UNITS + RES_UNITS, family="prox", stage="HISTORICAL")
    c = O("T0", Call.POS, "CONFIRMATION", CONF_UNITS, family="prox")
    r = O("T0", Call.POS, "RESERVE", RES_UNITS, family="prox")
    g = graph([hist, c, r])
    assert g.families()[c.outcome_id] == g.families()[r.outcome_id]            # loose rule (1.0-1.2) unchanged: bridged
    assert g.families(strict=True)[c.outcome_id] != g.families(strict=True)[r.outcome_id]
