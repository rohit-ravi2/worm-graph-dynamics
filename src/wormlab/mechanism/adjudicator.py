"""Deterministic mechanism adjudication (policy in docs/mechanism_engine/MECHANISM_ADJUDICATION_POLICY.md).

Evidence grades
    CONFIRMATION   stage CONFIRMATION (or SYNTHETIC in fixture mode) and contamination UNTOUCHED
    PARTIAL        stage CONFIRMATION with PARTIAL contamination, or stage DISCOVERY that was UNTOUCHED
    EXPLORATORY    DISCOVERY with PARTIAL/CONTAMINATED contamination, HISTORICAL imports

Evidence is aggregated per EVIDENCE FAMILY (evidence_graph.families): within a family, conflicting relations
are UNRESOLVED and count for nothing; one family counts once, at its best grade.

Hypothesis states (first matching rule wins)
    FALSIFIED_WITHIN_MODEL  >= 1 strong contradiction (NECESSARY/FALSIFYING) at CONFIRMATION grade
    STRONGLY_DISFAVORED     >= 2 strong contradictions at PARTIAL grade, or 1 strong PARTIAL + 1 weak CONFIRMATION
    WEAKENED                any other contradiction at PARTIAL or CONFIRMATION grade, or strong EXPLORATORY contradiction
    SUPPORTED               >= 2 supporting CONFIRMATION families and no contradictions at PARTIAL+ grade
    WEAKLY_SUPPORTED        >= 1 supporting family at PARTIAL+ grade and no contradictions at PARTIAL+ grade
    NEUTRAL                 only EXPLORATORY support, or decided outcomes that touch no committed prediction
    UNTESTED                no decided outcome on any committed prediction
    NON_IDENTIFIABLE        (overlay) survivor whose committed predictions coincide with another survivor on every
                            DECIDED test and no remaining usable test separates them

Terminal verdicts: MECHANISM_SUPPORTED, MECHANISM_DISFAVORED, MECHANISM_FALSIFIED_WITHIN_TESTED_MODEL,
MULTIPLE_MECHANISMS_REQUIRED, NON_IDENTIFIABLE_FROM_AVAILABLE_DATA, INSUFFICIENT_SIGNAL,
MEASUREMENT_MODEL_INADEQUATE, NEW_EXPERIMENT_REQUIRED — or CONTINUE (not terminal) while a discriminating,
permitted analysis remains.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .evidence_graph import EvidenceGraph, relation_for
from .spec import Call, HypothesisSpec, HypothesisStatus, OutcomeStatus, PredictionKind, Stage, TestOutcome

CONF, PART, EXPL = "CONFIRMATION", "PARTIAL", "EXPLORATORY"
RANK = {EXPL: 0, PART: 1, CONF: 2}
SURVIVING = {HypothesisStatus.SUPPORTED, HypothesisStatus.WEAKLY_SUPPORTED, HypothesisStatus.NEUTRAL,
             HypothesisStatus.UNTESTED, HypothesisStatus.NON_IDENTIFIABLE}
VERDICTS = ("MECHANISM_SUPPORTED", "MECHANISM_DISFAVORED", "MECHANISM_FALSIFIED_WITHIN_TESTED_MODEL", "MULTIPLE_MECHANISMS_REQUIRED",
            "NON_IDENTIFIABLE_FROM_AVAILABLE_DATA", "INSUFFICIENT_SIGNAL", "MEASUREMENT_MODEL_INADEQUATE", "NEW_EXPERIMENT_REQUIRED", "CONTINUE")


def evidence_grade(o: TestOutcome, *, synthetic_is_confirmation: bool = False, policy: str = "1.0") -> str:
    if o.stage == Stage.SYNTHETIC:
        return CONF if synthetic_is_confirmation else EXPL
    if o.stage == Stage.CONFIRMATION:
        return CONF if o.contamination_grade == "UNTOUCHED" else (PART if o.contamination_grade == "PARTIAL" else EXPL)
    if o.stage == Stage.DISCOVERY:
        if policy in ("1.3", "1.4"):
            return EXPL          # 1.3: statistics are developed on DISCOVERY; its outcomes never count above EXPLORATORY
        return PART if o.contamination_grade == "UNTOUCHED" else EXPL
    return EXPL


@dataclass
class HypothesisAssessment:
    hypothesis_id: str
    state: str
    supports: list = field(default_factory=list)       # [{"family", "grade", "tests"}]
    contradicts: list = field(default_factory=list)    # [{"family", "grade", "strength", "tests"}]
    unresolved_families: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _family_relations(h: HypothesisSpec, outcomes: list[TestOutcome], families: dict, synth_conf: bool, policy: str = "1.0",
                      invalid_predictions: frozenset = frozenset()):
    fam: dict[str, dict] = {}
    for o in outcomes:
        if o.status != OutcomeStatus.DECIDED:
            continue
        p = h.prediction_for(o.test_id)
        rel = relation_for(p, o.call)
        if rel is None:
            continue
        key = families.get(o.outcome_id, o.outcome_id)
        g = evidence_grade(o, synthetic_is_confirmation=synth_conf, policy=policy)
        strong_kind = p.kind in (PredictionKind.NECESSARY_PREDICTION, PredictionKind.FALSIFYING_OBSERVATION)
        if policy in ("1.3", "1.4") and (h.family, o.test_id) in invalid_predictions:
            strong_kind = False  # 1.3: a NECESSARY prediction contradicted by the hypothesis' own synthetic generator is not strong
        strength = "strong" if (rel == "contradicts" and strong_kind) else "weak"
        rec = fam.setdefault(key, {"rels": set(), "grade": EXPL, "strength": "weak", "tests": set(), "necessary": False, "partitions": set(),
                                   "by_test": {}, "rel_grade": {}, "rel_strength": {}})
        rec["partitions"].add(o.partition)
        rec["rels"].add(rel)
        rec["by_test"].setdefault(o.test_id, set()).add(rel)
        if RANK[g] > RANK[rec["rel_grade"].get(rel, EXPL)] or rel not in rec["rel_grade"]:
            rec["rel_grade"][rel] = g if rel not in rec["rel_grade"] else (g if RANK[g] > RANK[rec["rel_grade"][rel]] else rec["rel_grade"][rel])
        if rel == "contradicts" and strength == "strong":
            rec["rel_strength"]["contradicts"] = "strong"
        if rel == "supports" and p.kind == PredictionKind.NECESSARY_PREDICTION and g == CONF and not (policy in ("1.3", "1.4") and (h.family, o.test_id) in invalid_predictions):
            rec["necessary"] = True
        rec["tests"].add(o.test_id)
        if RANK[g] > RANK[rec["grade"]]:
            rec["grade"] = g
        if strength == "strong":
            rec["strength"] = "strong"
    return fam


def assess(h: HypothesisSpec, outcomes: list[TestOutcome], families: dict, *, synthetic_is_confirmation=False, policy: str = "1.0",
           invalid_predictions: frozenset = frozenset()) -> HypothesisAssessment:
    fam = _family_relations(h, outcomes, families, synthetic_is_confirmation, policy, invalid_predictions)
    a = HypothesisAssessment(h.hypothesis_id, HypothesisStatus.UNTESTED)
    for key, rec in fam.items():
        entry = {"family": key, "grade": rec["grade"], "tests": sorted(rec["tests"]), "necessary": rec["necessary"], "partitions": sorted(rec["partitions"])}
        if policy == "1.4" and len(rec["rels"]) > 1:
            # 1.4: a dependence cluster mixing DIFFERENT tests contributes one support and one contradiction (each at its own
            # best grade); only a test that itself carries both relations is unresolved
            conflicted = sorted(t for t, r in rec["by_test"].items() if len(r) > 1)
            if conflicted:
                a.unresolved_families.append(dict(entry, conflicted_tests=conflicted))
            clean = {t: next(iter(r)) for t, r in rec["by_test"].items() if len(r) == 1}
            ct = sorted(t for t, r in clean.items() if r == "contradicts")
            st = sorted(t for t, r in clean.items() if r == "supports")
            if ct:
                a.contradicts.append(dict(entry, tests=ct, grade=rec["rel_grade"]["contradicts"], strength=rec["rel_strength"].get("contradicts", "weak")))
            if st:
                a.supports.append(dict(entry, tests=st, grade=rec["rel_grade"]["supports"]))
            continue
        if len(rec["rels"]) > 1:
            a.unresolved_families.append(entry)
        elif "contradicts" in rec["rels"]:
            a.contradicts.append(dict(entry, strength=rec["strength"]))
        else:
            a.supports.append(entry)
    sc = [c for c in a.contradicts if c["strength"] == "strong"]
    wc = [c for c in a.contradicts if c["strength"] == "weak"]
    n = lambda lst, g: sum(1 for x in lst if x["grade"] == g)
    if n(sc, CONF) >= 1:
        a.state = HypothesisStatus.FALSIFIED_WITHIN_MODEL
    elif n(sc, PART) >= 2 or (n(sc, PART) >= 1 and n(wc, CONF) >= 1):
        a.state = HypothesisStatus.STRONGLY_DISFAVORED
    elif n(sc, PART) + n(wc, PART) + n(wc, CONF) >= 1 or n(sc, EXPL) >= 1:
        a.state = HypothesisStatus.WEAKENED
    elif n(a.supports, CONF) >= 2 and (policy not in ("1.2", "1.3", "1.4") or any(x["grade"] == CONF and x.get("necessary") for x in a.supports)) \
            and (policy not in ("1.3", "1.4") or len({p for x in a.supports if x["grade"] == CONF and x.get("necessary") for p in x["partitions"]}) >= 2):
        a.state = HypothesisStatus.SUPPORTED
    elif n(a.supports, CONF) + n(a.supports, PART) >= 1:
        a.state = HypothesisStatus.WEAKLY_SUPPORTED
    elif a.supports or a.unresolved_families or a.contradicts:
        a.state = HypothesisStatus.NEUTRAL
    return a


def _committed_agreement(a: HypothesisSpec, b: HypothesisSpec, decided_tests: set[str]) -> bool:
    for t in decided_tests:
        pa, pb = a.prediction_for(t), b.prediction_for(t)
        ea = pa.expected if pa else Call.ANY
        eb = pb.expected if pb else Call.ANY
        if ea != Call.ANY and eb != Call.ANY and ea != eb:
            return False
    return True


POLICY_VERSIONS = ("1.0", "1.1", "1.2", "1.3", "1.4")
# 1.1 (declared 2026-09-16 after g0; conservative): WEAKENED hypotheses remain unresolved survivors; MECHANISM_DISFAVORED /
#     FALSIFIED verdicts require every single hypothesis to be STRONGLY_DISFAVORED or FALSIFIED; mixtures that are SUPPORTED
#     but do not meet the component-contradiction rule never produce MULTIPLE_MECHANISMS_REQUIRED.
# 1.2 (declared 2026-09-16 after g1; strictly adds requirements to support): 1.1 plus
#     (a) state SUPPORTED additionally requires >= 1 CONFIRMATION-grade supporting family on a NECESSARY prediction
#         (otherwise at most WEAKLY_SUPPORTED);
#     (b) MECHANISM_SUPPORTED additionally requires, for EVERY other surviving single hypothesis (including WEAKENED),
#         a DECIDED CONFIRMATION-grade outcome on a test where the two make different committed predictions and the
#         leader's prediction matched; any rival lacking such a separating outcome makes the pair NON_IDENTIFIABLE.
# 1.3 (declared 2026-09-17, Deep Mechanism Campaign hostile audit; see docs/mechanism_engine/deep_campaign/ADJUDICATOR_AUDIT.md):
#     1.2 plus (a) DISCOVERY outcomes are EXPLORATORY; (b) evidence families are STRICT: outcomes on the same partition and data
#     stream with overlapping units form one family (shared animals/events/response variables); (c) SUPPORTED requires
#     CONFIRMATION-grade NECESSARY support on >= 2 partitions (independent animals); (d) NECESSARY predictions listed as
#     invalid by the synthetic prediction audit are weak; (e) only hypotheses committing POS on a phenomenon-family test
#     are rival explanations of the phenomenon (others are reported, not paired); (f) quantitative mixture overlay: a
#     CONFIRMATION-grade POS outcome of a declared full-contamination-model test (mixture_share_tests) whose details carry a
#     contamination explained-fraction CI inside (0.2, 0.9) yields MULTIPLE_MECHANISMS_REQUIRED.
# 1.4 (declared 2026-09-17 before any g3 CONFIRMATION/RESERVE access; audit finding A9): under 1.3 a strict dependence cluster
#     whose DIFFERENT tests point in different directions was marked unresolved and discarded, erasing decisive evidence (e.g.
#     g1 H7 FALSIFIED became NEUTRAL). 1.4 = 1.3 except such a cluster contributes one support and one contradiction, each at
#     its own best grade and strength; only a test that itself carries both relations is unresolved.


def adjudicate(hypotheses: list[HypothesisSpec], outcomes: list[TestOutcome], graph: EvidenceGraph, *,
               phenomenon_test: str | None = None, discrimination_report: dict | None = None,
               measurement_blocked_tests: set[str] = frozenset(), synthetic_is_confirmation: bool = False,
               allow_mixtures: bool = True, policy: str = "1.0", invalid_predictions: frozenset = frozenset(),
               phenomenon_family_tests: tuple = (), mixture_share_tests: tuple = ()) -> dict:
    assert policy in POLICY_VERSIONS
    families = graph.families(strict=(policy in ("1.3", "1.4")))
    singles = [h for h in hypotheses if h.mechanism_class != "MIXTURE_MODEL"]
    assessed = {h.hypothesis_id: assess(h, outcomes, families, synthetic_is_confirmation=synthetic_is_confirmation, policy=policy,
                                        invalid_predictions=frozenset(invalid_predictions)) for h in hypotheses}
    not_explanations = []
    if policy in ("1.3", "1.4"):
        ptests = set(phenomenon_family_tests) | ({phenomenon_test} if phenomenon_test else set())
        relevant = []
        for h in singles:
            if any((h.prediction_for(t) is not None and h.prediction_for(t).expected == Call.POS) for t in ptests):
                relevant.append(h)
            else:
                not_explanations.append(h.hypothesis_id)
                assessed[h.hypothesis_id].notes.append("not a rival explanation of the phenomenon (commits POS on no phenomenon-family test)")
        singles = relevant
    decided = {o.test_id for o in outcomes if o.status == OutcomeStatus.DECIDED}

    # ---- phenomenon gate ----
    if phenomenon_test is not None:
        ph = [o for o in outcomes if o.test_id == phenomenon_test and o.status == OutcomeStatus.DECIDED]
        if not ph or not any(o.call == Call.POS for o in ph):
            return _result("INSUFFICIENT_SIGNAL", assessed, [], [], None, notes=[f"phenomenon test {phenomenon_test} not decided POS"], policy=policy)

    # ---- pre-registered mixtures: each component contradicted (CONFIRMATION grade) where the other is supported,
    #      and the mixture itself uncontradicted ----
    mixtures = []
    registered = [h for h in hypotheses if h.mechanism_class == "MIXTURE_MODEL"]
    if allow_mixtures:
        by_id = {h.hypothesis_id: h for h in singles}
        for M in registered:
            comps = [by_id[c] for c in M.components if c in by_id]
            if len(comps) != 2:
                continue
            A, B = comps
            aA, aB, aM = assessed[A.hypothesis_id], assessed[B.hypothesis_id], assessed[M.hypothesis_id]
            if aM.contradicts or not aA.contradicts or not aB.contradicts:
                continue
            cA = {t for c in aA.contradicts if c["grade"] == CONF for t in c["tests"]}
            cB = {t for c in aB.contradicts if c["grade"] == CONF for t in c["tests"]}
            sA = {t for x in aA.supports if x["grade"] == CONF for t in x["tests"]}
            sB = {t for x in aB.supports if x["grade"] == CONF for t in x["tests"]}
            if (cA & sB) and (cB & sA) and aM.state in (HypothesisStatus.SUPPORTED, HypothesisStatus.WEAKLY_SUPPORTED):
                mixtures.append(dict(mixture=M.hypothesis_id, components=[A.hypothesis_id, B.hypothesis_id], state=aM.state,
                                     explains_A_failures=sorted(cA & sB), explains_B_failures=sorted(cB & sA)))

    surviving_states = SURVIVING | ({HypothesisStatus.WEAKENED} if policy in ("1.1", "1.2", "1.3", "1.4") else set())
    survivors = [h for h in singles if assessed[h.hypothesis_id].state in surviving_states]
    # ---- non-identifiable overlay among survivors ----
    usable = set()
    if discrimination_report:
        usable = {c["test_id"] for c in discrimination_report.get("candidates", []) if c["score"] > 0}
    nonid_pairs = []
    for i, A in enumerate(survivors):
        for B in survivors[i + 1:]:
            if _committed_agreement(A, B, decided):
                sep = [t for t in usable
                       if (A.prediction_for(t) and B.prediction_for(t) and Call.ANY not in (A.prediction_for(t).expected, B.prediction_for(t).expected)
                           and A.prediction_for(t).expected != B.prediction_for(t).expected)]
                if not sep:
                    nonid_pairs.append([A.hypothesis_id, B.hypothesis_id])
    for pair in nonid_pairs:
        for hid in pair:
            if assessed[hid].state in (HypothesisStatus.WEAKLY_SUPPORTED, HypothesisStatus.NEUTRAL, HypothesisStatus.UNTESTED):
                assessed[hid].notes.append("NON_IDENTIFIABLE with " + [x for x in pair if x != hid][0])
                assessed[hid].state = HypothesisStatus.NON_IDENTIFIABLE

    remaining = discrimination_report is not None and discrimination_report.get("status") == "RANKED"
    supported = [h for h in survivors if assessed[h.hypothesis_id].state == HypothesisStatus.SUPPORTED]
    weakly = [h for h in survivors if assessed[h.hypothesis_id].state == HypothesisStatus.WEAKLY_SUPPORTED]

    if remaining:
        return _result("CONTINUE", assessed, mixtures, nonid_pairs, None, notes=["a permitted discriminating analysis remains"], policy=policy)
    if mixtures and not supported:
        return _result("MULTIPLE_MECHANISMS_REQUIRED", assessed, mixtures, nonid_pairs, None, policy=policy)
    if policy in ("1.3", "1.4"):
        quant = []
        for o in outcomes:
            if o.test_id not in mixture_share_tests or o.status != OutcomeStatus.DECIDED or o.call != Call.POS \
                    or evidence_grade(o, synthetic_is_confirmation=synthetic_is_confirmation, policy=policy) != CONF:
                continue
            ef = (o.details or {}).get("explained_fraction_ci")
            if ef and all(x is not None for x in ef) and ef[0] > 0.2 and ef[1] < 0.9:
                quant.append(dict(test_id=o.test_id, partition=o.partition, explained_fraction=(o.details or {}).get("explained_fraction"), ci=ef))
        if quant and len({q["partition"] for q in quant}) >= 1:
            mixtures = mixtures + [dict(quantitative=True, evidence=quant)]
            return _result("MULTIPLE_MECHANISMS_REQUIRED", assessed, mixtures, nonid_pairs, None, policy=policy,
                           notes=["quantitative mixture: the contamination model explains a bounded fraction (CI inside 0.2-0.9) of a POS phenomenon-family effect",
                                  {"not_explanations": not_explanations}])
    if len(supported) == 1 and not nonid_pairs and not weakly and policy in ("1.2", "1.3", "1.4"):
        L = supported[0]
        unseparated = []
        for R in survivors:
            if R.hypothesis_id == L.hypothesis_id:
                continue
            ok = False
            for o in outcomes:
                if o.status != OutcomeStatus.DECIDED or evidence_grade(o, synthetic_is_confirmation=synthetic_is_confirmation, policy=policy) != CONF:
                    continue
                pl, pr = L.prediction_for(o.test_id), R.prediction_for(o.test_id)
                if pl and pr and Call.ANY not in (pl.expected, pr.expected) and pl.expected != pr.expected and o.call == pl.expected:
                    ok = True
                    break
            if not ok:
                unseparated.append([L.hypothesis_id, R.hypothesis_id])
        if not unseparated:
            return _result("MECHANISM_SUPPORTED", assessed, mixtures, nonid_pairs, L.hypothesis_id, policy=policy)
        for pair in unseparated:
            assessed[pair[0]].notes.append(f"not separated from {pair[1]} by any CONFIRMATION-grade outcome")
        return _result("NON_IDENTIFIABLE_FROM_AVAILABLE_DATA", assessed, mixtures, unseparated, None, policy=policy,
                       notes=[f"leader {L.hypothesis_id} is SUPPORTED but not separated at CONFIRMATION grade from: {[p[1] for p in unseparated]}"])
    if len(supported) == 1 and not nonid_pairs and not weakly:
        return _result("MECHANISM_SUPPORTED", assessed, mixtures, nonid_pairs, supported[0].hypothesis_id, policy=policy)
    if not survivors and not mixtures:
        falsified = [h for h, a in assessed.items() if a.state == HypothesisStatus.FALSIFIED_WITHIN_MODEL]
        return _result("MECHANISM_FALSIFIED_WITHIN_TESTED_MODEL" if falsified else "MECHANISM_DISFAVORED", assessed, mixtures, nonid_pairs, None, policy=policy)
    if policy in ("1.1", "1.2", "1.3", "1.4") and len(survivors) >= 2:
        pair_report = []
        for i, A in enumerate(survivors):
            for B in survivors[i + 1:]:
                sep = [t for t in {p.test_id for p in A._preds()} | {p.test_id for p in B._preds()}
                       if A.prediction_for(t) and B.prediction_for(t) and Call.ANY not in (A.prediction_for(t).expected, B.prediction_for(t).expected)
                       and A.prediction_for(t).expected != B.prediction_for(t).expected]
                blocked = [t for t in sep if t in measurement_blocked_tests]
                pair_report.append(dict(pair=[A.hypothesis_id, B.hypothesis_id], separating_tests=sorted(sep), blocked=sorted(blocked),
                                        measurement_limited=bool(sep) and len(blocked) == len(sep)))
        if nonid_pairs or any(not r["separating_tests"] for r in pair_report):
            return _result("NON_IDENTIFIABLE_FROM_AVAILABLE_DATA", assessed, mixtures, nonid_pairs, None, policy=policy,
                           notes=[{"survivor_pairs": pair_report}])
        if pair_report and all(r["measurement_limited"] for r in pair_report):
            return _result("MEASUREMENT_MODEL_INADEQUATE", assessed, mixtures, nonid_pairs, None, policy=policy, notes=[{"survivor_pairs": pair_report}])
        return _result("NEW_EXPERIMENT_REQUIRED", assessed, mixtures, nonid_pairs, None, policy=policy,
                       notes=["surviving hypotheses differ only on tests that were inconclusive, underpowered, below confirmation grade or blocked",
                              {"survivor_pairs": pair_report}])
    if nonid_pairs:
        return _result("NON_IDENTIFIABLE_FROM_AVAILABLE_DATA", assessed, mixtures, nonid_pairs, None, policy=policy)
    if measurement_blocked_tests:
        return _result("MEASUREMENT_MODEL_INADEQUATE", assessed, mixtures, nonid_pairs, None,
                       notes=[f"discriminating tests blocked by UNKNOWN measurement fields: {sorted(measurement_blocked_tests)}"], policy=policy)
    return _result("NEW_EXPERIMENT_REQUIRED", assessed, mixtures, nonid_pairs, None,
                   notes=["surviving hypotheses make different predictions only on tests that are unavailable, underpowered or contaminated"], policy=policy)


def _result(verdict, assessed, mixtures, nonid_pairs, leader, notes=(), policy="1.0"):
    assert verdict in VERDICTS
    return dict(verdict=verdict, leading_hypothesis=leader, policy_version=policy,
                hypotheses={h: a.__dict__ for h, a in sorted(assessed.items())},
                mixtures=mixtures, non_identifiable_pairs=nonid_pairs,
                surviving=[h for h, a in sorted(assessed.items()) if a.state in SURVIVING],
                terminal=verdict != "CONTINUE", notes=list(notes))
