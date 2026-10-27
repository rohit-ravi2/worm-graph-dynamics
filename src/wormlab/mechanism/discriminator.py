"""Discriminator engine: rank candidate analyses by expected ability to separate unresolved hypotheses.

Transparent score (no pretence of objective posteriors):

    score = EIG x power x reliability x prospective x (1 - confound_risk) / sqrt(cost)

* EIG          expected reduction in Shannon entropy of the prior weight over unresolved hypotheses (single
               mechanisms weight 1, pre-registered mixtures MIXTURE_PRIOR), if the
               test outcome were the committed prediction of the true hypothesis (ANY spreads evenly over
               POS/NULL/NEG). A test on which all unresolved hypotheses agree has EIG = 0.
* power        min(1, expected independent units / test.min_units); 0 below the minimum.
* reliability  1 for KNOWN measurement fields, x0.8 per ASSUMED or INFERRED field, 0 if any required
               field is UNKNOWN.
* prospective  UNTOUCHED 1.0, PARTIAL 0.5, CONTAMINATED 0.2; 0 if the (partition, family) was already run.

Returns ``NO_AVAILABLE_TEST_HAS_DISCRIMINATING_POWER`` when the best score is below ``min_score``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .measurement import ASSUMED, INFERRED, UNKNOWN, MeasurementModelSpec
from .spec import Call, HypothesisSpec, HypothesisStatus, MechanismTestSpec

RESOLVED_OUT = {HypothesisStatus.FALSIFIED_WITHIN_MODEL}   # only confirmation-grade falsification removes a hypothesis from play
PROSPECTIVE = {"UNTOUCHED": 1.0, "PARTIAL": 0.5, "CONTAMINATED": 0.2, "SYNTHETIC": 1.0}
NO_TEST = "NO_AVAILABLE_TEST_HAS_DISCRIMINATING_POWER"


def _entropy(ws):
    s = sum(ws)
    return -sum((w / s) * math.log2(w / s) for w in ws if w > 0) if s > 0 else 0.0


MIXTURE_PRIOR = 0.25   # complexity penalty: a pre-registered mixture carries 1/4 the prior weight of a single mechanism


def expected_information_gain(preds: dict[str, str], weights: dict[str, float] | None = None) -> float:
    hyps = list(preds)
    if len(hyps) < 2:
        return 0.0
    raw = {h: (weights or {}).get(h, 1.0) for h in hyps}
    tot = sum(raw.values())
    w = {h: raw[h] / tot for h in hyps}
    prior = _entropy(list(w.values()))
    exp_post = 0.0
    for o in (Call.POS, Call.NULL, Call.NEG):
        like = {h: (1.0 if preds[h] == o else (1 / 3 if preds[h] == Call.ANY else 0.0)) for h in hyps}
        po = sum(w[h] * like[h] for h in hyps)
        if po <= 0:
            continue
        post = [w[h] * like[h] / po for h in hyps]
        exp_post += po * _entropy(post)
    return max(0.0, prior - exp_post)


@dataclass
class Candidate:
    test_id: str
    partition: str
    score: float
    eig: float
    power: float
    reliability: float
    prospective: float
    separated_pairs: list
    grade: str
    reasons: list = field(default_factory=list)


def rank_tests(hypotheses: list[HypothesisSpec], states: dict, tests: list[MechanismTestSpec], *, availability: dict,
               measurement: MeasurementModelSpec | None = None, already_run: set = frozenset(), min_score: float = 0.02) -> dict:
    """``availability[test_id] = {"partition": str, "grade": str, "expected_units": int}`` (missing -> unavailable)."""
    unresolved = [h for h in hypotheses if states.get(h.hypothesis_id, HypothesisStatus.UNTESTED) not in RESOLVED_OUT]
    cands = []
    for t in tests:
        av = availability.get(t.test_id)
        reasons = []
        preds = {h.hypothesis_id: (h.prediction_for(t.test_id).expected if h.prediction_for(t.test_id) else Call.ANY) for h in unresolved}
        pairs = [(a, b) for i, a in enumerate(sorted(preds)) for b in sorted(preds)[i + 1:]
                 if preds[a] != Call.ANY and preds[b] != Call.ANY and preds[a] != preds[b]]
        eig = expected_information_gain(preds, {h.hypothesis_id: (MIXTURE_PRIOR if h.mechanism_class == "MIXTURE_MODEL" else 1.0) for h in unresolved})
        if av is None:
            cands.append(Candidate(t.test_id, "-", 0.0, eig, 0.0, 0.0, 0.0, pairs, "-", ["data unavailable"]))
            continue
        power = 0.0 if av["expected_units"] < t.min_units else min(1.0, av["expected_units"] / max(t.min_units, 1))
        if power == 0:
            reasons.append(f"expected units {av['expected_units']} < min {t.min_units}")
        rel = 1.0
        if measurement is not None:
            for fname in t.required_measurement_fields:
                st = measurement.get(fname).status
                if st == UNKNOWN:
                    rel = 0.0
                    reasons.append(f"measurement field {fname} UNKNOWN")
                elif st in (ASSUMED, INFERRED):
                    rel *= 0.8
        pro = 0.0 if (av["partition"], t.statistic_family) in already_run else PROSPECTIVE.get(av["grade"], 0.2)
        if pro == 0:
            reasons.append("already run on this partition/family")
        score = eig * power * rel * pro * (1 - t.confound_risk) / math.sqrt(max(t.cost, 1e-9))
        if not pairs:
            reasons.append("unresolved hypotheses make identical committed predictions")
        cands.append(Candidate(t.test_id, av["partition"], round(score, 5), round(eig, 4), round(power, 3), round(rel, 3), pro, pairs, av["grade"], reasons))
    cands.sort(key=lambda c: -c.score)
    best = cands[0].score if cands else 0.0
    return dict(status=("RANKED" if best >= min_score else NO_TEST), min_score=min_score,
                unresolved=[h.hypothesis_id for h in unresolved], candidates=[c.__dict__ for c in cands])


def identifiability_matrix(hypotheses: list[HypothesisSpec], tests: list[MechanismTestSpec], usable_tests: set[str]) -> dict:
    """For each hypothesis pair: which usable tests separate them (committed, different calls)."""
    out = {}
    for i, a in enumerate(hypotheses):
        for b in hypotheses[i + 1:]:
            sep = []
            for t in tests:
                if t.test_id not in usable_tests:
                    continue
                pa, pb = a.prediction_for(t.test_id), b.prediction_for(t.test_id)
                if pa and pb and Call.ANY not in (pa.expected, pb.expected) and pa.expected != pb.expected:
                    sep.append(t.test_id)
            out[f"{a.hypothesis_id}|{b.hypothesis_id}"] = sep
    return out
