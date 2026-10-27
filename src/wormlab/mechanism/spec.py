"""Executable scientific objects for mechanism adjudication.

* ``HypothesisSpec``   — a mechanism claim with predictions that can count FOR and AGAINST it.
* ``MechanismTestSpec`` — a contrast: which hypotheses it separates, what each predicts, how it is run.
* ``TestOutcome``       — a validated result of running one test on one partition.

Predictions are expressed over a small, explicit outcome alphabet so the compiler, discriminator and
adjudicator can reason about them without reading prose:

    POS   effect CI lies entirely above +floor
    NULL  effect CI lies entirely inside [-floor, +floor]   (equivalence, not "not significant")
    NEG   effect CI lies entirely below -floor
    ANY   the hypothesis makes no commitment for this test (non-discriminating for it)

A CI that straddles a floor is INCONCLUSIVE and never counts for or against anything.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


class SpecError(ValueError):
    """A spec is logically incomplete or internally inconsistent."""


# ----------------------------------------------------------------------------------------------
# vocabularies
# ----------------------------------------------------------------------------------------------
class PredictionKind(str, Enum):
    SUPPORTIVE_PREDICTION = "SUPPORTIVE_PREDICTION"          # match supports; mismatch mildly weakens
    NECESSARY_PREDICTION = "NECESSARY_PREDICTION"            # mismatch counts strongly against
    FALSIFYING_OBSERVATION = "FALSIFYING_OBSERVATION"        # observing `expected` counts strongly against
    NON_DISCRIMINATING_OBSERVATION = "NON_DISCRIMINATING_OBSERVATION"


class Call(str, Enum):
    POS = "POS"
    NULL = "NULL"
    NEG = "NEG"
    ANY = "ANY"
    INCONCLUSIVE = "INCONCLUSIVE"


class MechanismClass(str, Enum):
    STATIC_MEASUREMENT_CONTAMINATION = "STATIC_MEASUREMENT_CONTAMINATION"
    DIRECT_OFF_TARGET_PHOTOSTIMULATION = "DIRECT_OFF_TARGET_PHOTOSTIMULATION"
    GAP_JUNCTION_OR_LOCAL_ELECTRICAL_COUPLING = "GAP_JUNCTION_OR_LOCAL_ELECTRICAL_COUPLING"
    DIRECT_CHEMICAL_SYNAPTIC_PROPAGATION = "DIRECT_CHEMICAL_SYNAPTIC_PROPAGATION"
    POLYSYNAPTIC_NETWORK_PROPAGATION = "POLYSYNAPTIC_NETWORK_PROPAGATION"
    EXTRASYNAPTIC_OR_NEUROPEPTIDERGIC_SIGNALING = "EXTRASYNAPTIC_OR_NEUROPEPTIDERGIC_SIGNALING"
    GLOBAL_OR_SHARED_STATE = "GLOBAL_OR_SHARED_STATE"
    LOCAL_SHARED_BIOLOGICAL_FIELD = "LOCAL_SHARED_BIOLOGICAL_FIELD"   # added g3 (Deep Mechanism Campaign); existing hashes unaffected
    MIXTURE_MODEL = "MIXTURE_MODEL"


class HypothesisStatus(str, Enum):
    UNTESTED = "UNTESTED"
    SUPPORTED = "SUPPORTED"
    WEAKLY_SUPPORTED = "WEAKLY_SUPPORTED"
    NEUTRAL = "NEUTRAL"
    WEAKENED = "WEAKENED"
    STRONGLY_DISFAVORED = "STRONGLY_DISFAVORED"
    FALSIFIED_WITHIN_MODEL = "FALSIFIED_WITHIN_MODEL"
    NON_IDENTIFIABLE = "NON_IDENTIFIABLE"


class Stage(str, Enum):
    DISCOVERY = "DISCOVERY"
    CONFIRMATION = "CONFIRMATION"
    SYNTHETIC = "SYNTHETIC"
    HISTORICAL = "HISTORICAL"          # imported prior analyses; never confirmation-grade


class OutcomeStatus(str, Enum):
    DECIDED = "DECIDED"                                  # call in {POS, NULL, NEG}
    INCONCLUSIVE = "INCONCLUSIVE"                        # CI straddles a floor
    UNDERPOWERED = "UNDERPOWERED"                        # coverage below the test's minimum
    NON_IDENTIFIABLE_RESOLUTION = "NON_IDENTIFIABLE_RESOLUTION"  # measurement resolution cannot separate
    NOT_RUN_DATA_MISSING = "NOT_RUN_DATA_MISSING"
    NOT_RUN_ACCESS_DENIED = "NOT_RUN_ACCESS_DENIED"


PARTITIONS = ("DISCOVERY", "CONFIRMATION", "RESERVE", "SYNTHETIC", "HISTORICAL")


# ----------------------------------------------------------------------------------------------
# hypothesis
# ----------------------------------------------------------------------------------------------
@dataclass
class Prediction:
    test_id: str
    expected: str                     # Call value (POS/NULL/NEG/ANY)
    kind: str                         # PredictionKind value
    rationale: str = ""
    delay_range_s: list | None = None  # optional [lo, hi] for timing tests

    def validate(self) -> None:
        if self.expected not in (Call.POS, Call.NULL, Call.NEG, Call.ANY):
            raise SpecError(f"prediction for {self.test_id}: expected must be POS/NULL/NEG/ANY, got {self.expected!r}")
        if self.kind not in PredictionKind.__members__:
            raise SpecError(f"prediction for {self.test_id}: unknown kind {self.kind!r}")
        if self.expected == Call.ANY and self.kind != PredictionKind.NON_DISCRIMINATING_OBSERVATION:
            raise SpecError(f"prediction for {self.test_id}: ANY must be NON_DISCRIMINATING_OBSERVATION")
        if self.delay_range_s is not None and (len(self.delay_range_s) != 2 or self.delay_range_s[0] > self.delay_range_s[1]):
            raise SpecError(f"prediction for {self.test_id}: delay_range_s must be [lo, hi]")

    def counts_against_when(self) -> set[str]:
        """Observed calls that count against the hypothesis."""
        if self.kind == PredictionKind.FALSIFYING_OBSERVATION:
            return {self.expected}
        if self.kind in (PredictionKind.NECESSARY_PREDICTION, PredictionKind.SUPPORTIVE_PREDICTION):
            return {c for c in (Call.POS, Call.NULL, Call.NEG) if c != self.expected}
        return set()


@dataclass
class HypothesisSpec:
    hypothesis_id: str                 # "<family>.g<generation>", e.g. "H5.g0"
    generation: int
    claim: str
    mechanism_class: str
    causal_entities: list
    causal_relations: list
    required_observables: list
    predicted_signatures: list         # human-readable signature list (each tied to a prediction test_id)
    predicted_absences: list
    falsifiers: list                   # human-readable; each must correspond to a counting-against prediction
    predictions: list                  # list[Prediction]
    known_confounds: list
    alternative_hypotheses: list
    identifiability_conditions: list
    required_controls: list
    allowed_data_partitions: list
    forbidden_data_partitions: list
    status: str = HypothesisStatus.UNTESTED
    parent_hypothesis_ids: list = field(default_factory=list)
    created_from_evidence_ids: list = field(default_factory=list)
    components: list = field(default_factory=list)   # for MIXTURE_MODEL: component hypothesis ids

    @property
    def family(self) -> str:
        return self.hypothesis_id.split(".g")[0]

    @property
    def supporting_tests(self) -> list[str]:
        return sorted({p.test_id for p in self._preds() if p.kind in (PredictionKind.SUPPORTIVE_PREDICTION, PredictionKind.NECESSARY_PREDICTION)})

    @property
    def discriminating_tests(self) -> list[str]:
        return sorted({p.test_id for p in self._preds() if p.expected != Call.ANY})

    def _preds(self) -> list[Prediction]:
        return [p if isinstance(p, Prediction) else Prediction(**p) for p in self.predictions]

    def prediction_for(self, test_id: str) -> Prediction | None:
        for p in self._preds():
            if p.test_id == test_id:
                return p
        return None

    def content(self) -> dict:
        d = asdict(self)
        d["predictions"] = [asdict(p) if isinstance(p, Prediction) else dict(p) for p in self.predictions]
        d.pop("status")
        return d

    @property
    def spec_hash(self) -> str:
        return canonical_sha(self.content())

    def validate(self) -> None:
        fam, _, gen = self.hypothesis_id.partition(".g")
        if not fam or not gen.isdigit() or int(gen) != self.generation:
            raise SpecError(f"{self.hypothesis_id}: id must be '<family>.g<generation>' matching generation={self.generation}")
        if not self.claim.strip():
            raise SpecError(f"{self.hypothesis_id}: empty claim")
        if self.mechanism_class not in MechanismClass.__members__:
            raise SpecError(f"{self.hypothesis_id}: unknown mechanism_class {self.mechanism_class!r}")
        preds = self._preds()
        if not preds:
            raise SpecError(f"{self.hypothesis_id}: no predictions")
        seen = set()
        for p in preds:
            p.validate()
            if p.test_id in seen:
                raise SpecError(f"{self.hypothesis_id}: duplicate prediction for {p.test_id}")
            seen.add(p.test_id)
        if not any(p.counts_against_when() for p in preds):
            raise SpecError(f"{self.hypothesis_id}: not judgeable — no observation could count against it")
        if not self.falsifiers:
            raise SpecError(f"{self.hypothesis_id}: falsifiers list is empty")
        if not self.discriminating_tests:
            raise SpecError(f"{self.hypothesis_id}: no discriminating prediction")
        bad = set(self.allowed_data_partitions) & set(self.forbidden_data_partitions)
        if bad:
            raise SpecError(f"{self.hypothesis_id}: partitions both allowed and forbidden: {sorted(bad)}")
        for part in list(self.allowed_data_partitions) + list(self.forbidden_data_partitions):
            if part not in PARTITIONS:
                raise SpecError(f"{self.hypothesis_id}: unknown partition {part!r}")
        if self.generation > 0 and not self.parent_hypothesis_ids:
            raise SpecError(f"{self.hypothesis_id}: generation > 0 requires parent_hypothesis_ids")
        if self.generation > 0 and not self.created_from_evidence_ids:
            raise SpecError(f"{self.hypothesis_id}: generation > 0 must cite the evidence that motivated revision")
        if self.mechanism_class == MechanismClass.MIXTURE_MODEL and len(self.components) < 2:
            raise SpecError(f"{self.hypothesis_id}: mixture needs >= 2 components")
        if self.status not in HypothesisStatus.__members__:
            raise SpecError(f"{self.hypothesis_id}: unknown status {self.status!r}")


def mixture_of(hid: str, components: list[HypothesisSpec], *, claim: str) -> HypothesisSpec:
    """Compile a mixture hypothesis from components.

    A mixture predicts POS wherever any component predicts POS (effects add), NEG where any predicts NEG
    and none POS, NULL only where every committed component predicts NULL, ANY otherwise. The mixture's
    predictions are SUPPORTIVE (a mixture is harder to falsify; the adjudicator applies a complexity
    penalty and only prefers it when single components are each contradicted).
    """
    tests = sorted({p.test_id for c in components for p in c._preds()})
    preds = []
    for t in tests:
        calls = [c.prediction_for(t).expected for c in components if c.prediction_for(t) is not None and c.prediction_for(t).expected != Call.ANY]
        if not calls:
            exp = Call.ANY
        elif Call.POS in calls:
            exp = Call.POS
        elif Call.NEG in calls:
            exp = Call.NEG
        elif all(x == Call.NULL for x in calls) and len(calls) == len(components):
            exp = Call.NULL
        else:
            exp = Call.ANY
        kind = PredictionKind.NON_DISCRIMINATING_OBSERVATION if exp == Call.ANY else PredictionKind.SUPPORTIVE_PREDICTION
        preds.append(Prediction(test_id=t, expected=exp, kind=kind, rationale="compiled from components"))
    return HypothesisSpec(
        hypothesis_id=hid, generation=int(hid.partition(".g")[2]), claim=claim,
        mechanism_class=MechanismClass.MIXTURE_MODEL,
        causal_entities=sorted({e for c in components for e in c.causal_entities}),
        causal_relations=sorted({r for c in components for r in c.causal_relations}),
        required_observables=sorted({o for c in components for o in c.required_observables}),
        predicted_signatures=[f"union of {', '.join(c.hypothesis_id for c in components)}"],
        predicted_absences=[], falsifiers=["every compiled POS signature observed NULL on confirmation data"],
        predictions=preds, known_confounds=sorted({k for c in components for k in c.known_confounds}),
        alternative_hypotheses=[c.hypothesis_id for c in components],
        identifiability_conditions=["each component must be separately identified by at least one test"],
        required_controls=sorted({k for c in components for k in c.required_controls}),
        allowed_data_partitions=sorted(set.intersection(*[set(c.allowed_data_partitions) for c in components])),
        forbidden_data_partitions=sorted(set.union(*[set(c.forbidden_data_partitions) for c in components])),
        components=[c.hypothesis_id for c in components],
    )


# ----------------------------------------------------------------------------------------------
# test
# ----------------------------------------------------------------------------------------------
@dataclass
class MechanismTestSpec:
    test_id: str
    version: str
    question: str
    distinguishes: list                 # hypothesis ids (families or full ids) the test claims to separate
    data_requirements: list
    covariates: list
    matching_variables: list
    independence_unit: str
    statistic_family: str               # contamination accounting key
    partition_policy: dict              # {"discovery": "DISCOVERY", "confirmation": "CONFIRMATION"}
    preprocessing: list
    statistical_model: str
    hostile_null: str
    effect_metric: str
    min_effect: float                   # effect floor (|effect| below this is not meaningful)
    uncertainty_estimator: str
    stopping_rule: str
    multiple_testing_family: str
    min_units: int                      # minimum independent units (animals) to decide
    forbidden_leakage_paths: list
    required_measurement_fields: list = field(default_factory=list)
    cost: float = 1.0                   # relative compute/IO cost
    confound_risk: float = 0.2          # 0..1

    def content(self) -> dict:
        return asdict(self)

    @property
    def spec_hash(self) -> str:
        return canonical_sha(self.content())

    def validate(self) -> None:
        if not self.test_id or not self.version:
            raise SpecError("test_id and version are required")
        if self.min_effect <= 0:
            raise SpecError(f"{self.test_id}: min_effect must be > 0 (effect sizes before p-values)")
        if self.min_units < 2:
            raise SpecError(f"{self.test_id}: min_units must be >= 2 independent units")
        if not self.hostile_null.strip():
            raise SpecError(f"{self.test_id}: a hostile null is required")
        if not self.independence_unit.strip():
            raise SpecError(f"{self.test_id}: independence unit must be explicit")
        if not self.multiple_testing_family.strip():
            raise SpecError(f"{self.test_id}: multiple-testing family must be declared")
        if len(self.distinguishes) < 2:
            raise SpecError(f"{self.test_id}: must name >= 2 hypotheses it distinguishes")


def compile_test_against_hypotheses(test: MechanismTestSpec, hypotheses: list[HypothesisSpec]) -> dict:
    """Compiler check: does this test actually separate the hypotheses it claims to?

    Returns the predicted outcome per claimed hypothesis and the disagreeing pairs. Raises SpecError
    if no pair of claimed hypotheses makes different committed predictions.
    """
    test.validate()
    by_family = {}
    for h in hypotheses:
        by_family.setdefault(h.family, h)
        by_family[h.hypothesis_id] = h
    preds = {}
    for name in test.distinguishes:
        h = by_family.get(name)
        if h is None:
            raise SpecError(f"{test.test_id}: claims to distinguish unknown hypothesis {name!r}")
        p = h.prediction_for(test.test_id)
        preds[h.hypothesis_id] = p.expected if p is not None else Call.ANY
    ids = sorted(preds)
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
             if preds[a] != Call.ANY and preds[b] != Call.ANY and preds[a] != preds[b]]
    if not pairs:
        raise SpecError(f"{test.test_id}: non-discriminating — claimed hypotheses make identical or uncommitted predictions {preds}")
    return {"test_id": test.test_id, "predictions": preds, "disagreeing_pairs": pairs, "spec_hash": test.spec_hash}


# ----------------------------------------------------------------------------------------------
# outcome
# ----------------------------------------------------------------------------------------------
@dataclass
class TestOutcome:
    __test__ = False
    test_id: str
    test_spec_hash: str
    stage: str                  # Stage
    partition: str
    status: str                 # OutcomeStatus
    call: str                   # Call (POS/NULL/NEG/INCONCLUSIVE)
    estimate: float | None
    ci: list | None
    min_effect: float
    n_units: int                # independent units (animals)
    n_observations: int
    unit_ids: list              # animal ids used (for evidence-family overlap)
    feature_family: str         # statistic family
    contamination_grade: str    # UNTOUCHED / PARTIAL / CONTAMINATED / SYNTHETIC / HISTORICAL
    details: dict = field(default_factory=dict)
    derived_from: list = field(default_factory=list)   # evidence ids this outcome reuses
    outcome_id: str = ""

    def finalize(self) -> "TestOutcome":
        body = asdict(self)
        body.pop("outcome_id")
        self.outcome_id = "O-" + canonical_sha(body)[:16]
        return self

    def validate(self) -> None:
        if self.status not in OutcomeStatus.__members__:
            raise SpecError(f"{self.test_id}: bad outcome status {self.status!r}")
        if self.status == OutcomeStatus.DECIDED and self.call not in (Call.POS, Call.NULL, Call.NEG):
            raise SpecError(f"{self.test_id}: DECIDED outcome must carry POS/NULL/NEG, got {self.call!r}")
        if self.status != OutcomeStatus.DECIDED and self.call in (Call.POS, Call.NULL, Call.NEG):
            raise SpecError(f"{self.test_id}: non-DECIDED outcome cannot carry a decided call")
        if self.status == OutcomeStatus.DECIDED and (self.ci is None or len(self.ci) != 2):
            raise SpecError(f"{self.test_id}: DECIDED outcome requires an uncertainty interval")


def call_from_ci(ci, floor: float) -> str:
    lo, hi = ci
    if lo > floor:
        return Call.POS
    if hi < -floor:
        return Call.NEG
    if lo >= -floor and hi <= floor:
        return Call.NULL
    return Call.INCONCLUSIVE


def dumps(obj) -> str:
    def enc(o):
        if hasattr(o, "content"):
            return o.content()
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        if isinstance(o, Enum):
            return o.value
        raise TypeError(type(o))
    return json.dumps(obj, default=enc, sort_keys=True, indent=1)
