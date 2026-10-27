"""Unit checks for mechanism specs, compiler, measurement model and capsule tamper detection."""
from __future__ import annotations

import copy
import json
import shutil

import pytest

from wormlab.mechanism.measurement import MeasurementField, MeasurementModelSpec, UNKNOWN
from wormlab.mechanism.randi import benchmark as B
from wormlab.mechanism.spec import HypothesisSpec, Prediction, SpecError, compile_test_against_hypotheses, mixture_of


def test_benchmark_hypotheses_validate_and_are_judgeable():
    for h in B.hypotheses():
        h.validate()
        assert any(p.counts_against_when() for p in h._preds())
        assert h.discriminating_tests


def test_unjudgeable_hypothesis_rejected():
    h = copy.deepcopy(B.hypotheses()[0])
    h.predictions = [Prediction(p.test_id, "ANY", "NON_DISCRIMINATING_OBSERVATION") for p in h._preds()]
    with pytest.raises(SpecError, match="not judgeable"):
        h.validate()


def test_generation_requires_parent_and_evidence():
    h = copy.deepcopy(B.hypotheses()[1])
    h.hypothesis_id, h.generation = "H2.g1", 1
    with pytest.raises(SpecError, match="parent"):
        h.validate()
    h.parent_hypothesis_ids = ["H2.g0"]
    with pytest.raises(SpecError, match="evidence"):
        h.validate()
    h.created_from_evidence_ids = ["O-x"]
    h.validate()


def test_compiler_rejects_non_discriminating_test():
    H = [h for h in B.hypotheses() if h.family in ("H2", "H3", "H4")]
    t5 = next(t for t in B.tests() if t.test_id == "T5_control_channel")
    t5 = copy.deepcopy(t5)
    t5.distinguishes = ["H2", "H3", "H4"]          # all predict NULL for the red channel
    with pytest.raises(SpecError, match="non-discriminating"):
        compile_test_against_hypotheses(t5, H)


def test_all_benchmark_tests_compile():
    H = B.hypotheses()
    for t in B.tests():
        assert compile_test_against_hypotheses(t, H)["disagreeing_pairs"]


def test_mixture_compilation():
    H = {h.family: h for h in B.hypotheses()}
    m = mixture_of("H8[H2+H4].g0", [H["H2"], H["H4"]], claim="m")
    m.validate()
    assert m.prediction_for("T1_distance_matched_topology").expected == "POS"
    assert m.prediction_for("T5_control_channel").expected == "NULL"


def test_measurement_unknown_cannot_carry_value():
    m = MeasurementModelSpec("x", "y", {"a": MeasurementField(1.0, UNKNOWN)})
    with pytest.raises(SpecError):
        m.validate()
    B.measurement_model().validate()


def test_capsule_one_byte_tamper_detected(tmp_path):
    from wormlab.evidence.capsule import ArtifactSpec, create_capsule, verify
    f = tmp_path / "a.json"
    f.write_text(json.dumps({"x": 1}))
    create_capsule(tmp_path / "cap", capsule_id="t", replay_command="none", summary=ArtifactSpec("summary", "a.json", f))
    assert bool(verify(tmp_path / "cap"))
    blob = next((tmp_path / "cap" / "store").iterdir())
    blob.chmod(0o644)
    b = bytearray(blob.read_bytes()); b[0] ^= 1; blob.write_bytes(bytes(b))
    assert not bool(verify(tmp_path / "cap"))
