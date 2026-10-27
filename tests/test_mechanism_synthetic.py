"""Mechanism engine synthetic acceptance suite (S1-S10): known ground truth through the full engine."""
from __future__ import annotations

import numpy as np
import pytest

from wormlab.mechanism import contrasts, engine, generators as G, measurement as M
from wormlab.mechanism.adjudicator import adjudicate
from wormlab.mechanism.evidence_graph import EvidenceGraph
from wormlab.mechanism.firewall import AccessBroker, AccessLedger, AccessViolation, ContaminationViolation, FreezeRegistry, FreezeViolation, Role
from wormlab.mechanism.randi import benchmark as B
from wormlab.mechanism.spec import Call, HypothesisSpec, OutcomeStatus, Prediction, TestOutcome
from wormlab.mechanism.synthetic import SyntheticProvider

MANIFEST = {"manifest_sha256": "synthetic", "partitions": {"DISCOVERY": [], "CONFIRMATION": [], "RESERVE": []}}


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(contrasts, "N_BOOT", 150)
    monkeypatch.setattr(contrasts, "N_PERM", 60)


def _synthetic_tests():
    return [t for t in B.tests() if t.test_id != "X1_nucleus_vs_background"]


def _run(tmp_path, *, hypotheses=None, tests=None, policy="1.0", **provider_kw):
    prov = SyntheticProvider(**provider_kw)
    run = engine.MechanismRun(tmp_path / "run", hypotheses or B.hypotheses(), tests or _synthetic_tests(), B.measurement_model(),
                              MANIFEST, [], prov, engine.RunConfig(max_judge_steps=16, policy=policy))
    res = run.run()
    run.write_outputs()
    return run, res["verdict"]


def _state(v, fam):
    return v["hypotheses"][fam + ".g0"]["state"]


def _call(run, test_id):
    conf = [o for o in run.outcomes if o.test_id == test_id and o.partition == "CONFIRMATION"]
    disc = [o for o in run.outcomes if o.test_id == test_id and o.partition == "DISCOVERY"]
    return (conf or disc)[-1]


# ----------------------------------------------------------------------------------------------------
def test_S1_pure_optical(tmp_path):
    run, v = _run(tmp_path, seed=101, p_opt=0.8, sigma_opt_um=3.0)
    assert _call(run, "T0_proximity_effect").call == Call.POS
    assert _call(run, "T1_distance_matched_topology").call in (Call.NULL, Call.INCONCLUSIVE)   # no topology effect at fixed distance
    assert _call(run, "T6_target_flat_nearfield").call == Call.POS
    assert _call(run, "T3_onset_latency").call in (Call.NULL, Call.INCONCLUSIVE)
    assert v["verdict"] == "MECHANISM_SUPPORTED" and v["leading_hypothesis"] == "H2.g0", v["verdict"]
    for fam in ("H1", "H3", "H4", "H9"):
        assert _state(v, fam) in ("FALSIFIED_WITHIN_MODEL", "STRONGLY_DISFAVORED")


def test_S2_pure_direct_network(tmp_path):
    run, v = _run(tmp_path, seed=202, p_chem=0.6)
    assert _call(run, "T1_distance_matched_topology").call == Call.POS
    assert _call(run, "T3_onset_latency").call == Call.POS
    assert _call(run, "T6_target_flat_nearfield").call in (Call.NULL, Call.INCONCLUSIVE)
    assert _state(v, "H2") in ("FALSIFIED_WITHIN_MODEL", "STRONGLY_DISFAVORED")
    assert _state(v, "H4") in ("SUPPORTED", "WEAKLY_SUPPORTED")
    assert v["verdict"] in ("MECHANISM_SUPPORTED", "NON_IDENTIFIABLE_FROM_AVAILABLE_DATA", "NEW_EXPERIMENT_REQUIRED")
    if v["verdict"] == "MECHANISM_SUPPORTED":
        assert v["leading_hypothesis"] == "H4.g0"


def test_S3_polysynaptic(tmp_path):
    run, v = _run(tmp_path, seed=303, p_poly=0.5)
    t7 = _call(run, "T7_path_length")
    t1 = _call(run, "T1_distance_matched_topology")
    assert t7.call == Call.POS
    assert t1.estimate is None or t1.estimate < t7.estimate          # direct edge underperforms higher-order paths
    assert _call(run, "T3_onset_latency").call == Call.POS
    assert _state(v, "H2") in ("FALSIFIED_WITHIN_MODEL", "STRONGLY_DISFAVORED")
    assert _state(v, "H5") not in ("FALSIFIED_WITHIN_MODEL", "STRONGLY_DISFAVORED")


def test_S4_extraction_contamination(tmp_path):
    run, v = _run(tmp_path, seed=404, contam=0.6, contam_sigma_um=3.0, p_spont=0.05)
    assert _call(run, "T11_baseline_covariance").call == Call.POS          # mask/measurement signature exposes it
    assert _call(run, "T6_target_flat_nearfield").call in (Call.NULL, Call.INCONCLUSIVE)
    assert _state(v, "H1") in ("SUPPORTED", "WEAKLY_SUPPORTED")
    for fam in ("H2", "H9", "H6"):
        assert _state(v, fam) in ("FALSIFIED_WITHIN_MODEL", "STRONGLY_DISFAVORED", "WEAKENED")


def test_S5_mixture(tmp_path):
    run, v = _run(tmp_path, seed=505, p_opt=0.6, sigma_opt_um=2.5, p_chem=0.7)
    assert _state(v, "H2") in ("FALSIFIED_WITHIN_MODEL", "STRONGLY_DISFAVORED", "WEAKENED")
    assert _state(v, "H4") in ("FALSIFIED_WITHIN_MODEL", "STRONGLY_DISFAVORED", "WEAKENED")
    assert v["verdict"] == "MULTIPLE_MECHANISMS_REQUIRED", (v["verdict"], v["mixtures"])
    assert any(set(m["components"]) == {"H2.g0", "H4.g0"} for m in v["mixtures"])


def test_S6_non_identifiable(tmp_path):
    hyps = [h for h in B.hypotheses() if h.family == "H2"]
    h3 = [h for h in B.hypotheses() if h.family == "H3"][0]
    h3.predictions = [Prediction(**{**p.__dict__, "expected": "POS", "kind": "NECESSARY_PREDICTION"}) if p.test_id == "T0_proximity_effect" else p
                      for p in h3.predictions]
    hyps.append(h3)
    keep = {"T0_proximity_effect", "T3_onset_latency", "T5_control_channel", "T10_farfield_breadth"}
    tests = [t for t in _synthetic_tests() if t.test_id in keep]
    for world in (dict(p_opt=0.8, sigma_opt_um=3.0), dict(p_gap=0.8, conn_kw=dict(gap_nearest_only=True, p_chem=0.0))):
        run, v = _run(tmp_path / str(len(world)), hypotheses=hyps, tests=tests, seed=606, n_sham=0, p_target=1.0, drop_labels=True, **world)
        assert v["verdict"] == "NON_IDENTIFIABLE_FROM_AVAILABLE_DATA", (world, v["verdict"])
        assert ["H2.g0", "H3.g0"] in v["non_identifiable_pairs"]


def _broker():
    ledger = AccessLedger()
    reg = FreezeRegistry(ledger)
    return AccessBroker(MANIFEST, ledger, reg), reg, ledger


def test_S7_adaptive_leak_attack():
    broker, reg, ledger = _broker()
    t = B.tests()[1]
    for h in B.hypotheses():
        reg.freeze_hypothesis(h, by="c")
    reg.freeze_test(t, by="c")
    for role in (Role.SCOUT, Role.COMPILER, Role.AUDITOR):
        with pytest.raises(AccessViolation):
            broker.request(role=role, actor="attacker", partition="CONFIRMATION", test=t, hypothesis_ids=["H1.g0"])
    unfrozen = B.tests()[2]
    with pytest.raises(AccessViolation):
        broker.request(role=Role.JUDGE, actor="judge", partition="CONFIRMATION", test=unfrozen, hypothesis_ids=["H1.g0"])
    denied = [e for e in ledger.entries if e.get("granted") is False]
    assert len(denied) == 4
    ledger.verify()


def test_S8_hypothesis_revision_attack():
    broker, reg, ledger = _broker()
    t = B.tests()[1]
    h5 = [h for h in B.hypotheses() if h.family == "H5"][0]
    reg.freeze_hypothesis(h5, by="c")
    reg.freeze_test(t, by="c")
    broker.request(role=Role.JUDGE, actor="judge", partition="CONFIRMATION", test=t, hypothesis_ids=["H5.g0"])
    # single look: the same lineage cannot re-query the same family
    with pytest.raises(ContaminationViolation):
        broker.request(role=Role.JUDGE, actor="judge", partition="CONFIRMATION", test=t, hypothesis_ids=["H5.g0"])
    # editing a frozen generation in place is refused
    edited = HypothesisSpec(**{**h5.__dict__, "claim": h5.claim + " (revised)"})
    with pytest.raises(FreezeViolation):
        reg.freeze_hypothesis(edited, by="c")
    # a revised generation cannot reuse the inspected confirmation data
    g1 = HypothesisSpec(**{**h5.__dict__, "hypothesis_id": "H5.g1", "generation": 1, "claim": h5.claim + " (revised)",
                           "parent_hypothesis_ids": ["H5.g0"], "created_from_evidence_ids": ["O-demo"]})
    reg.freeze_hypothesis(g1, by="c")
    with pytest.raises(ContaminationViolation):
        broker.request(role=Role.JUDGE, actor="judge", partition="CONFIRMATION", test=t, hypothesis_ids=["H5.g1"])
    with pytest.raises(AccessViolation):
        broker.request(role=Role.JUDGE, actor="judge", partition="RESERVE", test=t, hypothesis_ids=["H5.g1"])
    broker.release_reserve(actor="auditor", hypothesis_ids=["H5.g1"], reason="untouched data for revised generation")
    assert broker.request(role=Role.JUDGE, actor="judge", partition="RESERVE", test=t, hypothesis_ids=["H5.g1"])["granted"]


def test_S9_scan_timing_artifact():
    rng = np.random.default_rng(9)
    lags_naive, intervals = [], []
    for k in range(200):
        t_on = 20.0 + rng.uniform(0, 0.5)
        vols = np.arange(30, 70)
        a = t_on + 0.01                                          # simultaneous activation of both cells
        ts_t = vols * 0.5 + 24 * 0.02                           # target imaged at the END of each sweep
        ts_r = vols * 0.5 + 0 * 0.02                            # receiver imaged at the START
        f = lambda ts: 200 * (1 + G._calcium(ts, a, 1.5)) * (1 + rng.normal(0, 0.02, len(ts)))
        yt, yr = f(ts_t), f(ts_r)
        sv = int(np.searchsorted(vols * 0.5, t_on))
        nt = M.naive_volume_onset(vols, yt, sv, n_base=8)
        nr = M.naive_volume_onset(vols, yr, sv, n_base=8)
        if nt is not None and nr is not None:
            lags_naive.append(nr - nt)
        it = M.onset_interval(ts_t, yt, t_on, n_base=8)
        ir = M.onset_interval(ts_r, yr, t_on, n_base=8)
        if it and ir:
            intervals.append(M.delay_interval(ir, it))
    assert np.mean(np.array(lags_naive) > 0) > 0.5                 # naive estimator manufactures a lag
    agg = M.delay_call(intervals)
    assert agg["frac_lag"] < 0.05                                   # corrected intervals do not claim precedence


def test_S10_evidence_double_counting():
    g = EvidenceGraph()
    h = [x for x in B.hypotheses() if x.family == "H2"][0]
    t = [x for x in B.tests() if x.test_id == "T0_proximity_effect"][0]
    outs = []
    for k in range(5):
        o = TestOutcome("T0_proximity_effect", t.spec_hash, "CONFIRMATION", "CONFIRMATION", OutcomeStatus.DECIDED, Call.POS, 0.2 + k * 0.01,
                        [0.1, 0.3], 0.03, 10, 1000, [f"a{i}" for i in range(10)], "proximity_rate", "UNTOUCHED",
                        details={"variant": k}).finalize()
        g.add_outcome(o, data_stream="same_stream")
        g.relate_hypothesis(o.outcome_id, h.hypothesis_id, "supports", kind="NECESSARY_PREDICTION", grade="UNTOUCHED")
        outs.append(o)
    s = g.support_summary()[h.hypothesis_id]["supports"]
    assert s == {"analyses": 5, "independent_families": 1}
    v = adjudicate([h], outs, g, phenomenon_test=None, discrimination_report={"status": "NO_AVAILABLE_TEST_HAS_DISCRIMINATING_POWER", "candidates": []})
    assert v["hypotheses"][h.hypothesis_id]["state"] == "WEAKLY_SUPPORTED"      # one family cannot make SUPPORTED


def test_S11_policy_all_weakened_is_not_disfavored():
    """Every hypothesis carries only weak/exploratory contradictions: policy 1.0 says DISFAVORED, 1.1 stays uncommitted."""
    H = [h for h in B.hypotheses() if h.family in ("H1", "H2")]
    g = EvidenceGraph()
    t = {x.test_id: x for x in B.tests()}
    outs = []
    # T4 NULL (untouched confirmation): weak contradiction of H2 (SUPPORTIVE POS); T11 POS contradicts H2 weakly; X1 POS exploratory strong vs H1
    for tid, call, grade, stage in (("T4_offset_directionality", "NULL", "UNTOUCHED", "CONFIRMATION"), ("T11_baseline_covariance", "POS", "UNTOUCHED", "CONFIRMATION"),
                                    ("X1_nucleus_vs_background", "POS", "CONTAMINATED", "HISTORICAL")):
        o = TestOutcome(tid, t[tid].spec_hash, stage, stage, OutcomeStatus.DECIDED, call, 0.1, [0.06, 0.2], 0.03, 20, 100,
                        [f"u{i}" for i in range(20)] if stage != "HISTORICAL" else ["h"], t[tid].statistic_family, grade).finalize()
        g.add_outcome(o, data_stream="s")
        outs.append(o)
    rep = {"status": "NO_AVAILABLE_TEST_HAS_DISCRIMINATING_POWER", "candidates": []}
    v10 = adjudicate(H, outs, g, discrimination_report=rep, policy="1.0")
    v11 = adjudicate(H, outs, g, discrimination_report=rep, policy="1.1")
    assert {a["state"] for a in v11["hypotheses"].values()} == {"WEAKENED"}
    assert v10["verdict"] == "MECHANISM_DISFAVORED"
    assert v11["verdict"] in ("NEW_EXPERIMENT_REQUIRED", "NON_IDENTIFIABLE_FROM_AVAILABLE_DATA")


def test_S12_frame_cosampling_artifact_removed_by_plane_matching():
    coh = G.generate_cohort(1212, 16, frame_noise=0.08, noise=0.03)
    from wormlab.mechanism import observations as O
    tab = O.concat([O.build_table(p, a, conn) for a, p, conn in coh])
    raw = contrasts.baseline_covariance(tab, seed=1)
    matched = contrasts.baseline_covariance_plane_matched(tab, seed=1)
    assert raw["ci"][0] > 0.05                                  # naive near-far covariance manufactured by shared frames
    assert matched["ci"][0] >= -0.05 and matched["ci"][1] <= 0.05   # plane matching removes it


def test_S13_supported_only_by_shared_predictions_is_not_mechanism_supported():
    """Reproduces the g1 defect: leader supported only by SUPPORTIVE predictions it shares with a rival whose only contradiction
    is exploratory, while the leader's NECESSARY discriminating prediction is undecided."""
    H = [h for h in B.hypotheses_g1(["O-x"]) if h.family in ("H1", "H3")]
    tests = {t.test_id: t for t in B.tests_g1()}
    g, outs = EvidenceGraph(), []
    spec = [("T11_baseline_covariance", "POS", "CONFIRMATION", "UNTOUCHED", "DECIDED"),
            ("T11b_baseline_covariance_plane_matched", "POS", "CONFIRMATION", "UNTOUCHED", "DECIDED"),
            ("T4_offset_directionality", "NULL", "CONFIRMATION", "UNTOUCHED", "DECIDED"),
            ("T10_farfield_breadth", "NULL", "CONFIRMATION", "UNTOUCHED", "DECIDED"),
            ("X1_nucleus_vs_background", "POS", "HISTORICAL", "CONTAMINATED", "DECIDED"),
            ("T1_distance_matched_topology", "INCONCLUSIVE", "CONFIRMATION", "PARTIAL", "INCONCLUSIVE")]
    for tid, call, stage, grade, status in spec:
        o = TestOutcome(tid, tests[tid].spec_hash, stage, stage, status, call, 0.1, [0.05, 0.2] if status == "DECIDED" else [-0.01, 0.05],
                        tests[tid].min_effect, 40, 400, [f"u{i}" for i in range(40)] if stage != "HISTORICAL" else ["h"], tests[tid].statistic_family, grade).finalize()
        g.add_outcome(o, data_stream="s")
        outs.append(o)
    rep = {"status": "NO_AVAILABLE_TEST_HAS_DISCRIMINATING_POWER", "candidates": []}
    v11 = adjudicate(H, outs, g, discrimination_report=rep, policy="1.1")
    v12 = adjudicate(H, outs, g, discrimination_report=rep, policy="1.2")
    assert v11["verdict"] == "MECHANISM_SUPPORTED" and v11["leading_hypothesis"] == "H3.g1"      # the defect, reproduced
    assert v12["verdict"] in ("NON_IDENTIFIABLE_FROM_AVAILABLE_DATA", "NEW_EXPERIMENT_REQUIRED")          # never MECHANISM_SUPPORTED
    assert v12["hypotheses"]["H3.g1"]["state"] == "WEAKLY_SUPPORTED"                               # no NECESSARY prediction confirmed


@pytest.mark.parametrize("world,leader", [(dict(seed=101, p_opt=0.8, sigma_opt_um=3.0), "H2.g0")])
def test_S1_policy_1_2_still_supports_when_data_separate(tmp_path, world, leader):
    run, v = _run(tmp_path, policy="1.2", **world)
    assert v["verdict"] == "MECHANISM_SUPPORTED" and v["leading_hypothesis"] == leader, (v["verdict"], v["non_identifiable_pairs"], v["notes"])


def test_design_receives_single_mechanism_survivors_under_policy_1_2(tmp_path):
    """Regression (found in the g2 real run): proposals must be generated from surviving single mechanisms, not mixtures."""
    hyps = [h for h in B.hypotheses() if h.family in ("H1", "H3")]
    keep = {"T0_proximity_effect", "T3_onset_latency", "T10_farfield_breadth"}
    tests = [t for t in _synthetic_tests() if t.test_id in keep]
    run, v = _run(tmp_path, hypotheses=hyps, tests=tests, policy="1.2", seed=909, contam=0.5, n_sham=0, drop_labels=True)
    assert run.result["proposal"]["status"] in ("PROPOSED", "NO_MANIPULATION_SEPARATES", "NOT_REQUIRED")
    if v["verdict"] in ("NEW_EXPERIMENT_REQUIRED", "NON_IDENTIFIABLE_FROM_AVAILABLE_DATA"):
        assert run.result["proposal"]["status"] != "NO_PROPOSAL_NEEDED"
