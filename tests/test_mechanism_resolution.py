"""Deterministic checks for the experiment-resolution model (src/wormlab/mechanism/resolution.py).

Every assertion encodes the COMPUTED behaviour of the model (see results/mechanism/deep_campaign/resolution/), not a
hoped-for result. Seeded; total runtime well under 60 s.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from wormlab.mechanism import resolution as R
from wormlab.mechanism.measurement import ASSUMED, delay_call, onset_interval
from wormlab.mechanism.spec import SpecError


def _spec(**kw):
    return R.default_spec(**kw)


# ---------------------------------------------------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------------------------------------------------
def test_spec_validates_and_hash_is_stable_and_sensitive():
    a, b = _spec(), _spec()
    a.validate()
    assert a.spec_hash == b.spec_hash and len(a.spec_hash) == 64
    c = replace(a, biology=replace(a.biology, coupling_latency_s=0.05))
    assert c.spec_hash != a.spec_hash
    assert replace(a, n_animals=11).spec_hash != a.spec_hash


@pytest.mark.parametrize("bad", [
    dict(readout="telepathy"),
    dict(n_animals=0),
])
def test_spec_rejects_invalid_fields(bad):
    with pytest.raises(SpecError):
        replace(_spec(), **bad).validate()


def test_spec_rejects_neighbour_response_not_below_target():
    s = _spec()
    with pytest.raises(SpecError, match="neighbour_response_dff"):
        replace(s, biology=replace(s.biology, neighbour_response_dff=0.6)).validate()


def test_indicator_presets_are_assumed_with_ranges_and_sweepable():
    for name in R.INDICATOR_PRESETS:
        ind = R.indicator_preset(name)
        ind.validate()
        for k in ("tau_rise_s", "tau_decay_s", "hill_n", "kd_nM", "tau_nuclear_s", "dff_max"):
            m = ind.provenance[k]
            assert m["status"] == ASSUMED and m["source"]
            lo, hi = m["range"]
            assert lo <= getattr(ind, k) <= hi
    nuc = R.indicator_preset("nuclear-GCaMP6s")
    assert nuc.at_bound("tau_nuclear_s", "high").tau_nuclear_s == 3.0
    assert [i.tau_rise_s for i in nuc.sweep("tau_rise_s", [0.1, 0.3])] == [0.1, 0.3]
    with pytest.raises(SpecError):
        R.indicator_preset("GCaMP99")


def test_randi_acquisition_alternating_sweep_timing():
    acq = R.acquisition_preset("randi2023")
    period, offs = acq.plane_offsets()
    assert period == pytest.approx(1.0)
    # middle plane is sampled every 0.5 s; the turn-around plane alternates 0.98 / 0.02 s
    assert np.diff(offs[12]).tolist() == pytest.approx([0.5])
    assert sorted([np.diff(offs[0])[0], period - np.diff(offs[0])[0]]) == pytest.approx([0.02, 0.98])


# ---------------------------------------------------------------------------------------------------------------------
# forward model + lag estimation
# ---------------------------------------------------------------------------------------------------------------------
def test_calibration_matches_rival_amplitudes():
    s = _spec()
    for mech in ("H1", "H3", "H4"):
        tm = R.build_templates(s, mech)
        assert tm.target_unit.max() == pytest.approx(s.biology.target_response_dff, rel=0.02)
        if mech != "H1":
            assert tm.neighbour_unit.max() == pytest.approx(s.biology.neighbour_response_dff, rel=0.02)
    cal = R.calibrate(s, "H1")
    r = s.biology.brightness_ratio
    assert cal["kappa"] / (r + cal["kappa"]) * cal["target_peak_ref"] == pytest.approx(s.biology.neighbour_response_dff, rel=1e-6)


def test_h1_contamination_gives_zero_lag_at_randi_acquisition():
    s = _spec(n_events_pooled=500)
    rng = np.random.default_rng(11)
    ref = R.reference_shape(s)
    d = np.array([R.lag_estimate(s, R.simulate_events(s, "H1", 500, rng), ref)["delta"] for _ in range(8)])
    # computed: MC SD 7.7 ms, CRB SE 7.0 ms at N=500
    assert abs(d.mean()) < 0.010
    assert np.all(np.abs(d) < 0.030)


def test_large_injected_lag_detected_at_fast_acquisition():
    s = R.default_spec(acquisition="single_plane_200hz", indicator="jGCaMP7f", n_events_pooled=100)
    s = replace(s, biology=replace(s.biology, coupling_latency_s=0.100, coupling_tau_s=0.0))
    rng = np.random.default_rng(3)
    est = R.lag_estimate(s, R.simulate_events(s, "H3", 100, rng))["delta"]
    se = R.lag_crb(s)["se_per_event_s"] / math.sqrt(100)
    assert se < 0.005
    assert est - 1.96 * se > 0.05
    assert est == pytest.approx(R.expected_lag(s, "H3"), abs=5 * se)


def test_20ms_lag_not_resolvable_per_pair_or_by_onset_intervals_at_randi_but_pooling_recovers_it():
    """Computed truth for nuclear GCaMP6s at 25 planes x 20 ms, alternating sweeps:
    per pair (5 events) the 20 ms lag is invisible; the engine's threshold onset-interval estimator has ~1 s resolution;
    pooling hundreds of stimulus-phase-dithered events recovers ~20 ms detection statistically (systematics aside)."""
    s = _spec()
    per_pair = R.lag_crb(s, target_plane=12, seed=1)["se_per_event_s"] / math.sqrt(5)
    assert R.lag_power_from_se(per_pair, 0.020)["detection"] < 0.15
    assert R.lag_power_from_se(per_pair, 0.020)["separation"] == 0.0
    pooled = R.lag_crb(s, seed=1)["se_per_event_s"] / math.sqrt(500)
    assert R.mde_lag(pooled)["detection_s"] < 0.03
    assert R.mde_lag(pooled)["separation_s"] > 0.02          # a 20 ms lag is NOT separable engine-style even at N=500
    # engine T3 onset-interval estimator on simulated H3 (20 ms) events
    s3 = replace(s, biology=replace(s.biology, coupling_latency_s=0.02))
    b = R.simulate_events(s3, "H3", 120, np.random.default_rng(5))
    iv = []
    for e in range(120):
        oT = onset_interval(b.tT[e][b.mT[e]], b.yT[e][b.mT[e]], 0.0, n_base=int((b.tT[e][b.mT[e]] < 0).sum()))
        oN = onset_interval(b.tN[e][b.mN[e]], b.yN[e][b.mN[e]], 0.0, n_base=int((b.tN[e][b.mN[e]] < 0).sum()))
        if oT and oN:
            iv.append((oN[0] - oT[1], oN[1] - oT[0]))
    assert len(iv) > 30
    assert delay_call(iv)["resolution_s"] > 0.5


def test_crb_agrees_with_monte_carlo_at_randi():
    s = _spec(n_events_pooled=200)
    rng = np.random.default_rng(21)
    ref = R.reference_shape(s)
    d = np.array([R.lag_estimate(s, R.simulate_events(s, "H1", 200, rng), ref)["delta"] for _ in range(30)])
    crb = R.lag_crb(s)["se_per_event_s"] / math.sqrt(200)
    assert 0.6 < d.std(ddof=1) / crb < 1.8


# ---------------------------------------------------------------------------------------------------------------------
# contrasts
# ---------------------------------------------------------------------------------------------------------------------
def test_sparse_expression_nonexpressing_neighbour_signal_is_optical_only():
    s = replace(_spec(n_animals=4, n_pairs_per_animal=4, n_events_per_pair=3), readout="sparse_neighbour_nonexpressing")
    rng = np.random.default_rng(8)
    resp = {}
    for mech in ("H1", "H3"):
        b = R.simulate_events(s, mech, 48, rng)
        post = b.mN & (b.tN > 0.25) & (b.tN < 3.0)
        pre = b.mN & (b.tN < 0)
        per = np.where(post, b.cN, 0).sum(1) / post.sum(1) - np.where(pre, b.cN, 0).sum(1) / pre.sum(1)
        resp[mech] = (per.mean(), per.std(ddof=1) / math.sqrt(len(per)))
    assert resp["H1"][0] > 5 * resp["H1"][1]                 # target photons leak into the non-expressing ROI
    assert abs(resp["H3"][0]) < 3 * resp["H3"][1]            # no indicator in the neighbour -> nothing to see
    r = R.ResolutionCompiler().compile(replace(s, readout="both_fluorescent"), ("H1", "H3"), "sparse_expression", reps=12)
    assert r["mean_ratio"]["H1"] == pytest.approx(1.0, abs=0.25)
    assert abs(r["mean_ratio"]["H3"]) < 0.1
    assert r["power_separation"] >= 0.8


def test_two_colour_inverts_the_sparse_logic():
    s = _spec(n_animals=4, n_pairs_per_animal=3, n_events_per_pair=3)
    r = R.ResolutionCompiler().compile(s, ("H1", "H3"), "two_colour", reps=12)
    assert abs(r["mean_ratio"]["H1"]) < 0.15
    assert r["mean_ratio"]["H3"] == pytest.approx(1.0, abs=0.3)


def test_genotype_contrast_sample_size_and_calls():
    s = _spec(n_animals=40)
    gm = s.genotype_model
    assert R.genotype_predicted_reduction(gm, "H1") == 0.0
    assert R.genotype_predicted_reduction(gm, "H3") == pytest.approx(0.083 * 0.8)
    r = R.ResolutionCompiler().compile(s, ("H1", "H3"), "genotype", reps=2000)
    n80, n90 = r["n_required"]["80"], r["n_required"]["90"]
    assert n80["detect_reduction"] < n90["detect_reduction"] < n90["separation"]
    assert n80["detect_reduction"] == 51           # computed: (1.96+0.84)^2 * 0.105^2 * (1 + 1.25^2) / 0.0664^2
    big = R.ResolutionCompiler().compile(replace(s, n_animals=n80["separation"]), ("H1", "H3"), "genotype", reps=2000)
    assert big["power_separation"] > 0.7


def test_compiler_rejects_non_discriminating_pairs_and_unknown_contrasts():
    s = _spec()
    C = R.ResolutionCompiler()
    with pytest.raises(SpecError, match="non-discriminating"):
        C.compile(s, ("H2", "H3"), "sparse_expression", reps=10)
    with pytest.raises(SpecError, match="unknown contrast"):
        C.compile(s, ("H1", "H3"), "vibes", reps=10)


def test_compile_is_deterministic_under_seed():
    s = _spec(n_animals=6)
    C = R.ResolutionCompiler()
    a = C.compile(s, ("H1", "H3"), "kappa_calibration", reps=200)
    b = C.compile(s, ("H1", "H3"), "kappa_calibration", reps=200)
    assert a["p_correct"] == b["p_correct"] and a["spec_hash"] == b["spec_hash"]
