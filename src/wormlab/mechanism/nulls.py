"""Counterfactual (synthetic-null) comparison: can a single mechanism class reproduce the observed signature?

Each generator config below is fixed a priori (documented in docs/mechanism_engine/COUNTERFACTUAL_MODELS.md); the
only data-derived nuisance setting is the target response probability, taken from the discovery measurement and
marked INFERRED. Synthetic cohorts go through the identical observation builder and contrast library. For each
generator we report its per-test calls/estimates and agreement with the observed DISCOVERY calls. This is a
hostile-null map, not a fit: no parameter is tuned to maximise agreement.
"""
from __future__ import annotations

import numpy as np

from . import contrasts, generators as G, observations as O
from .spec import Call, OutcomeStatus, Stage

GENERATORS = {
    "optical_only": dict(p_opt=0.8, sigma_opt_um=3.0, axial_factor=2.0),
    "extraction_overlap_only": dict(contam=0.6, contam_sigma_um=3.0),
    "light_artifact_only": dict(stim_artifact=0.5),
    "gap_only": dict(p_gap=0.6),
    "chem_only": dict(p_chem=0.6),
    "polysynaptic_only": dict(p_poly=0.5),
    "peptidergic_only": dict(p_pep=0.8, pep_radius_um=10.0),
    "global_state_only": dict(p_global=0.3, p_global_node=0.3),
    "mixture_optical_plus_chem": dict(p_opt=0.6, sigma_opt_um=2.5, p_chem=0.7),
    "mixture_contamination_plus_optical": dict(contam=0.4, contam_sigma_um=3.0, p_opt=0.6, sigma_opt_um=2.5),
}


def signature(tab: dict, tests, *, seed: int) -> dict:
    out = {}
    for t in tests:
        fn = contrasts.REGISTRY.get(t.test_id)
        if fn is None:
            continue
        raw = fn(tab, seed=seed)
        o = contrasts.to_outcome(t, raw, stage=Stage.SYNTHETIC, partition="SYNTHETIC", units=sorted(set(tab["animal"].tolist())) if len(tab.get("animal", [])) else [],
                                 grade="SYNTHETIC")
        out[t.test_id] = dict(call=o.call, status=o.status, estimate=o.estimate, ci=o.ci)
    return out


def compare(observed: list, tests, *, n_animals: int, p_target: float, seed: int = 7, n_events: int = 60) -> dict:
    obs = {o.test_id: o for o in observed if o.status == OutcomeStatus.DECIDED}
    report = {}
    for name, params in GENERATORS.items():
        coh = G.generate_cohort(seed, n_animals, p_target=p_target, n_events=n_events, **params)
        tab = O.concat([O.build_table(p, f"{name}-{a}", conn) for a, p, conn in coh])
        sig = signature(tab, tests, seed=seed)
        comparable = [t for t in obs if t in sig and sig[t]["status"] == OutcomeStatus.DECIDED]
        agree = [t for t in comparable if sig[t]["call"] == obs[t].call]
        report[name] = dict(params=params, signature=sig, n_comparable=len(comparable), n_agree=len(agree),
                            agreement=round(len(agree) / len(comparable), 3) if comparable else None,
                            disagree=sorted(set(comparable) - set(agree)),
                            proximity_effect=sig.get("T0_proximity_effect", {}).get("estimate"))
    ranked = sorted(report, key=lambda k: -(report[k]["agreement"] or -1))
    return dict(p_target_inferred_from_discovery=p_target, n_animals=n_animals, observed={t: dict(call=o.call, estimate=o.estimate) for t, o in obs.items()},
                generators=report, ranking=ranked,
                note="Agreement counts DECIDED observed tests where the generator's call matches. Generators are a priori; none fitted.")
