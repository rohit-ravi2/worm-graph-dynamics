"""Experiment proposals generated from unresolved hypotheses.

A manipulation catalogue declares, for each manipulation, the observable it produces and what EACH mechanism
class predicts. Identical predicted outcomes MUST use identical text (the separation count compares strings); catalogue v1.1
(2026-09-16) canonicalised entries after the g2 proposal over-counted H1-vs-H3 separation by spot displacement. Proposals are selected by how many unresolved hypothesis pairs a manipulation separates
(per unit cost); nothing here names a preferred mechanism. Catalogue predictions are mechanism-class
consequences, stated with their assumptions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .spec import HypothesisSpec

CLASSES = ["STATIC_MEASUREMENT_CONTAMINATION", "DIRECT_OFF_TARGET_PHOTOSTIMULATION", "GAP_JUNCTION_OR_LOCAL_ELECTRICAL_COUPLING",
           "DIRECT_CHEMICAL_SYNAPTIC_PROPAGATION", "POLYSYNAPTIC_NETWORK_PROPAGATION", "EXTRASYNAPTIC_OR_NEUROPEPTIDERGIC_SIGNALING",
           "GLOBAL_OR_SHARED_STATE"]


@dataclass
class Manipulation:
    name: str
    manipulation: str
    measurement: str
    predictions: dict            # mechanism class (or hypothesis id) -> predicted outcome string
    cost: float
    sample_structure: str
    controls: list
    resolution: str
    analysis: str
    decision_criterion: str
    assumptions: list


CATALOGUE = [
    Manipulation(
        "spot_displacement_series",
        "Stimulate a fixed target with the spot displaced laterally (0, 2, 4, 6 um) and axially (0, 3, 6 um) toward a matched neighbour; "
        "record target and neighbour at each displacement.",
        "neighbour response probability vs spot-neighbour distance, holding target activation measured",
        {"DIRECT_OFF_TARGET_PHOTOSTIMULATION": "neighbour response rises as spot approaches neighbour even when target response falls",
         "H9": "artifact tracks spot position in BOTH green and red, including displacement into empty space",
         "STATIC_MEASUREMENT_CONTAMINATION": "neighbour response tracks target activation, not spot position",
         "GAP_JUNCTION_OR_LOCAL_ELECTRICAL_COUPLING": "neighbour response tracks target activation, not spot position",
         "DIRECT_CHEMICAL_SYNAPTIC_PROPAGATION": "neighbour response tracks target activation, not spot position",
         "POLYSYNAPTIC_NETWORK_PROPAGATION": "neighbour response tracks target activation, not spot position",
         "EXTRASYNAPTIC_OR_NEUROPEPTIDERGIC_SIGNALING": "neighbour response tracks target activation, not spot position",
         "GLOBAL_OR_SHARED_STATE": "no spot-position dependence"},
        cost=2.0, sample_structure=">= 8 animals x >= 10 target/neighbour pairs x 5 displacements x 5 repeats",
        controls=["displacement into empty space (no nucleus)", "sham epochs", "red channel"],
        resolution="displacement step <= footprint FWHM/2 (~1.5 um lateral)",
        analysis="mixed model: neighbour response ~ spot-neighbour distance + target response amplitude, random effects animal/pair; preregister contrast of the two slopes",
        decision_criterion="spot-distance slope CI excludes 0 at fixed target response -> direct photostimulation component; target-amplitude slope only -> mediated/contamination",
        assumptions=["spot displacement is calibrated in 3D", "target activation measured independently"]),
    Manipulation(
        "effector_negative_target",
        "Stimulate neurons that lack the optogenetic effector (sparse-expression strain or effector-negative identified neuron) while neighbours express it.",
        "near-field response when the targeted cell cannot be activated",
        {"DIRECT_OFF_TARGET_PHOTOSTIMULATION": "near-field responses persist",
         "STATIC_MEASUREMENT_CONTAMINATION": "near-field disappears",
         "GAP_JUNCTION_OR_LOCAL_ELECTRICAL_COUPLING": "near-field disappears",
         "DIRECT_CHEMICAL_SYNAPTIC_PROPAGATION": "near-field disappears",
         "POLYSYNAPTIC_NETWORK_PROPAGATION": "near-field disappears",
         "EXTRASYNAPTIC_OR_NEUROPEPTIDERGIC_SIGNALING": "near-field disappears",
         "GLOBAL_OR_SHARED_STATE": "unchanged broad responses"},
        cost=3.0, sample_structure=">= 6 animals x >= 20 stimulations",
        controls=["effector-positive targets in the same animals", "sham epochs"], resolution="volume rate as in Randi 2023",
        analysis="near-far risk difference, effector-negative vs positive targets, animal-cluster bootstrap",
        decision_criterion="near-field risk difference CI > floor for effector-negative targets -> direct photostimulation",
        assumptions=["effector expression verified per cell"]),
    Manipulation(
        "fast_two_plane_imaging",
        "Image only the target plane and one neighbour plane at >= 50 Hz during stimulation.",
        "receiver-minus-target onset delay at ~20 ms resolution",
        {"DIRECT_OFF_TARGET_PHOTOSTIMULATION": "delay ~ 0 (within 20 ms), independent waveform",
         "STATIC_MEASUREMENT_CONTAMINATION": "delay exactly 0 and identical waveform (including target noise)",
         "H9": "blip coincident with the light only",
         "GAP_JUNCTION_OR_LOCAL_ELECTRICAL_COUPLING": "delay 5-100 ms",
         "DIRECT_CHEMICAL_SYNAPTIC_PROPAGATION": "delay >= 200 ms",
         "POLYSYNAPTIC_NETWORK_PROPAGATION": "delay >= 500 ms",
         "EXTRASYNAPTIC_OR_NEUROPEPTIDERGIC_SIGNALING": "not assumed",
         "GLOBAL_OR_SHARED_STATE": "no target-locked delay"},
        cost=2.5, sample_structure=">= 8 animals x >= 30 responding target/neighbour pairs",
        controls=["plane-order reversal", "sham epochs"], resolution="<= 20 ms per sample",
        analysis="onset intervals with per-sample timestamps (engine T3 at high rate)",
        decision_criterion="delay interval lower bound > 20 ms in most pairs -> propagation; within +/-20 ms -> simultaneous",
        assumptions=["GCaMP rise detectable at 50 Hz"]),
    Manipulation(
        "innexin_mutant",
        "Repeat the stimulation atlas protocol in unc-7/unc-9 innexin loss-of-function animals with per-animal coordinates deposited.",
        "near-field and wired-neighbour response rates vs WT at matched distance",
        {"GAP_JUNCTION_OR_LOCAL_ELECTRICAL_COUPLING": "gap-coupled near-field responses reduced",
         "DIRECT_OFF_TARGET_PHOTOSTIMULATION": "unchanged",
         "STATIC_MEASUREMENT_CONTAMINATION": "unchanged",
         "DIRECT_CHEMICAL_SYNAPTIC_PROPAGATION": "unchanged (unless indirect)",
         "POLYSYNAPTIC_NETWORK_PROPAGATION": "partially reduced",
         "EXTRASYNAPTIC_OR_NEUROPEPTIDERGIC_SIGNALING": "unchanged",
         "GLOBAL_OR_SHARED_STATE": "unchanged"},
        cost=4.0, sample_structure=">= 20 animals per genotype", controls=["WT recorded interleaved", "matched optics"],
        resolution="as Randi 2023", analysis="genotype x distance x topology risk-difference contrast with animal-cluster CIs",
        decision_criterion="genotype x gap-edge interaction CI excludes 0", assumptions=["optics matched across genotypes"]),
    Manipulation(
        "unc31_with_geometry",
        "Repeat the stimulation protocol in unc-31 animals WITH per-animal coordinates deposited, analysed under a WT-frozen prediction.",
        "near-field response rate at matched distance and target amplitude, unc-31 vs WT",
        {"EXTRASYNAPTIC_OR_NEUROPEPTIDERGIC_SIGNALING": "near-field reduced",
         "DIRECT_OFF_TARGET_PHOTOSTIMULATION": "unchanged",
         "STATIC_MEASUREMENT_CONTAMINATION": "unchanged",
         "GAP_JUNCTION_OR_LOCAL_ELECTRICAL_COUPLING": "unchanged",
         "DIRECT_CHEMICAL_SYNAPTIC_PROPAGATION": "unchanged or reduced (DCV modulation)",
         "POLYSYNAPTIC_NETWORK_PROPAGATION": "reduced far-field",
         "GLOBAL_OR_SHARED_STATE": "reduced breadth"},
        cost=4.0, sample_structure=">= 40 unc-31 animals (existing arm: 18 recordings, no coordinates)", controls=["WT interleaved", "matched stimulus duration"],
        resolution="as Randi 2023", analysis="frozen WT model predictions scored on untouched unc-31 data",
        decision_criterion="near-field genotype contrast CI excludes 0 in predicted direction", assumptions=["stimulus duration adjusted as in Randi 2023"]),
]


def _pred_for(m: Manipulation, h: HypothesisSpec) -> str | None:
    return m.predictions.get(h.hypothesis_id) or m.predictions.get(h.family) or m.predictions.get(h.mechanism_class) or None


def propose(unresolved: list[HypothesisSpec], *, catalogue=CATALOGUE, top_k: int = 2) -> dict:
    if len(unresolved) < 2:
        return dict(status="NO_PROPOSAL_NEEDED", proposals=[])
    scored = []
    for m in catalogue:
        preds = {h.hypothesis_id: _pred_for(m, h) for h in unresolved}
        pairs = [(a, b) for i, a in enumerate(sorted(preds)) for b in sorted(preds)[i + 1:]
                 if preds[a] and preds[b] and preds[a] != preds[b]]
        # hypotheses of the same class with identical text are not separated
        scored.append((len(pairs) / m.cost, m, preds, pairs))
    scored.sort(key=lambda x: -x[0])
    out = []
    for s, m, preds, pairs in scored[:top_k]:
        if not pairs:
            continue
        out.append(dict(asdict(m), unresolved_hypotheses=[h.hypothesis_id for h in unresolved], predicted_outcome_by_hypothesis=preds,
                        separated_pairs=pairs, estimated_discrimination_value=round(s, 3)))
    return dict(status="PROPOSED" if out else "NO_MANIPULATION_SEPARATES", proposals=out)
