"""Generation 3 (Deep Mechanism Campaign): signal-provenance and evoked-unmixing families.

Declared after g0-g2 and after the synthetic battery (results/mechanism/deep_campaign/synthetic_battery), BEFORE any
g3 statistic is computed on CONFIRMATION or RESERVE. New families use new data streams (provenance_traces/1.0,
event_crops/1.0) and statistics never computed on any partition; families that are conceptually related to spent ones
are graded PARTIAL on CONFIRMATION/RESERVE (see historical_contamination_g3).

Hypothesis revisions relative to g2 (each with motivating evidence that is not a g0-g2 outcome):
* H2.g3: T6 is no longer a NECESSARY POS prediction. Synthetic fixture S16 (stimulation effectiveness shared by target
  and off-target neighbours) produces a NULL target-flat near field under pure off-target stimulation; the literature
  audit (BIOPHYSICAL_PARAMETERS.json) gives a 2P axial FWHM of 2.6 um with a 15-18 % tail at 3-4 um. The g0-g2
  STRONGLY_DISFAVORED state of H2 rested on that prediction; it is not rewritten there, and H2.g3 must earn or lose
  support on the new families.
* H10.g3 (new): a local shared biological field (e.g. extracellular ionic/diffusive coupling of neighbouring somata)
  that is wiring-independent, target-triggered AND spontaneously shared. Distinct predictions from H3 (no gap-junction
  dependence at fixed distance) and from H6 (fast spontaneous co-fluctuation present).
* H1.g3: the measurement class explicitly includes activity-dependent halo/extended-kernel contamination (the synthetic
  B_HALO fixture), so T13 (static-kernel subtraction) is non-discriminating for H1; T17 (unmixing with a free radial
  halo) carries H1's necessary prediction.
"""
from __future__ import annotations

from ..firewall import HistoricalContamination
from ..measurement import ASSUMED, INFERRED, KNOWN, UNKNOWN, MeasurementField, MeasurementModelSpec
from ..spec import HypothesisSpec, MechanismTestSpec, Prediction
from . import benchmark as B

S, N, ND = B.S, B.N, B.ND
# T14_brightness_elasticity: demoted to descriptive after synthetic battery v1 (false POS in B_CHEM, no power in B_H2/B_H3)
NEW_TESTS = ["T13_optical_copy_subtracted_proximity", "T16_spontaneous_excess", "T17_unmixed_proximity",
             "T18_unmixed_gap_topology", "T19_unmixed_spot_side"]
# T20_axial_lateral_anisotropy: removed from adjudicated tests after battery v2 (POS in every world containing optics, including
# H10/H2 worlds with real receiver-own responses: it measures the presence of axially shaped contamination, not the absence of
# biology); kept as descriptive (box and unmixed).
# T16_spontaneous_excess: removed after calibrated battery v3 (POS under pure contamination with realistic PSF tails: C_H1 3.3,
# C_H2 3.2) -> non-diagnostic for every rival; kept as descriptive.
# T6u_unmixed_target_flat is computed as an O-family audit descriptive (non-discriminating once H2.g3 carries ANY)

# expected call per (family letter, test)
G3_PREDICTIONS = {
    "H1": {"T13_optical_copy_subtracted_proximity": ("ANY", ND, "static kernel misses activity-dependent halo; not diagnostic"),
           "T14_brightness_elasticity": ("NULL", S, "a copy of target photons does not scale with receiver brightness"),
           "T16_spontaneous_excess": ("ANY", ND, "halo contamination also copies spontaneous activity"),
           "T17_unmixed_proximity": ("NULL", N, "compact and radial target light absorbed by unmixing: no receiver-own excess"),
           "T6u_unmixed_target_flat": ("NULL", S, "no target signal to copy"),
           "T18_unmixed_gap_topology": ("NULL", S, "optics ignore wiring"),
           "T19_unmixed_spot_side": ("NULL", S, "copy centred on target signal"), "T20_axial_lateral_anisotropy": ("POS", S, "optical kernel and extraction box are axially elongated")},
    "H2": {"T13_optical_copy_subtracted_proximity": ("POS", S, "receivers' own indicator responds"),
           "T14_brightness_elasticity": ("POS", S, "own-indicator response scales with receiver brightness"),
           "T16_spontaneous_excess": ("NULL", S, "light only during stimulation"),
           "T17_unmixed_proximity": ("POS", N, "direct activation of neighbours survives unmixing"),
           "T6u_unmixed_target_flat": ("ANY", ND, "shared stimulation-effectiveness latent (fixture S16) can null it"),
           "T18_unmixed_gap_topology": ("NULL", N, "light ignores wiring"),
           "T19_unmixed_spot_side": ("POS", S, "footprint follows the spot (small offsets: weak)"), "T20_axial_lateral_anisotropy": ("ANY", ND, "2P footprint roughly isotropic with an axial tail: direction not committed")},
    "H3": {"T13_optical_copy_subtracted_proximity": ("POS", S, ""),
           "T14_brightness_elasticity": ("POS", S, ""),
           "T16_spontaneous_excess": ("POS", S, "coupling propagates spontaneous activity (sub-threshold coupling may null it: S19)"),
           "T17_unmixed_proximity": ("POS", N, "coupled receivers' own responses survive unmixing, if H3 explains the near field"),
           "T6u_unmixed_target_flat": ("NULL", S, "mediated by target activation"),
           "T18_unmixed_gap_topology": ("POS", N, "gap-junction partners respond more at fixed distance"),
           "T19_unmixed_spot_side": ("NULL", S, ""), "T20_axial_lateral_anisotropy": ("NULL", S, "coupling is indifferent to the imaging axis once L/R homologs are excluded")},
    "H4": {"T13_optical_copy_subtracted_proximity": ("ANY", ND, ""), "T14_brightness_elasticity": ("ANY", ND, ""),
           "T16_spontaneous_excess": ("ANY", ND, ""), "T17_unmixed_proximity": ("ANY", ND, "chemical wiring is not intrinsically local"),
           "T6u_unmixed_target_flat": ("NULL", S, ""), "T18_unmixed_gap_topology": ("ANY", ND, "gap edges only"), "T19_unmixed_spot_side": ("NULL", S, ""), "T20_axial_lateral_anisotropy": ("ANY", ND, "")},
    "H5": {"T13_optical_copy_subtracted_proximity": ("ANY", ND, ""), "T14_brightness_elasticity": ("ANY", ND, ""),
           "T16_spontaneous_excess": ("ANY", ND, ""), "T17_unmixed_proximity": ("ANY", ND, ""),
           "T6u_unmixed_target_flat": ("NULL", S, ""), "T18_unmixed_gap_topology": ("ANY", ND, ""), "T19_unmixed_spot_side": ("NULL", S, ""), "T20_axial_lateral_anisotropy": ("ANY", ND, "")},
    "H6": {"T13_optical_copy_subtracted_proximity": ("POS", S, ""), "T14_brightness_elasticity": ("POS", S, ""),
           "T16_spontaneous_excess": ("ANY", ND, "slow release: not fast spontaneous covariance"), "T17_unmixed_proximity": ("POS", N, "diffusion is spatial"),
           "T6u_unmixed_target_flat": ("NULL", S, "requires target release"), "T18_unmixed_gap_topology": ("NULL", S, "wiring independent"),
           "T19_unmixed_spot_side": ("NULL", S, ""), "T20_axial_lateral_anisotropy": ("NULL", S, "diffusion is indifferent to the imaging axis")},
    "H7": {"T13_optical_copy_subtracted_proximity": ("NULL", S, "not target-local"), "T14_brightness_elasticity": ("ANY", ND, ""),
           "T16_spontaneous_excess": ("NULL", S, "no distance-specific covariance"), "T17_unmixed_proximity": ("NULL", N, "not target-local"),
           "T6u_unmixed_target_flat": ("NULL", S, ""), "T18_unmixed_gap_topology": ("NULL", S, ""), "T19_unmixed_spot_side": ("NULL", S, ""), "T20_axial_lateral_anisotropy": ("NULL", S, "")},
    "H9": {"T13_optical_copy_subtracted_proximity": ("POS", S, "light artifact is not a target copy"),
           "T14_brightness_elasticity": ("ANY", ND, "autofluorescence/indicator excitation may scale with brightness"),
           "T16_spontaneous_excess": ("NULL", S, "no light outside stimulation"),
           "T17_unmixed_proximity": ("NULL", S, "a haze around the spot is absorbed by the radial term"),
           "T6u_unmixed_target_flat": ("ANY", ND, ""), "T18_unmixed_gap_topology": ("NULL", S, ""), "T19_unmixed_spot_side": ("POS", S, ""), "T20_axial_lateral_anisotropy": ("ANY", ND, "")},
    "H10": {"T13_optical_copy_subtracted_proximity": ("POS", S, ""), "T14_brightness_elasticity": ("POS", S, ""),
            "T16_spontaneous_excess": ("POS", N, "spontaneously shared local field"), "T17_unmixed_proximity": ("POS", N, ""),
            "T6u_unmixed_target_flat": ("NULL", S, "field triggered by target activity"), "T18_unmixed_gap_topology": ("NULL", N, "wiring independent"),
            "T19_unmixed_spot_side": ("NULL", S, ""), "T20_axial_lateral_anisotropy": ("NULL", S, "local field indifferent to the imaging axis")},
}


def _preds(fam: str) -> list[Prediction]:
    return [Prediction(t, e, k, why) for t, (e, k, why) in G3_PREDICTIONS[fam].items()]


def hypotheses_g3(motivating_evidence_ids: list[str]) -> list[HypothesisSpec]:
    out = []
    for h in B.hypotheses_g2(["g2-parent"]):
        fam = h.family
        preds = [p for p in h._preds() if not (fam == "H2" and p.test_id == "T6_target_flat_nearfield")]
        if fam == "H2":
            preds.append(Prediction("T6_target_flat_nearfield", "ANY", ND, "revised in g3: shared stimulation-effectiveness latent (S16)"))
        preds += _preds(fam)
        falsifiers = list(h.falsifiers)
        if fam == "H2":
            falsifiers = [f for f in falsifiers if "T6" not in f] + ["no receiver-own excess after unmixing (T17 NULL)", "gap-junction topology at fixed distance (T18 POS)"]
        if fam == "H1":
            falsifiers = falsifiers + ["receiver-own excess after unmixing with a free radial halo (T17 POS)"]
        if fam == "H3":
            falsifiers = falsifiers + ["no receiver-own excess after unmixing (T17 NULL)", "no gap-junction dependence at fixed distance (T18 NULL)"]
        out.append(HypothesisSpec(**{**h.__dict__, "hypothesis_id": f"{fam}.g3", "generation": 3, "predictions": preds,
                                     "required_observables": sorted({p.test_id for p in preds}), "falsifiers": falsifiers,
                                     "parent_hypothesis_ids": [f"{fam}.g2"], "created_from_evidence_ids": list(motivating_evidence_ids),
                                     "allowed_data_partitions": ["DISCOVERY", "CONFIRMATION", "RESERVE", "SYNTHETIC", "HISTORICAL"],
                                     "forbidden_data_partitions": []}))
    h3 = next(h for h in out if h.family == "H3")
    h10_preds = [Prediction(p.test_id, p.expected, p.kind, p.rationale) for p in h3._preds() if p.test_id not in G3_PREDICTIONS["H10"]
                 and p.test_id not in ("T1_distance_matched_topology", "T7_path_length")]
    h10_preds += [Prediction("T1_distance_matched_topology", "NULL", S, "wiring independent"), Prediction("T7_path_length", "NULL", S, "")]
    h10_preds += _preds("H10")
    out.append(HypothesisSpec(
        hypothesis_id="H10.g3", generation=3, mechanism_class="LOCAL_SHARED_BIOLOGICAL_FIELD",
        claim=("A local shared biological field couples neighbouring somata independently of wiring: triggered by target activation "
               "and also shared spontaneously (e.g. extracellular ionic / ephaptic / fast diffusive coupling)."),
        causal_entities=["extracellular space around somata"], causal_relations=["target activity -> local field -> neighbours"],
        required_observables=sorted({p.test_id for p in h10_preds}), predicted_signatures=["spontaneous local covariance", "wiring-independent"],
        predicted_absences=["no gap-junction dependence at fixed distance"], falsifiers=["T16 NULL", "T18 POS", "T17 NULL"],
        predictions=h10_preds, known_confounds=["activity-dependent halo contamination mimics spontaneous local covariance"],
        alternative_hypotheses=["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H9", "H8"],
        identifiability_conditions=["T17 separates from H1; T18 from H3; T16 from H6/H2"], required_controls=["unmixing", "distance matching"],
        allowed_data_partitions=["DISCOVERY", "CONFIRMATION", "RESERVE", "SYNTHETIC", "HISTORICAL"], forbidden_data_partitions=[],
        parent_hypothesis_ids=["H3.g2", "H6.g2"], created_from_evidence_ids=list(motivating_evidence_ids)))
    return out


def _t(test_id, question, distinguishes, family, model, null, metric, floor, *, stream, fields=(), cost=1.0, risk=0.2, min_units=5):
    return MechanismTestSpec(
        test_id=test_id, version="1.0", question=question, distinguishes=distinguishes,
        data_requirements=[stream] + list(fields), covariates=[], matching_variables=[], independence_unit="animal", statistic_family=family,
        partition_policy={"scout": "DISCOVERY", "judge": ["CONFIRMATION", "RESERVE"]},
        preprocessing=["camera offset 100 (ASSUMED; differences are offset-free)", "static light model on the mean volume (signal_provenance.fit_light_model)",
                       "event windows: base = samples before onset in [v-8, v); post = 0.25-6.5 s by per-sample acquisition time",
                       "response = robust z (per nucleus across events) > 3"],
        statistical_model=model, hostile_null=null, effect_metric=metric, min_effect=floor,
        uncertainty_estimator="animal-cluster bootstrap 95% percentile interval (400 resamples)",
        stopping_rule="single look per (partition, family, lineage); no re-query", multiple_testing_family="randi_proximity_mechanism_g3",
        min_units=min_units, forbidden_leakage_paths=["confirmation/reserve outcomes -> specs", "floors chosen after outcomes"],
        required_measurement_fields=list(fields), cost=cost, confound_risk=risk)


def tests_g3() -> list[MechanismTestSpec]:
    P, C = "provenance_traces/1.0", "event_crops/1.0"
    return [
        _t("T13_optical_copy_subtracted_proximity", "Does the near-field response excess survive subtracting the optical copy predicted from the static image?",
           ["H2", "H3", "H7", "H10"], "psf_kernel_sufficiency",
           "MH risk difference near (<6 um) vs far (15-40 um) of receiver response after subtracting kappa*dC_target, TARGET-UP events",
           "static-kernel-only; fixture B_HALO shows halo contamination makes it POS", "risk difference", 0.03, stream=P, fields=["raw_voxels"]),
        _t("T17_unmixed_proximity", "After unmixing each evoked volume into nucleus components plus a free radial target halo, do near receivers still respond more?",
           ["H1", "H2", "H3", "H6", "H7", "H10"], "evoked_unmixing_proximity",
           "MH risk difference near (<6 um) vs mid (9-15 um) of unmixed receiver response, TARGET-UP (box z) events",
           "fixtures B_H1, B_HALO, S14 (contamination) vs B_H2, B_H3, B_H10; identical-row box RD reported", "risk difference", 0.03, stream=C,
           fields=["raw_voxels"], cost=3.0, risk=0.25),
        _t("T18_unmixed_gap_topology", "At fixed distance, do gap-junction partners show more unmixed responses?", ["H2", "H3", "H10"],
           "topology_given_distance_unmixed", "MH risk difference gap-edge vs none, strata animal x 2-um bin (<= 15 um), labeled pairs, TARGET-UP",
           "animal-cluster bootstrap", "risk difference", 0.03, stream=C, fields=["label_transfer", "connectome", "raw_voxels"], cost=3.0, risk=0.35),
        _t("T19_unmixed_spot_side", "Do near receivers on the spot-offset side show more unmixed responses?", ["H1", "H2", "H3", "H9", "H10"],
           "spot_offset_directionality_unmixed", "MH risk difference cos>0.5 vs cos<-0.5, offsets >= 1 px, strata animal x 2-um bin (< 8 um)",
           "animal-cluster bootstrap; offsets are small (weak test)", "risk difference", 0.03, stream=C, fields=["stimulation_coordinates", "raw_voxels"], cost=3.0, risk=0.45),
    ]


def measurement_model_g3() -> MeasurementModelSpec:
    m = B.measurement_model_g1()
    m.model_id = "randi2023_pumpprobe_v3"
    f = MeasurementField
    m.fields["optical_psf"] = f("anisotropic Gaussian (nucleus x PSF), shared sigma per animal", INFERRED,
                                "signal_provenance.fit_light_model on the mean volume (isolated nuclei grid search)",
                                uncertainty="sigma scale sensitivity 0.8-1.5x; activity-dependent halo handled by unmixing")
    m.fields["activity_halo"] = f("free radial basis 2-17 um around the target", INFERRED, "unmixing.HALO_KNOTS_UM", note="anisotropic halo not modelled")
    m.fields["per_volume_roi_coordinates"] = f("Leifer tracked coordinates (OSF exported_data_full brains.json)", INFERRED,
                                               "randi.leifer_osf (acquisition-order z converted; timestamp alignment)",
                                               note="114 reference neurons per animal; used for registration audit, not for extraction")
    m.fields["unc31_positions"] = f("OSF exported_data_full brains.json (unc-31 recordings)", KNOWN, "OSF osf.io/e2syt", note="deposited; not used in g3")
    m.fields["stimulation_axial_extent"] = f("2P FWHM 2.6 um axial (beads), 15-18% tail at 3-4 um", INFERRED,
                                             "Randi 2023 Source Data via docs/mechanism_engine/deep_campaign/BIOPHYSICAL_PARAMETERS.json", note="beads, not tissue")
    return m


def historical_contamination_g3(g0_outcomes, g1_outcomes, g2_outcomes) -> list[HistoricalContamination]:
    hc = B.historical_contamination_g2(g0_outcomes, g1_outcomes)
    for fam in sorted({o["feature_family"] for o in g2_outcomes if o["status"] != "NOT_RUN_ACCESS_DENIED" and o["partition"] in ("CONFIRMATION", "RESERVE")}):
        for part in ("CONFIRMATION", "RESERVE"):
            hc.append(HistoricalContamination(part, fam, "CONTAMINATED", "g2 JUDGE read"))
    related = {
        "spontaneous_evoked_slope_identity": "related to baseline_covariance (judged on CONFIRMATION g0 and RESERVE g1)",
        "target_flat_nearfield_unmixed": "related to target_flat_nearfield (judged g0/g1)",
        "topology_given_distance_unmixed": "related to topology_given_distance (judged g0/g1)",
        "spot_offset_directionality_unmixed": "related to spot_offset_directionality (judged g0/g1)",
    }
    for fam, why in related.items():
        for part in ("CONFIRMATION", "RESERVE"):
            hc.append(HistoricalContamination(part, fam, "PARTIAL", why))
    return hc
