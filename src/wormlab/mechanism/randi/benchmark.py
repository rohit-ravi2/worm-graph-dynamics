"""Randi proximity benchmark definitions: hypotheses, tests, measurement model, contamination ledger.

Predictions encode what each mechanism class implies for each contrast; they do not encode which mechanism
is true. Floors (min_effect) are fixed a priori from measurement considerations, not tuned on outcomes:
rate contrasts use absolute risk-difference floors of 0.02-0.05 (a few-percent change in response probability
is the smallest effect that would matter for a functional map with ~2% background responding); latency uses
0.25 s (half a volume period).
"""
from __future__ import annotations

from ..firewall import HistoricalContamination
from ..measurement import ASSUMED, INFERRED, KNOWN, UNKNOWN, MeasurementField, MeasurementModelSpec
from ..spec import HypothesisSpec, MechanismTestSpec, Prediction

S, N, F, ND = "SUPPORTIVE_PREDICTION", "NECESSARY_PREDICTION", "FALSIFYING_OBSERVATION", "NON_DISCRIMINATING_OBSERVATION"
ALLOWED = ["DISCOVERY", "CONFIRMATION", "SYNTHETIC", "HISTORICAL"]
FORBIDDEN = ["RESERVE"]


def P(test, exp, kind, why="", delay=None):
    return Prediction(test_id=test, expected=exp, kind=kind, rationale=why, delay_range_s=delay)


def _h(hid, cls, claim, preds, *, entities, relations, signatures, absences, falsifiers, confounds, identifiability, controls, alts):
    return HypothesisSpec(
        hypothesis_id=hid, generation=0, claim=claim, mechanism_class=cls, causal_entities=entities, causal_relations=relations,
        required_observables=sorted({p.test_id for p in preds}), predicted_signatures=signatures, predicted_absences=absences,
        falsifiers=falsifiers, predictions=preds, known_confounds=confounds, alternative_hypotheses=alts,
        identifiability_conditions=identifiability, required_controls=controls,
        allowed_data_partitions=ALLOWED, forbidden_data_partitions=FORBIDDEN)


ALL = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H9"]


def hypotheses() -> list[HypothesisSpec]:
    alts = lambda me: [h for h in ALL if h != me] + ["H8"]
    return [
        _h("H1.g0", "STATIC_MEASUREMENT_CONTAMINATION",
           "Near-field 'responses' are measurement contamination: receivers' extracted signal contains a distance-weighted copy of the "
           "target's signal (point spread, extraction overlap, shared measurement noise); no neural activation of receivers.",
           [P("T0_proximity_effect", "POS", N, "contamination decays with distance"),
            P("T1_distance_matched_topology", "NULL", N, "wiring irrelevant at fixed distance"),
            P("T2_topology_matched_distance", "POS", S),
            P("T3_onset_latency", "NULL", N, "copy of target signal: zero delay", [-0.05, 0.05]),
            P("T4_offset_directionality", "NULL", S, "centred on target signal, not on the light spot"),
            P("T5_control_channel", "ANY", ND, "PSF bleed does not reach red; shared noise could"),
            P("T6_target_flat_nearfield", "NULL", N, "no target signal -> nothing to copy"),
            P("T7_path_length", "NULL", S),
            P("T8_sham_nearfield", "POS", S, "spontaneous pseudo-target activity is copied too"),
            P("T9_animal_replication", "POS", S),
            P("T10_farfield_breadth", "NULL", S),
            P("T11_baseline_covariance", "POS", N, "copies target measured fluctuations"),
            P("X1_nucleus_vs_background", "NULL", N, "diffuse contamination does not concentrate at receiver nucleus centres")],
           entities=["target nucleus signal", "receiver extraction box", "optical point-spread"], relations=["target signal -> receiver measurement"],
           signatures=["zero-lag copy", "requires target signal", "wiring-independent"], absences=["no activation without target signal"],
           falsifiers=["near-field activation with flat target (T6 POS)", "receiver delay beyond resolution (T3 POS)", "discrete nuclear responses over flat background (X1 POS)"],
           confounds=["off-target photostimulation mimics distance dependence"], identifiability=["T6 or X1 separates from activation hypotheses"],
           controls=["sham windows", "red channel"], alts=alts("H1")),
        _h("H2.g0", "DIRECT_OFF_TARGET_PHOTOSTIMULATION",
           "The two-photon stimulus directly activates nuclei near the spot (GUR-3/PRDX-2 is pan-neuronal), independently of target "
           "activation and of wiring.",
           [P("T0_proximity_effect", "POS", N),
            P("T1_distance_matched_topology", "NULL", N, "light does not know wiring"),
            P("T2_topology_matched_distance", "POS", N),
            P("T3_onset_latency", "NULL", N, "simultaneous activation", [-0.05, 0.1]),
            P("T4_offset_directionality", "POS", S, "footprint follows the spot (offsets are small: weak test)"),
            P("T5_control_channel", "NULL", S, "neural activation does not change tagRFP"),
            P("T6_target_flat_nearfield", "POS", N, "light hits neighbours even when target fails to respond"),
            P("T7_path_length", "NULL", S),
            P("T8_sham_nearfield", "NULL", N, "no light in sham windows"),
            P("T9_animal_replication", "POS", S),
            P("T10_farfield_breadth", "NULL", S),
            P("T11_baseline_covariance", "NULL", S),
            P("X1_nucleus_vs_background", "POS", S)],
           entities=["2P stimulation footprint", "GUR-3/PRDX-2 in neighbours"], relations=["light -> neighbour activation"],
           signatures=["activation without target mediation", "no sham near-field"], absences=["no wiring dependence at fixed distance"],
           falsifiers=["sham near-field conditional on pseudo-target activity (T8 POS)", "no near-field when target flat (T6 NULL)",
                       "topology effect at fixed distance (T1 POS)"],
           confounds=["stimulus efficacy correlates target and neighbour activation"], identifiability=["T6/T8 separate from mediated mechanisms; timing cannot separate from gap junctions"],
           controls=["sham windows"], alts=alts("H2")),
        _h("H3.g0", "GAP_JUNCTION_OR_LOCAL_ELECTRICAL_COUPLING",
           "Target activation propagates to electrically coupled receivers with short latency.",
           [P("T0_proximity_effect", "ANY", ND, "depends on spatial layout of gap junctions"),
            P("T1_distance_matched_topology", "POS", N, "coupled pairs respond more at fixed distance"),
            P("T2_topology_matched_distance", "NULL", S),
            P("T3_onset_latency", "NULL", S, "tens of ms: below volume resolution", [0.005, 0.1]),
            P("T4_offset_directionality", "NULL", S),
            P("T5_control_channel", "NULL", S),
            P("T6_target_flat_nearfield", "NULL", N, "mediated: needs target activation"),
            P("T7_path_length", "POS", S),
            P("T8_sham_nearfield", "POS", S, "spontaneous target activity also propagates"),
            P("T9_animal_replication", "ANY", ND),
            P("T10_farfield_breadth", "NULL", S),
            P("T11_baseline_covariance", "POS", S, "electrical coupling correlates spontaneous activity"),
            P("X1_nucleus_vs_background", "POS", S)],
           entities=["innexin gap junctions"], relations=["target depolarisation -> coupled receiver"],
           signatures=["topology at fixed distance", "requires target activation"], absences=["no distance effect at fixed topology"],
           falsifiers=["no topology effect at fixed distance with adequate coverage (T1 NULL)", "near-field activation with flat target (T6 POS)"],
           confounds=["connectome spatially embedded"], identifiability=["latency cannot separate from H2 at 2 Hz volumes"],
           controls=["distance matching"], alts=alts("H3")),
        _h("H4.g0", "DIRECT_CHEMICAL_SYNAPTIC_PROPAGATION",
           "Target activation drives direct chemical postsynaptic receivers.",
           [P("T0_proximity_effect", "ANY", ND),
            P("T1_distance_matched_topology", "POS", N),
            P("T2_topology_matched_distance", "NULL", S),
            P("T3_onset_latency", "POS", N, "synaptic + calcium latency", [0.2, 3.0]),
            P("T4_offset_directionality", "NULL", S),
            P("T5_control_channel", "NULL", S),
            P("T6_target_flat_nearfield", "NULL", N),
            P("T7_path_length", "POS", S),
            P("T8_sham_nearfield", "POS", S),
            P("T9_animal_replication", "ANY", ND),
            P("T10_farfield_breadth", "NULL", S),
            P("T11_baseline_covariance", "ANY", ND),
            P("X1_nucleus_vs_background", "POS", S)],
           entities=["chemical synapses"], relations=["target -> postsynaptic receiver"],
           signatures=["topology at fixed distance", "receiver lags target"], absences=["no activation without target"],
           falsifiers=["receivers simultaneous with target within resolution (T3 NULL)", "T1 NULL with adequate coverage", "T6 POS"],
           confounds=["connectome spatially embedded", "shared input"], identifiability=["T1 and T3 jointly"],
           controls=["distance matching", "scan-time correction"], alts=alts("H4")),
        _h("H5.g0", "POLYSYNAPTIC_NETWORK_PROPAGATION",
           "Target activation propagates through multi-hop wired paths.",
           [P("T0_proximity_effect", "ANY", ND),
            P("T1_distance_matched_topology", "POS", S),
            P("T2_topology_matched_distance", "NULL", S),
            P("T3_onset_latency", "POS", N, "multi-hop latency", [0.5, 5.0]),
            P("T4_offset_directionality", "NULL", S),
            P("T5_control_channel", "NULL", S),
            P("T6_target_flat_nearfield", "NULL", N),
            P("T7_path_length", "POS", N, "short paths respond more at fixed distance"),
            P("T8_sham_nearfield", "POS", S),
            P("T9_animal_replication", "ANY", ND),
            P("T10_farfield_breadth", "POS", S, "propagation reaches beyond the near field"),
            P("T11_baseline_covariance", "ANY", ND),
            P("X1_nucleus_vs_background", "POS", S)],
           entities=["wired multi-hop paths"], relations=["target -> intermediate -> receiver"],
           signatures=["path length at fixed distance", "latency"], absences=["no activation without target"],
           falsifiers=["T7 NULL with adequate coverage", "T3 NULL", "T6 POS"],
           confounds=["connectome spatially embedded"], identifiability=["T7 and T3"], controls=["distance matching"], alts=alts("H5")),
        _h("H6.g0", "EXTRASYNAPTIC_OR_NEUROPEPTIDERGIC_SIGNALING",
           "Target activation releases diffusible signals acting on nearby receivers independent of wired paths.",
           [P("T0_proximity_effect", "POS", S, "diffusion is spatial"),
            P("T1_distance_matched_topology", "NULL", S, "wiring-independent"),
            P("T2_topology_matched_distance", "POS", S),
            P("T3_onset_latency", "ANY", ND, "latency not assumed"),
            P("T4_offset_directionality", "NULL", S),
            P("T5_control_channel", "NULL", S),
            P("T6_target_flat_nearfield", "NULL", N, "requires target release"),
            P("T7_path_length", "NULL", S),
            P("T8_sham_nearfield", "POS", S),
            P("T9_animal_replication", "ANY", ND),
            P("T10_farfield_breadth", "ANY", ND),
            P("T11_baseline_covariance", "NULL", S),
            P("X1_nucleus_vs_background", "POS", S)],
           entities=["dense-core-vesicle release", "neuropeptide receptors"], relations=["target release -> nearby receptor-expressing receivers"],
           signatures=["spatial, wiring-independent, target-dependent"], absences=["no activation without target"],
           falsifiers=["T6 POS", "T8 NULL with adequate power"], confounds=["indistinguishable from H2 without target-dependence tests"],
           identifiability=["unc-31 intervention (prospective) or T6/T8"], controls=["sham windows"], alts=alts("H6")),
        _h("H7.g0", "GLOBAL_OR_SHARED_STATE",
           "Responses reflect broad state changes coincident with stimulation, not target-local geometry.",
           [P("T0_proximity_effect", "NULL", N, "not target-local"),
            P("T1_distance_matched_topology", "NULL", S),
            P("T2_topology_matched_distance", "NULL", S),
            P("T3_onset_latency", "ANY", ND),
            P("T4_offset_directionality", "NULL", S),
            P("T5_control_channel", "ANY", ND),
            P("T6_target_flat_nearfield", "ANY", ND),
            P("T7_path_length", "NULL", S),
            P("T8_sham_nearfield", "NULL", S),
            P("T9_animal_replication", "NULL", S, "no consistent target-local near-far difference"),
            P("T10_farfield_breadth", "POS", N, "stimulation raises responses far from the target"),
            P("T11_baseline_covariance", "ANY", ND),
            P("X1_nucleus_vs_background", "ANY", ND)],
           entities=["global arousal / state"], relations=["stimulus epoch -> brain-wide activity"],
           signatures=["far-field responses on stim vs sham"], absences=["no target-local spatial kernel"],
           falsifiers=["strong proximity effect (T0 POS)", "no far-field stim-vs-sham difference (T10 NULL)"],
           confounds=["spontaneous state transitions"], identifiability=["T0 and T10"], controls=["sham windows"], alts=alts("H7")),
        _h("H9.g0", "STATIC_MEASUREMENT_CONTAMINATION",
           "Near-field 'responses' are a stimulation-light artifact: the 2P stimulus transiently excites fluorescence (green and red) "
           "near the spot during the stimulation epoch, independent of neural activity. (Generated from the measurement model: "
           "a light artifact is a measurement contamination distinct from point-spread copying of the target signal.)",
           [P("T0_proximity_effect", "POS", N),
            P("T1_distance_matched_topology", "NULL", N),
            P("T2_topology_matched_distance", "POS", S),
            P("T3_onset_latency", "NULL", S, "coincident with the light", [-0.05, 0.05]),
            P("T4_offset_directionality", "POS", S),
            P("T5_control_channel", "POS", N, "light excites tagRFP too"),
            P("T6_target_flat_nearfield", "POS", S, "artifact does not need a target response"),
            P("T7_path_length", "NULL", S),
            P("T8_sham_nearfield", "NULL", N, "no light in sham"),
            P("T9_animal_replication", "POS", S),
            P("T10_farfield_breadth", "NULL", S),
            P("T11_baseline_covariance", "NULL", S, "artifact only during stimulation"),
            P("X1_nucleus_vs_background", "NULL", S, "artifact is not nucleus-specific")],
           entities=["stimulation light", "fluorophores near spot"], relations=["light -> fluorescence artifact"],
           signatures=["red-channel near-field", "absent in sham"], absences=["no wiring dependence"],
           falsifiers=["no red near-field (T5 NULL)", "discrete nuclear responses (X1 POS)"],
           confounds=["off-target photostimulation also tracks the spot"], identifiability=["T5 separates from H2"],
           controls=["red channel", "sham windows"], alts=alts("H9")),
    ]


def _t(test_id, question, distinguishes, family, model, null, metric, floor, *, covariates=(), matching=(), fields=(), cost=1.0, risk=0.2, min_units=5):
    return MechanismTestSpec(
        test_id=test_id, version="1.0", question=question, distinguishes=distinguishes,
        data_requirements=["raw_traces/1.0 event traces"] + list(fields), covariates=list(covariates), matching_variables=list(matching),
        independence_unit="animal", statistic_family=family, partition_policy={"scout": "DISCOVERY", "judge": "CONFIRMATION"},
        preprocessing=["camera offset subtraction (ASSUMED 100)", "dF/F 0.5-6.5 s vs 4 s pre", "robust z per nucleus across events", "response = z > 3"],
        statistical_model=model, hostile_null=null, effect_metric=metric, min_effect=floor,
        uncertainty_estimator="animal-cluster bootstrap 95% percentile interval (400 resamples)",
        stopping_rule="single look per (partition, family, lineage); no re-query", multiple_testing_family="randi_proximity_mechanism_g0",
        min_units=min_units, forbidden_leakage_paths=["confirmation outcomes -> hypothesis/test specs", "floors chosen after outcomes"],
        required_measurement_fields=list(fields), cost=cost, confound_risk=risk)


def tests() -> list[MechanismTestSpec]:
    return [
        _t("T0_proximity_effect", "Do receivers near the target respond more than far receivers (target responding)?", ["H1", "H2", "H7"],
           "proximity_rate", "Mantel-Haenszel risk difference near (<6 um) vs far (15-40 um), strata = animal",
           "animal-cluster bootstrap; sham analogue in T8", "risk difference", 0.03, fields=["xy_um_per_px", "z_um_per_plane"]),
        _t("T1_distance_matched_topology", "At fixed physical distance, do wired (chem or gap) pairs respond more?", ["H1", "H2", "H3", "H4", "H6"],
           "topology_given_distance", "MH risk difference edge vs no-edge, strata = animal x 2-um distance bin (<=15 um), labeled pairs",
           "topology labels permuted within animal x distance strata (200 permutations)", "risk difference", 0.02,
           matching=["animal", "distance bin"], fields=["label_transfer", "connectome"], risk=0.3),
        _t("T2_topology_matched_distance", "At fixed topology class, does distance still predict response?", ["H1", "H2", "H3", "H4"],
           "distance_given_topology", "MH risk difference near (<6 um) vs 10-15 um, strata = animal x topology class", "animal-cluster bootstrap",
           "risk difference", 0.03, matching=["animal", "topology class"], fields=["label_transfer", "connectome"]),
        _t("T3_onset_latency", "After scan-time correction, do near responders lag the target?", ["H1", "H2", "H4", "H5"],
           "onset_latency_scan_corrected", "median midpoint of receiver-minus-target onset intervals (plane-resolved sample times)",
           "delay intervals straddling zero counted UNRESOLVED; precedence only when interval excludes 0", "seconds", 0.25,
           fields=["plane_acquisition_times", "stimulus_onset"], risk=0.3),
        _t("T4_offset_directionality", "Do near receivers on the side the spot was offset toward respond more?", ["H2", "H3", "H4", "H1"],
           "spot_offset_directionality", "MH risk difference cos>0.5 vs cos<-0.5, strata = animal x 2-um bin, offsets >= 1 px",
           "animal-cluster bootstrap", "risk difference", 0.03, fields=["stimulation_coordinates", "xy_um_per_px"], risk=0.4),
        _t("T5_control_channel", "Does the red (activity-independent) channel show a near-field effect?", ["H9", "H2", "H3", "H4"],
           "red_channel_nearfield", "MH risk difference of |red z|>3 near vs far, target responding", "animal-cluster bootstrap",
           "risk difference", 0.02, fields=["red_registration_transform"], risk=0.4),
        _t("T6_target_flat_nearfield", "When the target does NOT respond, do near receivers still respond more than far?", ["H1", "H2", "H3", "H4", "H6"],
           "target_flat_nearfield", "MH risk difference near vs far among events with |target z| < 1", "animal-cluster bootstrap",
           "risk difference", 0.03, risk=0.3),
        _t("T7_path_length", "At fixed distance, do receivers within 2 wired hops respond more?", ["H1", "H2", "H5", "H6"],
           "pathlength_given_distance", "MH risk difference path<=2 vs longer/none, strata = animal x 2-um bin",
           "path labels permuted within strata", "risk difference", 0.02, matching=["animal", "distance bin"], fields=["label_transfer", "connectome"], risk=0.3),
        _t("T8_sham_nearfield", "In no-stimulation windows with spontaneous pseudo-target activity, is there a near-field?", ["H1", "H2", "H3", "H6"],
           "sham_nearfield", "MH risk difference near vs far, sham events with pseudo-target z > 3", "animal-cluster bootstrap",
           "risk difference", 0.05, fields=["stimulus_onset"], min_units=5, risk=0.3),
        _t("T9_animal_replication", "Is the near-far difference positive in most animals?", ["H1", "H2", "H7"],
           "proximity_replication", "fraction of animals with near-far > 0, minus 0.5", "bootstrap over animals", "fraction - 0.5", 0.1),
        _t("T10_farfield_breadth", "Do far-field (15-40 um) receivers respond more on stim than sham windows?", ["H5", "H7", "H2", "H1"],
           "farfield_breadth", "MH risk difference stim vs sham, strata = animal", "animal-cluster bootstrap", "risk difference", 0.01, risk=0.3),
        _t("T11_baseline_covariance", "Do near receivers co-fluctuate with the target BEFORE stimulation more than far receivers?",
           ["H1", "H2", "H6", "H9"], "baseline_covariance", "mean Fisher-z of pre-stimulus receiver-target trace correlation, near (<6 um) minus far (15-40 um), per animal, weighted",
           "animal-cluster bootstrap", "Fisher-z difference", 0.05, risk=0.3),
        _t("X1_nucleus_vs_background", "Are near-field increases concentrated at receiver nucleus centres over inter-nuclear background?", ["H1", "H2"],
           "haze_vs_discrete", "median normalized dF nucleus-centre minus background, 3.5-5 um shell (imported discovery analysis)",
           "target-flat events", "normalized dF difference", 0.03, fields=["raw_voxels"], cost=5.0, risk=0.3),
    ]


def measurement_model() -> MeasurementModelSpec:
    f = MeasurementField
    return MeasurementModelSpec("randi2023_pumpprobe_v1", "DANDI:001075", {
        "volume_shape": f([25, 512, 512], KNOWN, "raw NWB PumpProbeImagingGreen shape / depth sweep length"),
        "frame_period_s": f(0.02, KNOWN, "PumpProbeImagingGreen/timestamps diffs"),
        "volume_period_s": f(0.5, KNOWN, "BaseGreenSignal timestamps"),
        "plane_acquisition_times": f("per-frame timestamps", KNOWN, "PumpProbeImagingGreen/timestamps"),
        "plane_ordering": f("alternating up/down sweeps", KNOWN, "depth_per_frame_in_um reversals"),
        "z_um_per_plane": f(1.22, KNOWN, "median |diff(depth_per_frame_in_um)|"),
        "xy_um_per_px": f(0.42, ASSUMED, "wormdatamodel recording.frameUmPerPixel code constant; not in NWB metadata"),
        "camera_offset_counts": f(100, ASSUMED, "wormdatamodel extract() docstring (Hamamatsu ORCA ~100 per pixel)"),
        "stimulus_onset": f("stim table start_time", KNOWN, "intervals/OptogeneticStimulusTable"),
        "stimulus_duration_s": f(0.5, KNOWN, "stim table stop - start (WT)"),
        "stimulation_power_W": f(0.0012, KNOWN, "stim table power (constant: power dependence untestable)"),
        "stimulation_coordinates": f("spot pixel (x, y) per stim", KNOWN, "TargetPlaneSegmentation/pixel_mask"),
        "stimulation_depth": f(None, UNKNOWN, note="TargetPlaneSegmentation/depth_in_um values ~ -0.1..0.16 with undocumented semantics"),
        "stimulation_footprint": f("3.1 um FWHM (lateral)", KNOWN, "Randi 2023 Extended Data Fig. 2c,d legend", note="axial extent not reported"),
        "stimulation_axial_extent": f(None, UNKNOWN),
        "green_segmentation": f("LoG nuclei on mean green volume", INFERRED, "raw_traces.py _segment"),
        "green_transform_dandi_to_raw": f({"zflip": True, "dx": 0, "dy": 0}, INFERRED,
                                          "reextract/probe_target_dF.json: target dF 7.34 vs <=0.5 other transforms; validated 21/24 animals"),
        "red_registration_transform": f({"zflip": True, "dy": 16, "dx": 3}, INFERRED, "reextract/probe_global_offset.json (tagRFP brightness 177 vs 107)",
                                        uncertainty="not validated on activity; T5 inherits this uncertainty"),
        "leifer_extraction_box": f([3, 5, 5], KNOWN, "catalystneuro session_to_box_shape.json (per session; 1x3x3 in 111 sessions)"),
        "dandi_voxel_mask": f(None, UNKNOWN, note="deposited voxel masks carry a conversion bug (offsets added 3x)"),
        "per_volume_roi_coordinates": f(None, UNKNOWN, note="Leifer tracks ROIs per volume; not deposited"),
        "label_transfer": f("DANDI centroid (x, y, 24-z) within 2 px / 1 plane", INFERRED, "raw_traces.py label transfer"),
        "connectome": f("wormneuroatlas aconnectome_default (chem[i,j]=j->i; gap)", KNOWN, "wormneuroatlas 0.0.7.3 package data"),
        "optical_psf": f(None, UNKNOWN, note="no PSF measurement deposited"),
        "red_channel": f("raw PumpProbeImagingRed", KNOWN, "raw NWB"),
        "raw_voxels": f("raw volumes", KNOWN, "raw NWB"),
        "animal_ids": f("DANDI subject ids", KNOWN, "DANDI:001075"),
        "unc31_positions": f(None, UNKNOWN, note="unc-31 arm not in DANDI:001075; exports carry labels but no coordinates"),
    })


def historical_contamination() -> list[HistoricalContamination]:
    H = HistoricalContamination
    return [
        H("DISCOVERY", "proximity_rate", "CONTAMINATED", "reextract/nearfield_by_method.json (these 24 animals, raw extraction)"),
        H("ALL_WT", "proximity_rate", "PARTIAL", "run 1 T0/T1b/T6b and pipeline_q within-pair on DANDI Base signals, all WT animals"),
        H("ALL_WT", "proximity_replication", "PARTIAL", "run 1 per-animal near-far effects (Base signals)"),
        H("ALL_WT", "topology_given_distance", "PARTIAL", "L3 gates (funatlas, distance+connectome AUROC), pipeline_q pq07 connectome increment; aggregated over all WT"),
        H("ALL_WT", "pathlength_given_distance", "PARTIAL", "v1.2 relational features (2-hop reachability) on funatlas, all WT aggregated"),
        H("ALL_WT", "distance_given_topology", "PARTIAL", "run 1 T2 identity prior vs measured distance (Base signals)"),
        H("ALL_WT", "onset_latency_scan_corrected", "PARTIAL", "run 1 T4 volume-level latency/xcorr (uncorrected) on Base signals"),
        H("ALL_WT", "target_flat_nearfield", "PARTIAL", "run 1 T4 P(up | target flat) near vs far (Base signals)"),
        H("ALL_WT", "red_channel_nearfield", "PARTIAL", "run 1 T1 red near-field on Base red signals"),
        H("DISCOVERY", "haze_vs_discrete", "CONTAMINATED", "reextract/haze_vs_discrete.json (these 24 animals)"),
    ]


UNC31_LEDGER = {
    "dataset": "Randi 2023 unc-31 arm (funatlas.h5 unc31/*; Leifer exports exported_data_unc31, 18 recordings)",
    "inspected": [
        {"what": "significant-pair counts, LOST/GAINED vs WT, effective n", "where": "docs/discovery/L3_FEASIBILITY_GATES.md (G2)", "grade": "CONTAMINATED"},
        {"what": "near-field response rate by atlas distance, stratified by target amplitude", "where": "analyses/randi_distance_map/out/t5_genotype.json", "grade": "CONTAMINATED"},
        {"what": "loss rate at occ>=2/3/5", "where": "diag2 (l3check)", "grade": "CONTAMINATED"},
        {"what": "release-machinery gate, PR dimensionality WT vs unc-31, WB-2 Spearman, peptide sign flips", "where": "legacy/tiered_simulator/release-machinery, legacy/neuromod/latent-dim, WB2_VERDICT, peptide-atlas", "grade": "CONTAMINATED"},
    ],
    "prospectively_usable": "NONE for proximity/near-field, significance-count, dimensionality or pair-sign families. Untouched only: "
                            "families requiring per-animal unc-31 geometry (not deposited) — therefore not computable.",
    "power_coverage": {"recordings": 18, "usable_in_t5": 15, "positions": "UNKNOWN", "neurons_covered_approx": 88},
    "verdict": "unc-31 prospective Level-3 test NOT EARNED on available data: contaminated for every computable family and missing geometry for the rest.",
}


# =====================================================================================================================
# Generation 1 (declared after g0; see docs/mechanism_engine/RANDI_PROXIMITY_BENCHMARK.md "Generation g1")
# =====================================================================================================================
def measurement_model_g1() -> MeasurementModelSpec:
    m = measurement_model()
    m.model_id = "randi2023_pumpprobe_v2"
    m.fields["red_registration_transform"] = MeasurementField(
        None, UNKNOWN,
        note=("VALIDATION FAILED (g1 scout, DISCOVERY): with the inferred transform the red box sits ~10 counts above offset "
              "(green ~58) with pre-stimulus red dF/F SD 0.21; an exhaustive translation search (z flip x dz +-3 x dy,dx +-40) "
              "finds no mapping that places raw-detected nuclei on tagRFP nuclei (best median ~3 counts above offset). "
              "g0 T5 outcomes were measured at unvalidated, near-background locations."))
    return m


def hypotheses_g1(motivating_evidence_ids: list[str]) -> list[HypothesisSpec]:
    out = []
    for h in hypotheses():
        t11 = h.prediction_for("T11_baseline_covariance")
        preds = list(h._preds()) + [Prediction("T11b_baseline_covariance_plane_matched", t11.expected, t11.kind,
                                               (t11.rationale + " (plane-matched: excludes shared-frame co-sampling)").strip())]
        g = HypothesisSpec(**{**h.__dict__, "hypothesis_id": f"{h.family}.g1", "generation": 1, "predictions": preds,
                              "required_observables": sorted({p.test_id for p in preds}),
                              "parent_hypothesis_ids": [h.hypothesis_id], "created_from_evidence_ids": list(motivating_evidence_ids)})
        out.append(g)
    return out


def tests_g1() -> list[MechanismTestSpec]:
    ts = tests()
    ts.insert(-1, _t("T11b_baseline_covariance_plane_matched",
                     "Plane-matched: do near receivers co-fluctuate with the target before stimulation more than far receivers in the same plane-difference class?",
                     ["H1", "H2", "H6", "H9"], "baseline_covariance_plane_matched",
                     "weighted mean Fisher-z difference near (<6 um) minus far (15-40 um) within animal x |dplane| class {0, 1-2, >=3}",
                     "animal-cluster bootstrap; synthetic shared-frame noise fixture S12 shows naive T11 inflation is removed", "Fisher-z difference", 0.05, risk=0.2))
    return ts


def historical_contamination_g1(g0_outcomes: list[dict]) -> list[HistoricalContamination]:
    hc = historical_contamination()
    seen = sorted({o["feature_family"] for o in g0_outcomes if o["partition"] == "CONFIRMATION" and o["status"] != "NOT_RUN_ACCESS_DENIED"})
    for fam in seen:
        hc.append(HistoricalContamination("CONFIRMATION", fam, "CONTAMINATED", "g0 JUDGE read (lineage parent); see g0 capsule access ledger"))
    return hc


# =====================================================================================================================
# Generation 2 (declared after the g1 run and its policy-1.2 reinterpretation)
# =====================================================================================================================
def hypotheses_g2(motivating_evidence_ids: list[str]) -> list[HypothesisSpec]:
    out = []
    for h in hypotheses_g1(["g1-parent"]):
        x1 = h.prediction_for("X1_nucleus_vs_background")
        preds = list(h._preds()) + [Prediction("T12_nucleus_vs_background_voxel", x1.expected, x1.kind,
                                               (x1.rationale + " (prospective voxel-level protocol, voxel_x1/1.0)").strip())]
        out.append(HypothesisSpec(**{**h.__dict__, "hypothesis_id": f"{h.family}.g2", "generation": 2, "predictions": preds,
                                     "required_observables": sorted({p.test_id for p in preds}),
                                     "parent_hypothesis_ids": [f"{h.family}.g1"], "created_from_evidence_ids": list(motivating_evidence_ids)}))
    return out


def tests_g2() -> list[MechanismTestSpec]:
    return [_t("T12_nucleus_vs_background_voxel",
               "When the target responds, is the 3.5-5 um near-field dF concentrated at receiver nucleus centres relative to inter-nuclear background?",
               ["H1", "H2", "H3", "H4", "H9"], "haze_vs_discrete",
               "per animal mean over TARGET_UP events of (median nucleus-centre dF - median background dF) / target dF, shell 3.5-5 um; pooled weighted by events",
               "TARGET_FLAT events reported as control; animal-cluster bootstrap", "normalised dF difference", 0.03,
               fields=["raw_voxels", "green_transform_dandi_to_raw"], cost=3.0, risk=0.3)]


def historical_contamination_g2(g0_outcomes: list[dict], g1_outcomes: list[dict]) -> list[HistoricalContamination]:
    hc = historical_contamination_g1(g0_outcomes)
    for fam in sorted({o["feature_family"] for o in g1_outcomes if o["partition"] == "RESERVE" and o["status"] != "NOT_RUN_ACCESS_DENIED"}):
        hc.append(HistoricalContamination("RESERVE", fam, "CONTAMINATED", "g1 JUDGE read (lineage parent); see g1 capsule access ledger"))
    return hc
