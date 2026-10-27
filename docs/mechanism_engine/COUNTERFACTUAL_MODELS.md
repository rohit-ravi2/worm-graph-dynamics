# Counterfactual models (implemented: `mechanism/generators.py`, `mechanism/nulls.py`)

Payloads are produced in the raw_traces format and pass through `observations.build_table` and `contrasts`,
identical to real data. Acquisition: 25 planes, 20 ms frames, alternating sweeps, one sample per nucleus per
volume at its plane's time; per-sample multiplicative noise; camera offset 100.

Mechanisms: optical (anisotropic Gaussian footprint around the spot, target-independent, latency 0–0.1 s);
contamination (receiver measurement += distance-weighted copy of the target's MEASURED signal including its noise);
stimulation-light artifact (green+red blip in the stim volume near the spot); gap (coupled receivers, 5–50 ms);
chem (direct chemical targets, 0.3–1.0 s); poly (two-hop, non-adjacent, 0.8–2.0 s); peptide (within radius,
wiring-independent, 2–5 s); global (broad activation, 0.5–3 s); spontaneous activity; sham windows with
spontaneous pseudo-target activity. Synthetic wiring (`spatial_connectome`) decays with distance so topology
correlates with proximity, making distance-matching necessary.

Hostile-null comparison (`nulls.GENERATORS`, parameters fixed a priori): optical_only, extraction_overlap_only,
light_artifact_only, gap_only, chem_only, polysynaptic_only, peptidergic_only, global_state_only,
mixture_optical_plus_chem, mixture_contamination_plus_optical. The only data-derived setting is the target response
probability (INFERRED from DISCOVERY). Output: per-generator signature and agreement with observed DISCOVERY calls.
No parameter is tuned to maximise agreement.
