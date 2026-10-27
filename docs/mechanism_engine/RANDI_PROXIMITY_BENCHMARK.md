# Randi proximity benchmark (implemented: `src/wormlab/mechanism/randi/`)

**Question.** Why does optogenetic stimulation of a nominally single neuron raise response probability in
spatially near neurons?

## Known limitations of the inputs
1. An earlier note claimed raw-image analysis "found a stable transformation between raw green and red spaces". In fact
   the DANDI raw↔segmentation linkage is BLOCKED: Base
   signals could not be reproduced (r ≤ 0.07); a green transform (z flip) is validated only at stim-target level;
   the red transform is inferred from tagRFP brightness only. The benchmark therefore uses an independent,
   frozen raw extractor, and records both transforms as INFERRED.
2. DANDI voxel masks carry a conversion bug (box offsets added three times); run-1 box-unit results were
   corrected in `analyses/randi_distance_map/REPORT.md`.
3. All 110 WT animals were already inspected in runs 1–2 for several statistic families → handled as historical
   contamination (below), not ignored.
4. v1.2 values reverified from artifacts (docs/discovery/V12_LIVE_GATE.md, capsule 9232e6ba verifies): R1 0.646,
   B0 0.574; L3 gates: distance-only 0.733; capsule tests pass.

## Data and partitions
Frozen extractor `randi/raw_traces.py` (EXTRACTOR_VERSION raw_traces/1.0, sha256 4c92fb94…): per animal, LoG nuclei on
the mean green volume; per stim (and sham windows) traces of target + all nuclei ≤ 15 µm + 30 seeded far nuclei, box
3×5×5 / core 1×3×3 / red box, each sample with its plane's acquisition time. Partitions (`partition_manifest.json`,
sha256 6c0626b1…): DISCOVERY 24, CONFIRMATION 40 (sealed), RESERVE 46. Unit = animal.

## Hypotheses (g0)
H1 static measurement contamination (point-spread / extraction copy of target signal); H2 direct off-target
photostimulation; H3 gap-junction / electrical coupling; H4 direct chemical propagation; H5 polysynaptic propagation;
H6 extrasynaptic / neuropeptidergic; H7 global / shared state; H9 stimulation-light artifact (generated from the
measurement model after the compiler rejected T5 as non-discriminating); H8 = 28 pre-registered pairwise mixtures.

## Tests
T0 proximity effect (phenomenon gate); T1 distance-matched topology (MH RD, animal × 2-µm strata, permutation null);
T2 topology-matched distance; T3 scan-corrected onset latency; T4 spot-offset directionality; T5 red control channel;
T6 near-field when the target is flat; T7 path length at fixed distance; T8 sham near-field (conditional on
spontaneous pseudo-target activity); T9 animal replication; T10 far-field breadth stim vs sham; T11 pre-stimulus
covariance (mask / measurement-sharing signature); X1 nucleus-vs-background (historical import only).
Floors fixed a priori: rate contrasts 0.02–0.05, latency 0.25 s, covariance 0.05 Fisher-z, replication 0.1.

## Historical contamination (`benchmark.historical_contamination`)
DISCOVERY: proximity_rate CONTAMINATED, haze_vs_discrete CONTAMINATED. ALL_WT PARTIAL: proximity_rate,
proximity_replication, topology_given_distance, pathlength_given_distance, distance_given_topology,
onset_latency_scan_corrected (coarse uncorrected latency seen), target_flat_nearfield, red_channel_nearfield.
UNTOUCHED on CONFIRMATION: spot_offset_directionality, sham_nearfield, farfield_breadth, baseline_covariance.
Consequence: only T4, T8, T10, T11 can produce CONFIRMATION-grade evidence; all other confirmation outcomes are
PARTIAL-grade (can weaken or weakly support; cannot falsify or support alone).

## unc-31 contamination ledger (`benchmark.UNC31_LEDGER`)
Inspected: significance counts / LOST–GAINED (L3 gates), near-field by atlas distance × target amplitude (run 1 T5),
loss rates at occ ≥ 2/3/5, release-machinery gate, PR dimensionality, WB-2 Spearman, peptide sign flips.
No per-animal unc-31 coordinates exist in DANDI:001075. Verdict: a prospective unc-31 Level-3 test is NOT earned on
available data — every computable family is contaminated and geometry-dependent families are not computable.

## Generation g0 (real run `randi_proximity_g0_20260916T201320Z`, capsule root b2088624…)
Verdict under policy 1.0: MECHANISM_DISFAVORED (see FINAL_REPORT.md for outcomes and the policy analysis).

## Generation g1 (declared after g0)
Motivating evidence (cited in every g1 hypothesis): g0 CONFIRMATION outcomes of T11 and T5, and the g1 scout diagnostics
(sha256 prefix recorded). Scout diagnostics on DISCOVERY:
* T11 is not a shared-frame artifact: near−far pre-stimulus Fisher-z within |Δplane| = 0: +0.35 [0.27, 0.44];
  |Δplane| 1–2: +0.13 [0.07, 0.19]; |Δplane| ≥ 3: +0.30 [0.10, 0.48].
* The proximity effect persists across plane classes (RD +0.08, +0.10, +0.16).
* The red channel measure is invalid (above).
Changes: hypotheses H*.g1 add a T11b (plane-matched) prediction identical to their T11 prediction; test T11b added;
measurement model v2; policy 1.1; historical contamination adds CONFIRMATION = CONTAMINATED for every family judged in g0;
JUDGE partition = RESERVE (46 animals never extracted before g1), released by the AUDITOR to the g1 hypotheses and mixtures;
capsule parent = g0 capsule.
