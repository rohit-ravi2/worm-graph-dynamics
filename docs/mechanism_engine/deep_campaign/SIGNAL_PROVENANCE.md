# Signal provenance: design, families, validation (§5-16, §22-25)

## Question

The near-field effect (receivers < 6 um from a stimulated target respond more often) is either
**measurement**, where photons from the target land in the receiver's measurement, or **receiver-own**, where the
receiver's own indicator responds (off-target light H2, electrical coupling H3, a local field H10, diffusion H6).
Every family below is built to separate those two, and each states which confound it cannot remove.

## Data streams (frozen extractors, sealed per partition)

| stream | module | content | why it exists |
|---|---|---|---|
| `provenance_traces/1.0` | `randi/provenance_traces.py` + `roi_extract.py` | every volume of the recording; for every LoG nucleus 10 ROI means (3 plane boxes, core, 4 half-boxes, ring, wide); per-frame background percentiles; per-plane times and depths; the static mean volume | outcome-independent optical kernel; long spontaneous periods; ROI-geometry tests |
| `event_crops/1.0` | `randi/event_crops.py` | per stim event: pre (v-8..v-1) and post (v+1..v+12) mean voxel crops (17x81x81) around the target | voxel-level unmixing |
| `leifer_osf` (loader, not an extractor) | `randi/leifer_osf.py` | Leifer per-volume tracked coordinates from the public OSF deposit (acquisition-order z converted; DANDI-timestamp alignment) | registration / event-locked displacement audit |

The synthetic generator `provenance_synth.py` renders volumes (nucleus x PSF light, alternating sweep with per-plane
timing, shot/read/frame noise, optional motion, PSF tails, nucleus-size heterogeneity, activity-dependent halo) and pushes
them through the same segmentation and ROI code, so contamination in synthetic worlds comes from optics. It is never
written as a copy term.

## Measurement model (all inferred from the static image, never from outcomes)

* **Static light model** (`signal_provenance.fit_light_model`): one anisotropic Gaussian per detected nucleus, with a
  shared sigma per animal chosen by grid search on isolated nuclei, sub-voxel centroids, non-negative amplitudes, and a
  per-plane background.
* **Optical kernel** kappa[t->n] for every ROI type = (W_roi[n] G_t) / (W_box[t] G_t).
* **Unmixing** (`unmixing.py`): each evoked dF crop is fitted by weighted least squares as nucleus shapes (free amplitude
  each), plus two free radial halo families around the target (isotropic and PSF-anisotropic, knots 1-17 um), plus a
  constant. The receiver's own dF/F is a_n / A_n, where A_n is the same design fitted to the pre crop.
* The halo basis was chosen on synthetic worlds only (`synthetic_battery/halo_basis_variants.json`); it is the most
  conservative variant against own-indicator claims.

## Families

| id | family | statistic | contamination predicts | receiver-own predicts | role |
|---|---|---|---|---|---|
| T13 | psf_kernel_sufficiency | near-far response RD after subtracting kappa*dC_target | NULL with a correct static kernel; POS with a halo | POS | measurement sufficiency of the static kernel |
| T14 | brightness_scaling_copy_ratio | per-bin WLS own-light slope, free kernel-scale and flat-copy terms | NULL | POS | **demoted to descriptive** (battery v1: false POS in B_CHEM, no power in B_H2/B_H3) |
| T15 | evoked_roi_profile | slope of evoked ROI profile on the kappa-predicted profile | about 1 | about 1 (optics exist in every world) | kernel calibration check, not a discriminator |
| T16 | spontaneous_evoked_slope_identity | standardised lag-1 cross-covariance excess over kappa prediction, near minus far | ANY (a halo copies spontaneous activity) | POS for spontaneous coupling (H3/H10), NULL for H2 | sub-classification after T17 |
| T17 | evoked_unmixing_proximity | near (<6) vs mid (9-15 um) RD of unmixed receiver responses; identical-row box RD and explained fraction | NULL | POS | **primary** measurement vs receiver-own discriminator |
| T6u | target_flat_nearfield_unmixed | T17 on target-flat events | NULL | NULL (H3/H6/H10), ANY (H2: fixture S16) | O-family audit |
| T18 | topology_given_distance_unmixed | gap-edge vs none RD, animal x 2-um bins, unmixed responses | NULL | POS only for H3 | H3 vs H10/H2 |
| T19 | spot_offset_directionality_unmixed | spot-side vs opposite RD | NULL | POS for H2 | H2 (weak: offsets small) |
| T20 | axial_lateral_anisotropy | axial (dz >= dxy) vs lateral RD in 3-6 um minus 9-15 um control, L/R homologs excluded | POS (axially elongated kernel and box) | NULL | measurement vs biology independent of the kernel model |
| W1-W10 | descriptives | band transfer, alpha stability, waveform, shared-voxel noise, plane-time vs naive lag, linearity, spontaneous windows, spatial copy curve, reciprocal pairs | | | exploratory maps |

## Known unresolved confounds (stated before the judge)

1. **Anisotropic activity-dependent structure of the target.** A neurite or other process passing through a receiver's
   light footprint is not a radial halo, so unmixing cannot absorb it.
2. **Real PSF misspecification.** Real PSF tails and nucleus-size variation are only approximated by a Gaussian model.
   Battery v3 measures the consequence at the real fitted sigma.
3. **Unlabeled L/R homologs.** In side-mounted worms these lie along the imaging axis and can be electrically coupled.
   T20 excludes labeled homolog pairs only.
4. **Shared stimulation effectiveness** (fixture S16). Target-flat nulls do not exclude off-target stimulation.

## Validation

* **Battery v1/v2.** 18-21 hostile worlds at 8 and 24 animals:
  `results/mechanism/deep_campaign/synthetic_battery/battery_report_v1.json` and `battery_report_v2.json`.
* **Battery v3.** Worlds calibrated to DISCOVERY static statistics (fitted sigma 2.8 px / 1.8 planes, brightness log-sd
  1.1, receiver density), with PSF tails and size heterogeneity, plus a §27 parameter grid:
  `synthetic_battery/v3/battery_report_v3.json`.
* **Prediction audit.** `results/mechanism/deep_campaign/prediction_audit.json` feeds `invalid_predictions` to policy 1.3.
* **Deviations.** All are recorded in the battery reports (`deviations`) and in this campaign's report.
