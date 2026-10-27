# Measurement model (implemented: `mechanism/measurement.py`, instance `randi/benchmark.measurement_model`)

Every field is `MeasurementField(value, status, source, uncertainty, note)`; status ∈ KNOWN_FROM_METADATA,
INFERRED_FROM_DATA, ASSUMED, UNKNOWN. KNOWN/INFERRED/ASSUMED require a source; UNKNOWN may not carry a value.
The discriminator gives reliability 0 to any test that requires an UNKNOWN field, ×0.8 per ASSUMED/INFERRED
field; the engine never runs a test with an UNKNOWN required field.

## Randi 2023 (DANDI:001075) instance, highlights
KNOWN: 25 planes/volume, 20 ms/frame, 0.5 s/volume, per-frame timestamps, alternating sweeps, 1.22 µm/plane,
stimulus onset/duration (0.5 s WT), power 1.2 mW **constant** (power dependence untestable), spot pixel per stim,
3.1 µm lateral FWHM footprint (Extended Data Fig. 2 legend), Leifer extraction box per session, connectome.
ASSUMED: 0.42 µm/px (wormdatamodel code constant, not in NWB); camera offset 100 counts.
INFERRED: green DANDI→raw transform (z flip, xy identity; validated by target dF in 21/24 animals), red transform
(z flip, dy +16, dx +3; tagRFP brightness only), LoG segmentation, label transfer.
UNKNOWN: stimulation depth semantics, axial footprint, optical PSF, DANDI voxel masks (conversion bug), per-volume
ROI coordinates, unc-31 positions.

## Scan-timing correction
Per-sample acquisition time = timestamp of the frame that imaged the nucleus's plane in that volume.
`onset_interval(t, y, t_stim)` → onset ∈ (t_last_below, t_first_above]; baseline = samples acquired before
`t_stim`; threshold mean + `ONSET_K_SD`·SD sustained for 2 samples. `ONSET_K_SD = 2.0` was chosen on DISCOVERY
detection coverage only (59% of top-30%-amplitude targets detected, 0% of bottom-30%; k=3 detected 33%), before
any latency outcome was computed. Delays are interval differences; `delay_call` reports LAG / LEAD / UNRESOLVED
fractions and the median interval width (effective resolution). Precedence is only claimed for intervals that
exclude zero. `naive_volume_onset` (volume index only) is retained to demonstrate the artifact (fixture S9).

Response windows in `observations.build_table` are also defined by per-sample acquisition time (baseline:
t < onset; response: onset+0.25 s … onset+6.5 s). A volume-indexed baseline leaked post-stimulus samples of
late planes and produced spurious pre-stimulus covariance in synthetic optical worlds; this was found by S1 and fixed.

## Measurement model v2 (generation g1): red registration demoted to UNKNOWN
g1 scout diagnostics on DISCOVERY (`results/mechanism/randi_proximity/scout_g1/`) found that the INFERRED red transform
places boxes on near-background pixels (red ~10 counts above offset vs green ~58; pre-stimulus red dF/F SD 0.21), and an
exhaustive translation search (z flip × dz ±3 × dy, dx ±40) finds no mapping from raw-detected green nuclei onto tagRFP
nuclei. The transform had only been validated against DANDI centroids by brightness. `red_registration_transform` is
therefore UNKNOWN in v2 and T5 is blocked; g0 T5 outcomes are recorded as measured at unvalidated locations (deviation).
