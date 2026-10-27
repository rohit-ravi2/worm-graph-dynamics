# Randi et al. 2023 significance statistic (q, q_eq, occ1): what the code does

Sources (read-only clones):
- `pumpprobe` commit 1dbc5e0: `pumpprobe/Funatlas.py`, `pumpprobe/Fconn.py`, `scripts/qvalues_map.py`, `scripts/fconnectivity/funatlas_plot_intensity_map.py`, `scripts/fconnectivity/build_wfc_atlas.py`, `pipelines/fit_kernels_multi_datasets.sh`.
- `wormdatamodel` commit 2ab9561: `wormdatamodel/signal/signal.py`.
- `multipy` 0.16 (PyPI sdist): `multipy/fdr.py`.

In this file, "verified" means checked numerically against `funatlas.h5`; the evidence is in `out/pq03_validation.json` and `out/pq04_repro_*.json`. Everything else is read from the code.

## 0. Which script produced `funatlas.h5` q?

- The h5 builder is not in the repo. `wormneuroatlas/data/funatlas.h5` has `time_compiled` = 2023-06-28.
- `build_wfc_atlas.py:33-38` loads `funatlas_intensity_map_cache_q.txt`. That file is written by `funatlas_plot_intensity_map.py:317-326` when the script runs with `--alpha-kolmogorov-smirnov --inclall-occ --save-cache`.
- That script calls `Funatlas.get_kolmogorov_smirnov_q(inclall_occ2, strain=..., return_tost=True, tost_th=1.2)` (`:134-139`). `qvalues_map.py:70-79` makes the same call.
- Inferred flags for the published product (verified in §5):
  - inclusive occurrence matrix: `inclall=True`
  - `req_auto_response=True`
  - no bilateral merging (300 ids)

## 1. Events: which (responder i, stimulated j) observations enter

**Datasets and labels**
- Datasets are the WT list with tag `mutant` excluded (`funatlas_plot_intensity_map.py:12`). The export has 113 of them.
- Labels map to atlas indices via `Funatlas.ids_to_i` (`Funatlas.py:441-482`) and `approximate_ids` (`:325-439`). With no merging this is an exact string match after removing `_`, with AWCOFF→AWCL and AWCON→AWCR (`:369-377`).
- Unmatched labels (blank, `RMG?`, trailing spaces) get -1 and are dropped.

**Stimulated neuron**
- `Fconn.stim_neurons[ie]` is a reference-volume ROI index, set to the closest neuron to the stim target (`Fconn.py:189-214`).
  - -1: no neuron
  - -2: excluded (bubble, `Fconn.py:511-513`; or artefact cross-check with `enforce_stim_crosscheck`, `Funatlas.py:217-239`)
  - -3: manually located target, not in the reference volume. Its label comes from `stim_neurons_compl_labels` (`Funatlas.py:259-262`).
- `aj = atlas_i[stim_neurons]`. Events with `aj < 0` are skipped (`Funatlas.py:763-764`).

**occ2 and occ1** (`Funatlas.get_occurrence_matrix`, `:716-807`)
- With `inclall=True`, every identified neuron in the dataset is an occurrence for every stim with `aj >= 0` (`:769-772`, `:795-796`).
- With `req_auto_response=True`, a stim is kept only if the stimulated ROI is in `Fconn.resp_neurons_by_stim[ie]` (`:778-781`).
- `occ2[i][j]` is a list of `{ds, stim, resp_neu_i}`. `occ1[i,j]` is its length (`:786-787`).
- `occ1` in `funatlas.h5` is this inclusive count, before any NaN check.

**Autoresponse classifier** (`Fconn.from_objects`, `Fconn.py:236-507`)
- Run by `responses_detect.py` with the flags in `fit_kernels_multi_datasets.sh` (`--ampl-min-time:4 --ampl-thresh:1.3 --deriv-min-time:4 --deriv-thresh:0.02 --smooth-mode:sg_causal --smooth-n:13 --smooth-poly:1 --matchless-nan-th:0.5 --matchless-nan-th-added-only --nan-th:0.05`).
- A neuron responds if all three criteria hold:
  - **Derivative:** |post derivative − mean |pre derivative|| / median(pre signal) > thresh for at least `deriv_min_time/Dt` volumes (`:274-290`).
  - **Amplitude:** contiguous excursion beyond `ampl_thresh ×` (3rd-largest |pre|), or half that for twice as long. There is an alternative criterion for neurons that responded to the previous stim (`:309-492`).
  - **NaN check:** `:270-272`.
- Manually flagged removals and additions (`Fconn.load_flagged_responses`, `:574-607`) edit the result. **Those files are not on disk**, so the classifier cannot be re-run faithfully. See §5 for how the inclusion set was recovered instead.

**Windows** (`Fconn.py:242-247`)
- `shift_vol = int(delta_t_pre/Dt) = 60`, with `delta_t_pre = 30 s` (`responses_detect.py:16`) and `Dt = 0.5 s`.
- `i0 = max(0, ee − 60)`.
- `i1 = min(i0 + 120 + shift_vol_ie, T)`. The segment is therefore 30 s before to 60 s after the stim.
- The feature code uses the global `fconn.shift_vol` (= 60) even for a first stim with `ee < 60` (`Funatlas.py:2636`).

## 2. Signal

- `Signal.from_file(folder, "green", remove_spikes, smooth sg_causal 13/1, photobl_appl, matchless options)` (`Funatlas.py:271-273`; `signal.py:188-329`).
- The preprocessing order in code is: NaN interpolation → inf removal → photobleaching double-exponential fit and application → spike removal (> 5 SD) → causal SG smoothing (`signal.py:178-185`).
- `nan_mask = isnan(raw)` is recorded before interpolation (`signal.py:104`). Manually added neurons with too many NaNs are replaced by "matchless" traces (`:293-324`).
- **Verified:** the per-trial dF/F in `funatlas.h5 wt/dFF_all` is reproduced exactly only when the features are computed on the Leifer-exported traces as-is (`Funatlas.export_to_txt` writes `sig[ds].data`, `Funatlas.py:279-309`). Adding the photobleaching, spike or smoothing steps on top changes the values by 1e-3 to 1e-2. The array the published statistics read is therefore the exported array. (Scratch probe on dataset 0: all six test events match to 0.0 only with none of those steps applied.)

## 3. Per-event features (`Funatlas.get_significance_features`, `Funatlas.py:2361-2421`)

For responder trace `x` (the whole recording) and segment `y = x[i0:i1]`:

**NaN gate**
- The event is dropped if the longest contiguous NaN run in `nan_mask[i0:i1]` exceeds `nan_th × (i1 − i0)` (`Fconn.nan_ok`, `Fconn.py:1741-1758`).
- `nan_th = Funatlas.nan_th = 0.05` (`Funatlas.py:11`), i.e. more than 9 volumes. **Verified:** on columns with a DANDI NaN mask, 0.05 predicts `dFF_all` membership with 100% agreement; 0.1 gives 93.9% and 0.3 gives 89.1%.

**Baseline and noise**
- `baseline = mean(y[:60])`.
- `pre = sqrt(median(rolling var, window 8, reflect-padded))` over the full segment (`signal.py:991-1035`, `1056-1063`). The event is dropped if `pre == 0`.

**Feature 1: `dff` = ΔF/σ**
- `mean(y[60:120] − baseline) / pre`, the mean over 30 s after the stim.
- **Decaying-response correction** (`:2400-2414`) applies if the causal SG first derivative at stim time is negative (`savgol_coeffs(39, 1, pos=38, deriv=1)`). The trace is then `y[60:60+L] − [A·exp(fd0·t) + zero]`, where:
  - `zero` = SG(39, 1)-smoothed value 60 volumes later (or the last sample)
  - `A` = smoothed value at stim time − `zero`
  - `t` = time in seconds
- `get_ctrl_distributions` prints "USING DF/SIGMA" (`:2485`). The Methods text says "ΔF/F0 amplitude"; the code tests ΔF/σ.

**Feature 2: `sd`**
- Mean over 15 s after the stim (volumes 60-89) of the SG second derivative (`savgol_coeffs(39, 2, deriv=2, delta=Dt)`, `np.convolve` mode "same"), `:2366-2368`, `:2397`.

**Published map value `dFF`** (not used for q)
- `mean(decay-corrected ΔF) / mean(y[:60])` (`funatlas_plot_intensity_map.py:192-200` with `--correct-decaying`).
- `dFF_all[i,j]` lists these per event. They pass the same `get_significance_features` gate, so `dFF_all` membership equals KS membership.

## 4. Null and pair statistics

**Null** (`Funatlas.get_ctrl_distributions`, `:2424-2552`)
- For each dataset in `/projects/LEIFER/Sophie/Lists/No_stim_list.txt` (WT; a separate list exists for unc-31, `:2441`), all neurons and all "stim" events get the same features with `nan_th = 0.05`. Their pooled `dff` and `sd` values are the empirical null.
- The null recordings carry optogenetics event indices, so the stimulation timing protocol exists without effective stimulation. Strain and animal count are not stated in code or paper.
- **The null recordings are not on disk.** That is the one irreducible SUBSTITUTION; see §6.

**Pair p** (`get_kolmogorov_smirnov_p`, `:2584-2666`)
- For each ordered atlas pair with ≥ 1 valid event:
  - `pdff = kstest(ctrl_dff, dff_, two-sided)`
  - `psd = kstest(ctrl_sd, sd_, two-sided)`
  - zeros → 1e-10
  - `p = Fisher(pdff, psd)`
- With two arrays, `kstest` is scipy's two-sample KS. Mode "auto" is asymptotic when either n > 10000, which holds for any realistic null.

**q** (`get_kolmogorov_smirnov_q`, `:2668-2704`)
- `p == 0 → 1e-10`, then multipy `qvalue` (Storey & Tibshirani 2003) over all finite p (`fdr.py:156-214`).
- π0 is a cubic `UnivariateSpline` over λ = 0…0.95 of #{p > λ}/(m(1−λ)), evaluated at 1. It is reset to 1 if outside [0, 1].
- `q_(i) = min(π0·m·p_(i)/i, q_(i+1))`.

**q_eq, the equivalence test**
- `tost_th = 1.2` (`qvalues_map.py:79`; `funatlas_plot_intensity_map.py:136`).
- ε = 1.2·std(control feature) (`:2592-2597`).
- `ttost_ind(ctrl, pair, −ε, ε)` (statsmodels, pooled variance) for each feature, zeros → 1e-10, Fisher, Storey (`:2655-2661`, `:2691-2695`).
- Quirk: `tost_pvalues[pvalues==0] = 1e-10` uses the regular-p mask (`:2692`).

**Binomial q** (`get_qvalues`, `:2211-2359`) is an older, unused path: binomial test of `occ1` vs `occ3` against the response probability in no-stim recordings. `get_qvalues2` (`:2198-2209`) loads a pickle.

## 5. occ1, occ2 and occ3 definitions

- **occ1[i,j]** (`get_occurrence_matrix`): number of occurrences of i when j was stimulated.
  - Default (`inclall=False`) counts classifier-detected responses only.
  - `inclall=True` counts all identified neurons.
  - `req_auto_response` keeps only stims with an autoresponse.
- **occ2[i,j]**: the list behind occ1, with keys ds, stim, resp_neu_i.
- **occ3[i,j]** (`get_observation_matrix`, `:809-855`): number of times j was stimulated while i was observed.
  - `get_observation_matrix_nanthresh` (`:857-907`) also requires the NaN gate and local std ≠ 0.
  - occ3 is not stored in `funatlas.h5`.
- **Verified for `funatlas.h5`:**
  - Our inclusive count over all labeled stims is ≥ published occ1 for 99.98% of pairs, but sums to 219,954 against 130,682. Published occ1 is therefore the autoresponse-filtered inclusive count.
  - With the inferred stim inclusion (§ below) our count equals published occ1 exactly for 95.0% of pairs (sum 129,408 vs 130,682).

**Recovering the published inclusion set.** The autoresponse classifier depends on manual flag files that are not on disk. We therefore recovered the inclusion set from the released product:
- Every per-event dF/F we computed was matched to the published `dFF_all[i,j]` list, with |relative difference| < 1e-9.
- 108,116 of 109,271 published per-event values (98.94%) are reproduced exactly.
- 1,875 of 3,224 labeled stims are included. For comparison, Extended Data Fig. 2i reports "0µm, n=1894" autoresponse events.
- The 1,155 unmatched published values cluster on about 18 source neurons, roughly 40 to 116 values each. They are consistent with stims whose target label was not exported: the 28 stims coded -3, plus stims coded -2 in the export. They cannot be recovered.
- One export/published target-label disagreement was found (dataset 97, stim 18: URYDL vs URYDR).

## 6. What cannot be reproduced from disk, and the substitutes

| Component | Status |
|---|---|
| Signal, windows, features, NaN gate | Reproduced exactly (per-event dF/F bit-identical for 98.9% of published events) |
| Autoresponse inclusion | SUBSTITUTION: recovered by exact value matching to `dFF_all` ("oracle membership"). Re-running the classifier is impossible without the manual flag files. |
| Null distribution (`No_stim_list` control recordings) | SUBSTITUTION, not on disk. See the options below. |
| KS / Fisher / Storey / TOST | Re-implemented. Fast KS equals scipy `kstest` (max \|Δp\| = 0 on 60 random pairs). Fast TOST equals statsmodels `ttost_ind` (max \|Δp\| = 0 on 20 pairs). |

**Null substitutes tested** (`pq04_reproduce_q.py`):
- **(a) `spont`:** the two Leifer "spontaneous_activity" recordings on disk, with a pseudo-stim at every volume. This is the only stimulation-free data on disk. It is far wider than the WT event distribution (dF/σ 5–95%: −5.0 to 13.8, vs −2.0 to 6.3 for all WT events).
- **(b) `wtall` and `wtallx<s>`:** all WT events pooled, optionally scaled.
- **(c) `wtqgt<τ>`:** WT events from pairs whose published q > τ. This is calibrated on the published product.

**Primary null: `wtqgt0.5`.** The selection rule was stated before looking at downstream results: match the published Storey π0 (published max q = 0.650 ≈ π0; `wtqgt0.5` gives π0 = 0.649). The full scan is reported, including τ = 0.55, which has the best Jaccard.

**Sensitivity null: `wtallx0.7`.** It does not use published q at all; it has a one-parameter width calibration.
