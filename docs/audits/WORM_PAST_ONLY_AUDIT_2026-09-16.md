# Worm past-only lag audit (2026-09-16)

**Scope:** a bounded audit of how the frozen Atanas within-forward-speed recoverability
result (worm_v1) handled time. **Result:** the neural gain over the AR(3) speed baseline
shows up only when the neural window contains the target frame or frames after it. With
a strictly past-only window, no representation from the headline family gives a
positive predictive gain. Everything frozen is untouched (see §6).

Numbers below come from
`results/_audits/worm_past_only_2026-09-16/past_only_audit_results.json`
(sha256 `d837fa6fa8103f1d130ca1f9cb77d959fa3a6dc51517449c7a8453290f7a8908`) and the
per-job files in `runs/`. The script is `scripts/audits/worm_past_only_audit.py`.

The audited decoding pipeline (`wormlab.sufficiency`, `wormlab.decisions`) and its frozen result files live in my
larger working repository and are not part of this release, so the script here documents exactly what was run but
does not execute on its own. The audit's outputs are included.

---

## 1. The issue

`src/wormlab/sufficiency/evaluate.py::_pair_lag` pairs:

- the **representation at frame t** with
- the **target `tgt[tp]`, where `tp = t + lag`**, and
- an **AR(3) baseline** `[tgt[tp-1], tgt[tp-2], tgt[tp-3]]`.

The representations look backward in time (`transforms.py`):

| representation | frames used by the feature at frame t |
|---|---|
| `raw` | t |
| `deriv` | t-1, t (a backward difference) |
| `delay_h` | t-h … t |

`LAG_GRID = range(-5, 6)`. A negative lag puts the target *before* the latest neural
frame, so the neural window reaches past the target. The last neural frame it uses is
`tp - lag`.

The headline `panel|delay_h5` picked modal lag **−2** (all 5 outer folds chose −2). So:

- The decoder saw neural frames `tp-3 … tp+2`: the target frame plus **two frames (1.2 s)
  after it**.
- The baseline saw only `tp-3 … tp-1`.

Every branch in the frozen 12-branch `recoverability_top` set had a modal lag of −1, −2
or −3. So the reading "neural activity predicts future speed about 1–2 s ahead" has the
direction backwards. `evaluate.py`'s own comment is right ("NEGATIVE = behavior-leading").
The comment at `evaluate_unit_local.py:58` ("L<0 neural-leading") is inverted, because
that module uses the same `tp = idx + lag` pairing.

Across the whole frozen v2-detector grammar, 46 branches meet all of: incremental > 0.0045,
lower CI bound > 0, and dual-confirmer above-null. Their modal lags split as:

| modal lag | branches |
|---|---|
| negative (behaviour-leading) | 43 |
| 0 | 3 |
| positive | 0 |

This count is read directly from
`results/empirical_sufficiency_mvp_v2detector/representation_results.json`.

## 2. What was run

- **Pipeline:** the same frozen pipeline, with no source edits.
  - `wormlab.sufficiency.evaluate.evaluate_representation(spec, dataset, splits, run_nonlinear=True, null_n=20, seed=0, detector_backend="v2")`.
  - This is the canonical v2 dual-confirmer detector (FROZEN.md §0; `run_mvp.run_v2_regression`).
- **Data and splits:** `data.load_dataset()` (38 worms, 50-neuron panel) and `make_outer_splits(k=5, seed=0)`.
- **How the grids were changed:** `evaluate.LAG_GRID` was **monkeypatched in-process**. The script asserts that every selected lag lies inside the patched grid.
- **Grids:**

  | grid | lags | what the neural window may contain |
  |---|---|---|
  | `original` | −5…5 | anything; reference run |
  | `concurrent` | 0…5 | may end *at* the target frame |
  | `past_only` | 1…5 | ends strictly *before* the target frame (neural window t-h..t, target tp = t+lag > t) |

- **Representations:** the full frozen 12-branch `recoverability_top` set plus the instantaneous `panel|raw`. That is `panel|{raw, delay_h1, delay_h2, delay_h3, delay_h5, deriv}`, `resid1|{delay_h1, delay_h2, delay_h3, delay_h5, deriv}` and `gmode3|{delay_h3, delay_h5}`.
- **Extras at the selected modal lag:**
  - The original **stable** linear-only null (`null_n=10`, as in `results/empirical_sufficiency_mvp`).
  - A descriptive per-lag profile: fixed-alpha ridge `_fast_incr` over lags −5…5.
- **Decision semantics:**
  - `wormlab.decisions.recoverability.classify_branch` with the policy from `specs/worm_positive_temporal.json` (floor 0.0045).
  - Branch numbers are mapped the same way as `decisions/applier.py::_apply_worm`. `null_significant` is the v2 dual-confirmer `above_null_p95`.
- **Temporal-history sufficiency:** recomputed with the frozen `analysis.compression_curve` (unmodified). The points are `panel|raw` (h=0) and `panel|delay_h{1,2,3,5}`, as in `run_mvp.run`.
- **Reproduction check:** all 13 representations under the `original` grid match the frozen v2-detector numbers exactly. That covers selected lags per fold, score, incremental, CI, nonlinear incremental, p_linear and p_nonlinear. The stable-null p95 also matches the original stable run.
- **Runtime:** 3 worker processes with single-threaded BLAS.
  - Smoke test: 187 s wall for `panel|delay_h5 × past_only`.
  - Main run: 1016.6 s for 30 jobs.
  - Extension: 101.4 s for 9 jobs.
  - Total ≈ 22 min wall.

## 3. Results

**Columns:**

- **incr:** outer-fold mean per-worm R² gain over AR(3), with the frozen 90% bootstrap CI.
- **p lin/nl:** v2 dual-confirmer null p-values. With 20 nulls the minimum is 0.0476.
- **above-null v2 / stable:** the v2 dual decision and the original linear-only p95 decision.
- **incr>floor / CI_lo>floor:** comparison with the frozen 0.0045 floor.
- **label:** the corrected software-v1 decision label.
- **last neural frame:** the latest neural frame the decoder saw, relative to the target frame `t_p`.

| grid | representation | selected lags (5 folds) | modal lag | last neural frame | incr | 90% CI | nonlinear incr | p lin/nl | above-null v2 | stable | incr>floor | CI_lo>floor | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original | panel\|raw | [-2, 4, 4, 1, 2] | +4 | t_p-4 | -0.0016 | [-0.0051, +0.0020] | -0.0190 | 0.4286/0.0476 | False | False | False | False | unresolved_no_signal |
| original | panel\|delay_h1 | [-1, -1, -1, -1, -1] | -1 | t_p+1 | +0.0414 | [+0.0351, +0.0488] | +0.0676 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| original | panel\|delay_h2 | [-1, -2, -2, -1, -1] | -1 | t_p+1 | +0.0466 | [+0.0391, +0.0546] | +0.0813 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| original | panel\|delay_h3 | [-2, -2, -2, -1, -2] | -2 | t_p+2 | +0.0526 | [+0.0443, +0.0614] | +0.0909 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| original | panel\|delay_h5 | [-2, -2, -2, -2, -2] | -2 | t_p+2 | +0.0531 | [+0.0444, +0.0627] | +0.0914 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| original | panel\|deriv | [-1, -1, -1, -1, -1] | -1 | t_p+1 | +0.0413 | [+0.0350, +0.0488] | +0.0676 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| original | resid1\|delay_h1 | [-1, -1, -1, -1, -1] | -1 | t_p+1 | +0.0375 | [+0.0311, +0.0451] | +0.0605 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| original | resid1\|delay_h2 | [-1, -1, -2, -1, -1] | -1 | t_p+1 | +0.0441 | [+0.0373, +0.0517] | +0.0757 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| original | resid1\|delay_h3 | [-2, -2, -2, -1, -2] | -2 | t_p+2 | +0.0479 | [+0.0401, +0.0568] | +0.0838 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| original | resid1\|delay_h5 | [-3, -2, -2, -2, -2] | -2 | t_p+2 | +0.0488 | [+0.0409, +0.0581] | +0.0838 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| original | resid1\|deriv | [-1, -1, -1, -1, -1] | -1 | t_p+1 | +0.0374 | [+0.0310, +0.0451] | +0.0606 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| original | gmode3\|delay_h3 | [-2, -2, -2, -2, -2] | -2 | t_p+2 | +0.0394 | [+0.0338, +0.0455] | +0.0579 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| original | gmode3\|delay_h5 | [-2, -3, -3, -2, -3] | -3 | t_p+3 | +0.0409 | [+0.0349, +0.0474] | +0.0613 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| concurrent | panel\|raw | [4, 4, 4, 1, 2] | +4 | t_p-4 | -0.0028 | [-0.0056, -0.0003] | -0.0190 | 0.4286/0.0476 | False | False | False | False | unresolved_no_signal |
| concurrent | panel\|delay_h1 | [0, 0, 0, 0, 0] | 0 | t_p | +0.0151 | [+0.0091, +0.0214] | +0.0159 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| concurrent | panel\|delay_h2 | [0, 0, 0, 0, 0] | 0 | t_p | +0.0170 | [+0.0109, +0.0235] | +0.0204 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| concurrent | panel\|delay_h3 | [0, 0, 0, 0, 0] | 0 | t_p | +0.0161 | [+0.0097, +0.0226] | +0.0184 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| concurrent | panel\|delay_h5 | [0, 0, 0, 0, 0] | 0 | t_p | +0.0122 | [+0.0058, +0.0191] | +0.0111 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| concurrent | panel\|deriv | [0, 0, 0, 0, 0] | 0 | t_p | +0.0150 | [+0.0090, +0.0214] | +0.0159 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| concurrent | resid1\|delay_h1 | [0, 0, 0, 0, 0] | 0 | t_p | +0.0135 | [+0.0080, +0.0193] | +0.0130 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| concurrent | resid1\|delay_h2 | [0, 0, 0, 0, 0] | 0 | t_p | +0.0151 | [+0.0095, +0.0212] | +0.0166 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| concurrent | resid1\|delay_h3 | [0, 0, 0, 0, 0] | 0 | t_p | +0.0144 | [+0.0084, +0.0208] | +0.0148 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| concurrent | resid1\|delay_h5 | [0, 0, 0, 0, 0] | 0 | t_p | +0.0105 | [+0.0040, +0.0172] | +0.0078 | 0.0476/0.0476 | True | True | True | **False** | positive_recoverability |
| concurrent | resid1\|deriv | [0, 0, 0, 0, 0] | 0 | t_p | +0.0136 | [+0.0079, +0.0196] | +0.0130 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| concurrent | gmode3\|delay_h3 | [0, 0, 0, 0, 0] | 0 | t_p | +0.0116 | [+0.0086, +0.0146] | +0.0084 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| concurrent | gmode3\|delay_h5 | [0, 0, 0, 0, 0] | 0 | t_p | +0.0117 | [+0.0087, +0.0147] | +0.0083 | 0.0476/0.0476 | True | True | True | True | positive_recoverability |
| past_only | panel\|raw | [4, 4, 4, 1, 2] | +4 | t_p-4 | -0.0028 | [-0.0056, -0.0003] | -0.0190 | 0.4286/0.0476 | False | False | False | False | unresolved_no_signal |
| past_only | panel\|delay_h1 | [3, 3, 4, 1, 1] | +1 | t_p-1 | -0.0042 | [-0.0076, -0.0012] | -0.0132 | 0.0476/0.0476 | True | True | False | False | null_relative_difference_detected |
| past_only | panel\|delay_h2 | [2, 1, 1, 1, 1] | +1 | t_p-1 | -0.0018 | [-0.0044, +0.0010] | -0.0144 | 0.0476/0.0476 | True | True | False | False | null_relative_difference_detected |
| past_only | panel\|delay_h3 | [1, 1, 1, 1, 1] | +1 | t_p-1 | -0.0015 | [-0.0043, +0.0013] | -0.0168 | 0.0476/0.0476 | True | True | False | False | null_relative_difference_detected |
| past_only | panel\|delay_h5 | [1, 1, 1, 1, 1] | +1 | t_p-1 | -0.0048 | [-0.0078, -0.0018] | -0.0227 | 0.0476/0.0476 | True | True | False | False | null_relative_difference_detected |
| past_only | panel\|deriv | [3, 3, 4, 1, 1] | +1 | t_p-1 | -0.0044 | [-0.0077, -0.0015] | -0.0132 | 0.0476/0.0476 | True | True | False | False | null_relative_difference_detected |
| past_only | resid1\|delay_h1 | [3, 3, 4, 1, 4] | +3 | t_p-3 | -0.0045 | [-0.0071, -0.0020] | -0.0222 | 0.381/0.0476 | False | False | False | False | unresolved_no_signal |
| past_only | resid1\|delay_h2 | [2, 1, 4, 1, 1] | +1 | t_p-1 | -0.0039 | [-0.0066, -0.0010] | -0.0179 | 0.0476/0.0476 | True | True | False | False | null_relative_difference_detected |
| past_only | resid1\|delay_h3 | [1, 1, 1, 1, 1] | +1 | t_p-1 | -0.0028 | [-0.0055, -0.0000] | -0.0205 | 0.0476/0.0476 | True | True | False | False | null_relative_difference_detected |
| past_only | resid1\|delay_h5 | [1, 1, 1, 1, 2] | +1 | t_p-1 | -0.0068 | [-0.0095, -0.0038] | -0.0260 | 0.0476/0.0476 | True | True | False | False | null_relative_difference_detected |
| past_only | resid1\|deriv | [3, 3, 4, 1, 4] | +3 | t_p-3 | -0.0047 | [-0.0073, -0.0022] | -0.0222 | 0.6667/0.0476 | False | False | False | False | unresolved_no_signal |
| past_only | gmode3\|delay_h3 | [1, 1, 1, 1, 2] | +1 | t_p-1 | **+0.0025** | [+0.0007, +0.0043] | -0.0094 | 0.0476/0.0476 | True | True | False | False | null_relative_difference_detected |
| past_only | gmode3\|delay_h5 | [1, 1, 1, 1, 2] | +1 | t_p-1 | **+0.0024** | [+0.0004, +0.0045] | -0.0099 | 0.0476/0.0476 | True | True | False | False | null_relative_difference_detected |

**Label counts (13 representations per grid):**

| grid | positive_recoverability | null_relative_difference_detected | unresolved_no_signal |
|---|---|---|---|
| original | 12 | 0 | 1 |
| concurrent | 12 | 0 | 1 |
| past_only | **0** | 10 | 3 |

**Notes on the past-only rows:**

- **"Above null" with a negative increment.** Many past-only branches are "above null"
  even though their increment is negative. Phase-randomised extra features hurt the
  decoder even more (for example, `panel|delay_h5` has linear null p95 −0.0105 against an
  observed −0.0048). This is the sign-blindness that the software-v1 semantics were built
  to catch. `obs>0` and `ci_lower>0` fail, so these branches land in
  `null_relative_difference_detected`, not in a positive claim.
- **The only positive past-only increments** are the global-mode branches
  `gmode3|delay_h3` and `gmode3|delay_h5`, at +0.0025 and +0.0024. Their lower CI bounds
  are above 0, but the effects sit **below the 0.0045 practical floor** (the guard fails),
  and the nonlinear confirmer increment is negative.
- **The concurrent `resid1|delay_h5` branch** passes the decision policy, which compares
  the incremental to the floor. Its CI lower bound (0.0040) does not clear the floor, so
  it would fail the stricter compression-curve CI guard.

**Descriptive per-lag profile.** This is fixed-alpha ridge `_fast_incr` on the global
transform with the same outer worm splits. It is descriptive, not the nested estimate.

| representation | −5 | −4 | −3 | −2 | −1 | 0 | +1 | +2 | +3 | +4 | +5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| panel\|raw | +.0028 | +.0036 | +.0039 | +.0060 | +.0042 | −.0002 | −.0004 | −.0014 | −.0016 | −.0021 | −.0018 |
| panel\|delay_h1 | +.0014 | +.0043 | +.0087 | +.0152 | **+.0414** | +.0151 | +.0004 | −.0029 | −.0029 | −.0046 | −.0038 |
| panel\|delay_h2 | +.0023 | +.0086 | +.0167 | **+.0475** | **+.0477** | +.0166 | −.0004 | −.0042 | −.0050 | −.0063 | −.0061 |
| panel\|delay_h3 | +.0071 | +.0156 | **+.0468** | **+.0529** | **+.0489** | +.0155 | −.0019 | −.0065 | −.0066 | −.0083 | −.0083 |
| panel\|delay_h5 | **+.0418** | **+.0495** | **+.0521** | **+.0524** | **+.0462** | +.0113 | −.0053 | −.0093 | −.0102 | −.0125 | −.0119 |
| panel\|deriv | +.0014 | +.0043 | +.0086 | +.0152 | **+.0413** | +.0150 | +.0004 | −.0029 | −.0029 | −.0046 | −.0038 |
| resid1\|delay_h5 | +.0376 | +.0462 | +.0489 | +.0484 | +.0426 | +.0097 | −.0066 | −.0106 | −.0110 | −.0126 | −.0117 |
| resid1\|deriv | +.0007 | +.0030 | +.0062 | +.0095 | +.0371 | +.0135 | −.0007 | −.0037 | −.0036 | −.0050 | −.0037 |

**Reading the profile.** A large gain (≈ +0.04–0.05) appears exactly when the neural
window contains **both the target frame `tp` and the next frame `tp+1`**:

- `delay_h1` and `deriv` gain only at lag −1.
- `delay_h5` gains at every lag from −5 to −1.

A window that ends at `tp` (lag 0) gains ≈ +0.01–0.017. A window that ends before `tp`
(lag ≥ 1) gains ≤ 0. A single frame (`raw`) carries at most +0.006 at any lag. So the
recoverable information is essentially the **neural temporal change around and just
after the target frame**.

### Temporal-history sufficiency (frozen `compression_curve`, unmodified)

| grid | h=0 | h=1 | h=2 | h=3 | h=5 | full − baseline | min sufficient history | estimable |
|---|---|---|---|---|---|---|---|---|
| original | −0.065 | 0.837 | **0.904** | 1.000 | 0.985 | 0.0521 | **2** | yes |
| concurrent | 0.021 | 0.884 | **1.000** | 0.945 | 0.706 | 0.0163 | **2** | yes |
| past_only | −199 | −124 | 1.000 | −59 | −219 | −0.0004 | none | **NOT ESTIMABLE** |

The values under each h are retention values. The past-only row fails the guard with the
reason "full-space incremental signal below the declared floor
(S_full-S_baseline=-0.0004 < 0.0045)". Its retention values are meaningless, because the
denominator is about 0.

The degeneracy classes, pairwise relations and destructive tests were **not** recomputed
under the new grids. The frozen versions were computed at the original, behaviour-leading
modal lags and describe that pairing.

## 4. Superseding interpretation

**What survives (unchanged numbers, corrected meaning).** The frozen worm_v1 result is a
real, reproducible, grouped-by-worm neural gain over the AR(3) speed baseline. It is:

- up to +0.053 R² when the neural window may extend 1–3 frames (0.6–1.8 s) *after* the
  target frame, and
- **+0.010 to +0.017 R² (all 12 top branches `positive_recoverability`)** when the window
  ends *at* the target frame.

The correct description is:

> **concurrent / behaviour-leading neural recoverability of current within-forward
> speed**. Neural activity (mainly its frame-to-frame change) at and just after frame
> `tp` carries information about speed at `tp` beyond speed's own recent past.

Within the concurrent grid, temporal-history sufficiency still holds: min history = 2
frames, guard admitted, full − baseline 0.0163. The 7-member degeneracy class and the
destructive-test results stay valid as descriptions of the original (behaviour-leading)
pairing.

**What does not survive.** The claim that neural activity **predicts future speed**
("~1–2 s ahead") is **not supported**, because:

- The original negative modal lags mean the neural data came *after* the behaviour, not
  before.
- Under the strictly past-only grid (neural window ends 1–5 frames = 0.6–3.0 s before the
  target), none of the 13 representations gives a positive predictive gain:
  - 11/13 have a negative point estimate. Examples: `panel|delay_h5` −0.0048
    [−0.0078, −0.0018]; `panel|deriv` −0.0044 [−0.0077, −0.0015].
  - The only positive estimates (`gmode3|delay_h3/h5`, +0.0025/+0.0024) are below the
    practical floor, and their nonlinear confirmers are negative.
- Past-only temporal-history sufficiency is not estimable.

The behaviour-leading timing fits neural signals that *follow or integrate* behaviour.
Candidate explanations include calcium-indicator kinetics, encoding of recent behaviour
(as in the dataset authors' behaviour-encoding models), or smoothing in how the velocity
channel was derived. **This audit does not separate these.**

**Status: OPEN, not closed.** The null applies only to this operationalization:

- target: within-forward speed;
- baseline: an AR(3) speed baseline, which is already strong (R² ≈ 0.50);
- lags: 1–5 frames;
- this 13-representation subset, with ridge + poly-ridge decoders.

It says nothing about the other 123 grammar branches under a past-only grid (for example
window-10 families such as `swavg_w10`, `cov_*_w10`, `spec_w10`), about longer histories,
or about other targets.

Strongest next tests:

1. Re-run the full 136-branch grammar with `LAG_GRID = 1..5` (≈ 1–1.5 h with 3 processes,
   using the same script pattern).
2. Change the target to future *speed change* (`tgt[tp] − tgt[t]`) or to event targets
   such as reversal or turn onset, where a past-only neural lead is physiologically
   plausible.
3. Compare against a deconvolved or indicator-kinetics-corrected neural signal, to test
   whether the lag is an indicator artifact.

## 5. Other flags (not acted on)

- **Inverted comment.** `evaluate_unit_local.py:58` says "L<0 neural-leading", but the
  code uses `tp = idx + lag`, so L<0 is behaviour-leading. The file is frozen and was not
  edited.
- **Fish runs not checked.** The fish unit-local runs use the same symmetric grid. Whether
  any fish selected lag includes post-target neural frames was **not** audited here.

## 6. Preservation of frozen artefacts

Nothing frozen was modified:

- **Hash snapshots:** sha256 hashes of all 302 files under `src/wormlab/sufficiency/`,
  `results/empirical_sufficiency_mvp/`, `results/empirical_sufficiency_mvp_v2detector/`
  and `results/_capsules/` were taken before and after the runs. The two snapshots are
  byte-identical (`integrity_before.sha256` = `integrity_after.sha256` in the audit
  directory).
- **Manifest:** `MANIFEST.sha256` (root `34621f50…`) still verifies 32/32.
- **Capsule root:** the worm_v1 capsule root `3fcdf1694195606bb16eabbdf3377ff129fa0d01788e90f1b75e4f97c4438c29`
  is unchanged.
- **Where the new numbers live:** this audit supersedes the *interpretation* only. The
  frozen numbers stay the record of the original symmetric-grid run. All new numbers are
  in `results/_audits/worm_past_only_2026-09-16/`.
