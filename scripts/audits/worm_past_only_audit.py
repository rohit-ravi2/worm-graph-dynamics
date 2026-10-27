"""Strictly past-only re-audit of the Atanas within-forward-speed recoverability MVP.

Issue: evaluate._pair_lag predicts target[t+lag] from the representation at frame t,
and delay/deriv representations span frames t-h..t. The frozen symmetric grid
LAG_GRID=range(-5,6) therefore let the headline `panel|delay_h5` (modal lag -2) use
neural frames up to 2 frames AFTER the predicted frame, while the AR(3) baseline saw
only earlier target frames.

This script re-runs the SAME frozen pipeline (evaluate.evaluate_representation, v2
detector, null_n=20, seed=0, grouped 5-fold worm splits) under three lag grids by
MONKEYPATCHING `evaluate.LAG_GRID` in-process. No file under src/wormlab/sufficiency
or results/empirical_sufficiency_mvp* is modified; everything is written to
results/_audits/worm_past_only_2026-09-16/.

  original   : range(-5, 6)   (reference; must reproduce the frozen v2 numbers)
  concurrent : range(0, 6)    (neural window may END AT the target frame)
  past_only  : range(1, 6)    (neural window ends strictly BEFORE the target frame)

Usage:
  python scripts/audits/worm_past_only_audit.py smoke            # one rep x past_only, timed
  python scripts/audits/worm_past_only_audit.py run [--workers 3] [--reps a,b]
  python scripts/audits/worm_past_only_audit.py aggregate
"""
from __future__ import annotations

import os

# single-threaded BLAS per process (shared machine; <=3 CPU processes total)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import argparse
import json
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "results/_audits/worm_past_only_2026-09-16")
RUNS = os.path.join(OUT, "runs")
V2_FROZEN = os.path.join(REPO, "results/empirical_sufficiency_mvp_v2detector/representation_results.json")
STABLE_FROZEN = os.path.join(REPO, "results/empirical_sufficiency_mvp/representation_results.json")
WORM_POLICY_SPEC = os.path.join(REPO, "specs/worm_positive_temporal.json")

GRIDS = {
    "original": list(range(-5, 6)),
    "concurrent": list(range(0, 6)),
    "past_only": list(range(1, 6)),
}
REPS = ["panel|raw", "panel|delay_h1", "panel|delay_h2", "panel|delay_h3", "panel|delay_h5",
        "panel|deriv", "resid1|delay_h2", "resid1|delay_h3", "resid1|delay_h5", "resid1|deriv",
        # remaining members of the frozen 12-branch recoverability_top set
        "resid1|delay_h1", "gmode3|delay_h3", "gmode3|delay_h5"]
# frozen run parameters (run_mvp.run_v2_regression / original stable run)
V2_NULL_N = 20
STABLE_NULL_N = 10
SEED = 0

_DS = None
_SPLITS = None


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _init():
    global _DS, _SPLITS
    if _DS is None:
        from wormlab.sufficiency.data import load_dataset
        from wormlab.sufficiency.evaluate import OUTER_K, make_outer_splits
        _DS = load_dataset()
        _SPLITS = make_outer_splits(_DS["worm_ids"], k=OUTER_K, seed=SEED)


def _spec(space_id):
    from wormlab.sufficiency.transforms import enumerate_grammar
    return next(s for s in enumerate_grammar() if s["space_id"] == space_id)


def run_job(job):
    """Evaluate one (grid, representation) through the frozen pipeline."""
    grid_name, sid = job
    path = os.path.join(RUNS, f"{grid_name}__{sid.replace('|', '__')}.json")
    if os.path.exists(path):
        return path, "cached", 0.0
    _init()
    import wormlab.sufficiency.evaluate as E
    from wormlab.sufficiency.detector import make_detector
    from wormlab.sufficiency.evaluate import _pair_lag
    from wormlab.sufficiency.transforms import build_representation

    grid = GRIDS[grid_name]
    E.LAG_GRID = list(grid)            # monkeypatch (module global read at call time)
    spec = _spec(sid)
    t0 = time.time()
    r = E.evaluate_representation(spec, _DS, splits=_SPLITS, run_nonlinear=True,
                                  null_n=V2_NULL_N, seed=SEED, detector_backend="v2")
    t_v2 = time.time() - t0
    d = r.to_dict()
    assert all(l in grid for l in d["selected_lag_frames"]), "selected lag outside patched grid"
    assert d["selected_lag_mode"] in grid

    # companion: the ORIGINAL stable (linear-only p95) null at the modal lag, exactly
    # as evaluate_representation's headline block does for backend='stable'.
    t1 = time.time()
    wids_all = _DS["worm_ids"]
    grep = build_representation(spec, _DS, wids_all)
    Fg = {w: grep.transform(_DS["worms"][w]["neural"]) for w in wids_all}
    nw, bw, yw = _pair_lag(_DS, Fg, d["selected_lag_mode"])
    wids = [w for w in wids_all if w in nw]
    det_s = make_detector("linear", STABLE_NULL_N, backend="stable").detect(
        nw, bw, yw, wids, list(_SPLITS), run_null=True, seed=SEED)
    n_pairs = int(sum(len(yw[w]) for w in wids))

    # descriptive per-lag profile (fast fixed-alpha ridge, global transform, outer
    # worm splits) over the FULL -5..5 range: where the neural increment lives.
    profile = {}
    for lag in range(-5, 6):
        a, b, c = _pair_lag(_DS, Fg, lag)
        wl = [w for w in wids_all if w in a]
        profile[lag] = {"fast_incremental": round(float(E._fast_incr(a, b, c, wl, list(_SPLITS))), 5),
                        "n_pairs": int(sum(len(c[w]) for w in wl)), "n_worms": len(wl)}
    t_extra = time.time() - t1

    out = {
        "grid_name": grid_name, "lag_grid": grid, "space_id": sid,
        "detector_backend": "v2", "null_n_v2": V2_NULL_N, "seed": SEED,
        "result_v2": d,
        "stable_companion_at_modal_lag": {
            "null_n": STABLE_NULL_N, "incremental": det_s["incremental"],
            "incremental_ci": det_s["incremental_ci"], "null_comparison": det_s["null_comparison"],
            "n_pairs": n_pairs, "n_worms": len(wids),
        },
        "descriptive_lag_profile_fast_ridge": profile,
        "timing_s": {"evaluate_representation_v2": round(t_v2, 1), "companion_and_profile": round(t_extra, 1)},
        "support_semantics": ("feature at frame t spans frames t-h..t (delay_h, deriv h=1, raw h=0); "
                              "target frame tp=t+lag; AR(3) baseline uses tgt[tp-1..tp-3]."),
    }
    os.makedirs(RUNS, exist_ok=True)
    _dump(path, out)
    return path, "done", round(t_v2 + t_extra, 1)


# --------------------------------------------------------------------------- aggregate
def _policy():
    from wormlab.decisions.policy_spec import DecisionPolicy
    with open(WORM_POLICY_SPEC) as f:
        return DecisionPolicy.from_dict(json.load(f)["decision_policy"])


def _classify(sid, incr, ci, null_sig, policy):
    """Mirror decisions.applier._apply_worm for one branch (no recomputation)."""
    from wormlab.decisions.recoverability import BranchNumbers, classify_branch
    ci = ci or [None, None]
    bn = BranchNumbers(branch=sid, obs=incr, ci_lower=ci[0], ci_upper=ci[1],
                       null_significant=bool(null_sig),
                       cross_unit_consistent=bool(null_sig) and (ci[0] is not None and ci[0] > 0.0),
                       practical_effect=incr, guard_admitted=None, status="EVALUATED")
    dec = classify_branch(bn, policy)
    return dec.label, [c["component"] for c in dec.failed_components]


def aggregate():
    from wormlab.sufficiency.analysis import SIGNAL_FLOOR, compression_curve
    from wormlab.sufficiency.run_mvp import _fold_stable
    policy = _policy()
    frozen_v2 = {r["space_id"]: r for r in json.load(open(V2_FROZEN))["results"]}
    frozen_st = {r["space_id"]: r for r in json.load(open(STABLE_FROZEN))["results"]}

    rows, by = [], {}
    for g in GRIDS:
        for sid in REPS:
            p = os.path.join(RUNS, f"{g}__{sid.replace('|', '__')}.json")
            if not os.path.exists(p):
                continue
            j = json.load(open(p))
            d = j["result_v2"]; nc = d["null_comparison"] or {}
            st = j["stable_companion_at_modal_lag"]; stn = st["null_comparison"] or {}
            label_v2, failed_v2 = _classify(sid, d["incremental"], d["incremental_ci"],
                                            nc.get("above_null_p95"), policy)
            label_st, failed_st = _classify(sid, d["incremental"], d["incremental_ci"],
                                            stn.get("above_null_p95"), policy)
            ci = d["incremental_ci"]
            row = {
                "grid": g, "space_id": sid, "selected_lag_frames": d["selected_lag_frames"],
                "selected_lag_mode": d["selected_lag_mode"], "score": d["score"],
                "baseline_at_modal_lag": d["baseline"], "incremental": d["incremental"],
                "incremental_ci90": ci, "nonlinear_incremental": d["nonlinear_incremental"],
                "access_tag": d["access_tag"],
                "v2_p_linear": nc.get("p_linear"), "v2_p_nonlinear": nc.get("p_nonlinear"),
                "v2_linear_null_p95": nc.get("linear_null_p95"),
                "v2_dual_above_null": nc.get("above_null_p95"),
                "stable_null_p95": stn.get("null_incremental_p95"),
                "stable_above_null": stn.get("above_null_p95"),
                "guard_incr_gt_floor": bool(d["incremental"] is not None and d["incremental"] > SIGNAL_FLOOR),
                "guard_ci_lower_gt_floor": bool(ci and ci[0] is not None and ci[0] > SIGNAL_FLOOR),
                "decision_label_v2_null": label_v2, "failed_components_v2_null": failed_v2,
                "decision_label_stable_null": label_st, "failed_components_stable_null": failed_st,
                "positive_predictive_gain": label_v2 == "positive_recoverability",
                "n_pairs_at_modal_lag": st["n_pairs"], "n_worms": st["n_worms"],
                "max_neural_frame_relative_to_target": -d["selected_lag_mode"],
                "timing_s": j["timing_s"],
            }
            if g == "original":
                fv, fs = frozen_v2.get(sid, {}), frozen_st.get(sid, {})
                fvn = fv.get("null_comparison") or {}
                row["reproduces_frozen_v2"] = {
                    "selected_lag_frames": fv.get("selected_lag_frames") == d["selected_lag_frames"],
                    "incremental": fv.get("incremental") == d["incremental"],
                    "incremental_ci": fv.get("incremental_ci") == d["incremental_ci"],
                    "score": fv.get("score") == d["score"],
                    "nonlinear_incremental": fv.get("nonlinear_incremental") == d["nonlinear_incremental"],
                    "p_linear": fvn.get("p_linear") == nc.get("p_linear"),
                    "p_nonlinear": fvn.get("p_nonlinear") == nc.get("p_nonlinear"),
                    "stable_null_p95_matches_original_stable_run":
                        (fs.get("null_comparison") or {}).get("null_incremental_p95") == stn.get("null_incremental_p95"),
                }
            rows.append(row)
            by[(g, sid)] = (j, row)

    # temporal-history sufficiency via the frozen compression_curve (unmodified)
    curves = {}
    hist_ids = [(0, "panel|raw"), (1, "panel|delay_h1"), (2, "panel|delay_h2"),
                (3, "panel|delay_h3"), (5, "panel|delay_h5")]
    for g in GRIDS:
        pts = []
        for h, sid in hist_ids:
            if (g, sid) not in by:
                continue
            d = by[(g, sid)][0]["result_v2"]
            pts.append({"axis_value": h, "score": d["score"], "baseline": d["baseline"],
                        "incremental_ci": d["incremental_ci"],
                        "stable_across_folds": _fold_stable(SimpleNamespace(**d))})
        if len(pts) == len(hist_ids):
            curves[g] = compression_curve(pts, "temporal_history")
        else:
            curves[g] = {"not_computed": f"only {len(pts)}/{len(hist_ids)} history points available"}

    summary = {
        "audit": "worm past-only lag re-audit 2026-09-16",
        "frozen_sources_untouched": [V2_FROZEN, STABLE_FROZEN,
                                     "results/_capsules/worm_v1 (root 3fcdf169...)"],
        "grids": GRIDS, "representations": REPS,
        "pipeline": ("wormlab.sufficiency.evaluate.evaluate_representation, detector_backend='v2', "
                     f"null_n={V2_NULL_N}, seed={SEED}, make_outer_splits(k=5, seed=0); "
                     "LAG_GRID monkeypatched in-process; stable-detector companion null "
                     f"(null_n={STABLE_NULL_N}) at the modal lag"),
        "signal_floor": SIGNAL_FLOOR,
        "decision_policy_source": WORM_POLICY_SPEC,
        "rows": rows,
        "temporal_history_compression": curves,
    }
    _dump(os.path.join(OUT, "past_only_audit_results.json"), summary)
    for r in rows:
        print(f"{r['grid']:10s} {r['space_id']:16s} lags={r['selected_lag_frames']} mode={r['selected_lag_mode']:+d} "
              f"incr={r['incremental']} ci={r['incremental_ci90']} v2dual={r['v2_dual_above_null']} "
              f"stable={r['stable_above_null']} -> {r['decision_label_v2_null']}")
    for g, c in curves.items():
        print(g, c.get("min_sufficient_axis_value"), c.get("sufficiency_estimable"),
              [(p["axis_value"], p["retention"]) for p in c.get("curve", [])])
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "run", "aggregate"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--reps", default=None)
    ap.add_argument("--grids", default=None)
    a = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    if a.mode == "smoke":
        t0 = time.time()
        _init()
        t_load = time.time() - t0
        path, status, t = run_job(("past_only", "panel|delay_h5"))
        print(json.dumps({"load_s": round(t_load, 1), "job_s": t, "path": path, "status": status}))
        return
    if a.mode == "aggregate":
        aggregate()
        return
    reps = a.reps.split(",") if a.reps else REPS
    grids = a.grids.split(",") if a.grids else list(GRIDS)
    jobs = [(g, s) for g in grids for s in reps]
    workers = max(1, min(3, a.workers))
    log = open(os.path.join(OUT, "run_log.txt"), "a")
    t0 = time.time()
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers) as pool:
        for path, status, t in pool.imap_unordered(run_job, jobs):
            msg = f"[{time.time() - t0:8.1f}s] {status} {t}s {os.path.basename(path)}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
    log.write(f"TOTAL wall {time.time() - t0:.1f}s for {len(jobs)} jobs\n"); log.close()
    aggregate()


if __name__ == "__main__":
    sys.exit(main())
