"""Mechanism contrast library over event tables (``observations.build_table``).

Every contrast returns a raw result dict: estimate, animal-cluster bootstrap CI, units, observations,
hostile-null summary, coverage diagnostics. Turning that into a ``TestOutcome`` (call vs floor, status)
is done by ``to_outcome`` using the frozen ``MechanismTestSpec``.

Independence unit is the animal. Pairs within an animal are never treated as independent: CIs resample
animals, and matched risk differences are computed within animal x stratum cells and then pooled.
"""
from __future__ import annotations

import numpy as np

from .measurement import delay_call
from .spec import Call, MechanismTestSpec, OutcomeStatus, TestOutcome, call_from_ci

N_BOOT = 400
N_PERM = 200


def _sel(tab: dict, m: np.ndarray) -> dict:
    return {k: v[m] for k, v in tab.items()}


def _animals(tab):
    return np.unique(tab["animal"]) if len(tab.get("animal", [])) else np.array([])


def _boot(tab: dict, stat, *, seed: int, n: int = N_BOOT):
    A = _animals(tab)
    if len(A) < 2:
        return np.nan, [np.nan, np.nan]
    groups = {a: np.where(tab["animal"] == a)[0] for a in A}
    est = stat(tab)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(A, len(A))
        idx = np.concatenate([groups[a] for a in pick])
        v = stat(_sel(tab, idx))
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < n // 2:
        return est, [np.nan, np.nan]
    return est, [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _mh_rd(y: np.ndarray, x: np.ndarray, strata: np.ndarray) -> float:
    """Mantel-Haenszel risk difference of y between x=1 and x=0 across strata."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        a, b = y[m & x], y[m & ~x]
        if len(a) == 0 or len(b) == 0:
            continue
        w = len(a) * len(b) / (len(a) + len(b))
        num += w * (a.mean() - b.mean())
        den += w
    return num / den if den > 0 else np.nan


def _strata(tab, cols_bins):
    key = tab["animal"].astype(str)
    for col, bins in cols_bins:
        key = np.char.add(np.char.add(key, "|"), np.digitize(tab[col], bins).astype(str))
    return key


def _coverage(tab, x) -> dict:
    return dict(n_obs=int(len(x)), n_pos=int(x.sum()), n_animals=int(len(_animals(tab))),
                n_animals_with_both=int(sum(1 for a in _animals(tab) if x[tab["animal"] == a].any() and (~x[tab["animal"] == a]).any())))


# ----------------------------------------------------------------------------------------------------
# T0 proximity effect (the phenomenon itself)
# ----------------------------------------------------------------------------------------------------
def proximity_effect(tab, *, near_um=(0, 6), far_um=(15, 40), seed=0):
    m = (tab["kind"] == "stim") & tab["target_resp"]
    t = _sel(tab, m)
    near = (t["dist_um"] >= near_um[0]) & (t["dist_um"] < near_um[1])
    far = (t["dist_um"] >= far_um[0]) & (t["dist_um"] < far_um[1])
    u = _sel(t, near | far)
    x = ((u["dist_um"] >= near_um[0]) & (u["dist_um"] < near_um[1]))
    stat = lambda s: _mh_rd(s["resp"], (s["dist_um"] < near_um[1]), s["animal"].astype(str))
    est, ci = _boot(u, stat, seed=seed)
    return dict(estimate=est, ci=ci, coverage=_coverage(u, x), near_rate=float(u["resp"][x].mean()) if x.any() else np.nan,
                far_rate=float(u["resp"][~x].mean()) if (~x).any() else np.nan)


# ----------------------------------------------------------------------------------------------------
# T1 distance-matched topology  /  T7 path length
# ----------------------------------------------------------------------------------------------------
def distance_matched_topology(tab, *, topo_col="edge", max_um=15.0, bin_um=2.0, seed=0):
    m = (tab["kind"] == "stim") & tab["labeled"] & (tab["dist_um"] <= max_um)
    t = _sel(tab, m)
    if topo_col == "path_short":
        x_all = (t["path_len"] >= 1) & (t["path_len"] <= 2)
    else:
        x_all = t[topo_col].astype(bool)
    t["_x"] = x_all
    bins = np.arange(bin_um, max_um + bin_um, bin_um)
    strata = _strata(t, [("dist_um", bins)])
    t["_s"] = strata
    stat = lambda s: _mh_rd(s["resp"], s["_x"], s["_s"])
    est, ci = _boot(t, stat, seed=seed)
    # hostile null: permute topology labels within animal x distance-bin strata
    rng = np.random.default_rng(seed + 1)
    null = []
    for _ in range(N_PERM):
        xp = x_all.copy()
        for s in np.unique(strata):
            ii = np.where(strata == s)[0]
            xp[ii] = xp[rng.permutation(ii)]
        null.append(_mh_rd(t["resp"], xp, strata))
    null = np.array([v for v in null if np.isfinite(v)])
    p = float((np.sum(np.abs(null) >= abs(est)) + 1) / (len(null) + 1)) if np.isfinite(est) and len(null) else np.nan
    naive = float(t["resp"][x_all].mean() - t["resp"][~x_all].mean()) if x_all.any() and (~x_all).any() else np.nan
    return dict(estimate=est, ci=ci, coverage=_coverage(t, x_all), perm_p=p, naive_unmatched_rd=naive,
                null_sd=float(null.std()) if len(null) else np.nan)


# ----------------------------------------------------------------------------------------------------
# T2 topology-matched distance
# ----------------------------------------------------------------------------------------------------
def topology_matched_distance(tab, *, near_um=6.0, far_um=(10.0, 15.0), seed=0):
    m = (tab["kind"] == "stim") & tab["labeled"] & ((tab["dist_um"] < near_um) | ((tab["dist_um"] >= far_um[0]) & (tab["dist_um"] < far_um[1])))
    t = _sel(tab, m)
    x = t["dist_um"] < near_um
    t["_x"] = x
    topo_class = np.where(t["chem"] | t["gap"], "edge", np.where(t["two_hop"], "twohop", "none"))
    t["_s"] = np.char.add(t["animal"].astype(str), topo_class)
    stat = lambda s: _mh_rd(s["resp"], s["_x"], s["_s"])
    est, ci = _boot(t, stat, seed=seed)
    return dict(estimate=est, ci=ci, coverage=_coverage(t, x))


# ----------------------------------------------------------------------------------------------------
# T3 onset latency (scan-time corrected intervals)
# ----------------------------------------------------------------------------------------------------
def onset_latency(tab, *, max_um=8.0, seed=0):
    m = ((tab["kind"] == "stim") & tab["resp"] & tab["target_resp"] & (tab["dist_um"] <= max_um)
         & np.isfinite(tab["rt_lo"]) & np.isfinite(tab["tt_lo"]))
    t = _sel(tab, m)
    lo = t["rt_lo"] - t["tt_hi"]
    hi = t["rt_hi"] - t["tt_lo"]
    agg = delay_call(list(zip(lo, hi)))
    t["_mid"] = (lo + hi) / 2
    est, ci = _boot(t, lambda s: float(np.median(s["_mid"])) if len(s["_mid"]) else np.nan, seed=seed)
    return dict(estimate=est, ci=ci, coverage=dict(n_obs=int(m.sum()), n_animals=int(len(_animals(t)))), **agg)


# ----------------------------------------------------------------------------------------------------
# T4 stimulation geometry: spot-offset directionality
# ----------------------------------------------------------------------------------------------------
def offset_directionality(tab, *, max_um=8.0, min_offset_px=1.0, seed=0):
    m = (tab["kind"] == "stim") & (tab["dist_um"] <= max_um) & np.isfinite(tab["cos_offset"]) & (tab["offset_px"] >= min_offset_px)
    t = _sel(tab, m & ((tab["cos_offset"] > 0.5) | (tab["cos_offset"] < -0.5)))
    x = t["cos_offset"] > 0.5
    t["_x"] = x
    bins = np.arange(2.0, max_um + 2.0, 2.0)
    t["_s"] = _strata(t, [("dist_um", bins)])
    est, ci = _boot(t, lambda s: _mh_rd(s["resp"], s["_x"], s["_s"]), seed=seed)
    return dict(estimate=est, ci=ci, coverage=_coverage(t, x))


# ----------------------------------------------------------------------------------------------------
# T5 control channel (red) near-field
# ----------------------------------------------------------------------------------------------------
def control_channel(tab, *, near_um=(0, 6), far_um=(15, 40), seed=0):
    m = (tab["kind"] == "stim") & tab["target_resp"] & np.isfinite(tab["red_z"])
    t = _sel(tab, m & (((tab["dist_um"] >= near_um[0]) & (tab["dist_um"] < near_um[1])) | ((tab["dist_um"] >= far_um[0]) & (tab["dist_um"] < far_um[1]))))
    t["_rr"] = np.abs(t["red_z"]) > 3
    x = t["dist_um"] < near_um[1]
    est, ci = _boot(t, lambda s: _mh_rd(s["_rr"], s["dist_um"] < near_um[1], s["animal"].astype(str)), seed=seed)
    return dict(estimate=est, ci=ci, coverage=_coverage(t, x))


# ----------------------------------------------------------------------------------------------------
# T6 target mediation: near-field activation when the target did NOT respond
# ----------------------------------------------------------------------------------------------------
def target_flat_nearfield(tab, *, near_um=(0, 6), far_um=(15, 40), seed=0):
    m = (tab["kind"] == "stim") & (np.abs(tab["target_z"]) < 1.0)
    t = _sel(tab, m & (((tab["dist_um"] >= near_um[0]) & (tab["dist_um"] < near_um[1])) | ((tab["dist_um"] >= far_um[0]) & (tab["dist_um"] < far_um[1]))))
    x = t["dist_um"] < near_um[1]
    est, ci = _boot(t, lambda s: _mh_rd(s["resp"], s["dist_um"] < near_um[1], s["animal"].astype(str)), seed=seed)
    return dict(estimate=est, ci=ci, coverage=_coverage(t, x))


# ----------------------------------------------------------------------------------------------------
# T8 sham: near-field in no-stimulation windows, conditioned on pseudo-target spontaneous activity
# ----------------------------------------------------------------------------------------------------
def sham_nearfield(tab, *, near_um=(0, 6), far_um=(15, 40), seed=0, conditional=True):
    m = (tab["kind"] == "sham")
    if conditional:
        m &= tab["target_resp"]
    t = _sel(tab, m & (((tab["dist_um"] >= near_um[0]) & (tab["dist_um"] < near_um[1])) | ((tab["dist_um"] >= far_um[0]) & (tab["dist_um"] < far_um[1]))))
    x = t["dist_um"] < near_um[1] if len(t.get("dist_um", [])) else np.array([], bool)
    est, ci = _boot(t, lambda s: _mh_rd(s["resp"], s["dist_um"] < near_um[1], s["animal"].astype(str)), seed=seed) if len(x) else (np.nan, [np.nan, np.nan])
    return dict(estimate=est, ci=ci, coverage=_coverage(t, x) if len(x) else dict(n_obs=0, n_animals=0, n_animals_with_both=0))


# ----------------------------------------------------------------------------------------------------
# H7 breadth: far-field response rate on stim vs sham events
# ----------------------------------------------------------------------------------------------------
def farfield_breadth(tab, *, far_um=(15, 40), seed=0):
    m = ((tab["dist_um"] >= far_um[0]) & (tab["dist_um"] < far_um[1]))
    t = _sel(tab, m)
    x = t["kind"] == "stim"
    est, ci = _boot(t, lambda s: _mh_rd(s["resp"], s["kind"] == "stim", s["animal"].astype(str)), seed=seed)
    return dict(estimate=est, ci=ci, coverage=_coverage(t, x))


# ----------------------------------------------------------------------------------------------------
# T9 across-animal replication of the proximity effect
# ----------------------------------------------------------------------------------------------------
def animal_replication(tab, *, near_um=(0, 6), far_um=(15, 40), seed=0):
    m = (tab["kind"] == "stim") & tab["target_resp"]
    t = _sel(tab, m)
    per = []
    for a in _animals(t):
        s = t["animal"] == a
        near = s & (t["dist_um"] < near_um[1])
        far = s & (t["dist_um"] >= far_um[0]) & (t["dist_um"] < far_um[1])
        if near.sum() >= 3 and far.sum() >= 10:
            per.append(t["resp"][near].mean() - t["resp"][far].mean())
    per = np.array(per)
    if len(per) < 2:
        return dict(estimate=np.nan, ci=[np.nan, np.nan], coverage=dict(n_animals=int(len(per)), n_obs=int(m.sum())), per_animal=per.tolist())
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(per > 0, len(per))) for _ in range(N_BOOT)]
    return dict(estimate=float(np.mean(per > 0)) - 0.5, ci=[float(np.percentile(bs, 2.5)) - 0.5, float(np.percentile(bs, 97.5)) - 0.5],
                coverage=dict(n_animals=int(len(per)), n_obs=int(m.sum())), per_animal=per.tolist(),
                note="estimate = fraction of animals with positive near-far difference minus 0.5")


# ----------------------------------------------------------------------------------------------------
# T11 pre-stimulus covariance with the target (extraction / measurement-sharing signature)
# ----------------------------------------------------------------------------------------------------
def baseline_covariance(tab, *, near_um=(0, 6), far_um=(15, 40), seed=0):
    m = (tab["kind"] == "stim") & np.isfinite(tab["base_corr"]) & (((tab["dist_um"] >= near_um[0]) & (tab["dist_um"] < near_um[1])) | ((tab["dist_um"] >= far_um[0]) & (tab["dist_um"] < far_um[1])))
    t = _sel(tab, m)
    t["_fz"] = np.arctanh(np.clip(t["base_corr"], -0.999, 0.999))
    x = t["dist_um"] < near_um[1]

    def stat(s):
        diffs, ws = [], []
        for a in np.unique(s["animal"]):
            k = s["animal"] == a
            nr, fr = k & (s["dist_um"] < near_um[1]), k & (s["dist_um"] >= far_um[0])
            if nr.sum() and fr.sum():
                diffs.append(s["_fz"][nr].mean() - s["_fz"][fr].mean())
                ws.append(nr.sum() * fr.sum() / (nr.sum() + fr.sum()))
        return float(np.average(diffs, weights=ws)) if diffs else np.nan
    est, ci = _boot(t, stat, seed=seed)
    return dict(estimate=est, ci=ci, coverage=_coverage(t, x))


def baseline_covariance_plane_matched(tab, *, near_um=(0, 6), far_um=(15, 40), seed=0):
    """T11b: T11 with near and far receivers compared only within the same plane-difference class
    (|dplane| 0, 1-2, >=3), removing shared-frame co-sampling as an explanation by design."""
    m = (tab["kind"] == "stim") & np.isfinite(tab["base_corr"]) & (((tab["dist_um"] >= near_um[0]) & (tab["dist_um"] < near_um[1])) | ((tab["dist_um"] >= far_um[0]) & (tab["dist_um"] < far_um[1])))
    t = _sel(tab, m)
    t["_fz"] = np.arctanh(np.clip(t["base_corr"], -0.999, 0.999))
    dpl = np.rint(t["dz_um"] / 1.22).astype(int)
    t["_cls"] = np.where(dpl == 0, 0, np.where(dpl <= 2, 1, 2))
    x = t["dist_um"] < near_um[1]

    def stat(s):
        diffs, ws = [], []
        for a in np.unique(s["animal"]):
            for c in (0, 1, 2):
                k = (s["animal"] == a) & (s["_cls"] == c)
                nr, fr = k & (s["dist_um"] < near_um[1]), k & (s["dist_um"] >= far_um[0])
                if nr.sum() and fr.sum():
                    diffs.append(s["_fz"][nr].mean() - s["_fz"][fr].mean())
                    ws.append(nr.sum() * fr.sum() / (nr.sum() + fr.sum()))
        return float(np.average(diffs, weights=ws)) if diffs else np.nan
    est, ci = _boot(t, stat, seed=seed)
    return dict(estimate=est, ci=ci, coverage=_coverage(t, x))


def nucleus_vs_background_voxel(tab, *, shell=(3.5, 5.0), seed=0):
    """T12 (prospective X1): TARGET_UP events, pre-declared primary shell 3.5-5 um. Per animal: mean over events of
    (median nucleus-centre dF - median background dF), normalised by target dF; pooled weighted by events; animal bootstrap.
    TARGET_FLAT rows are reported as a control (not part of the call)."""
    if not len(tab.get("animal", [])):
        return dict(estimate=np.nan, ci=[np.nan, np.nan], coverage=dict(n_obs=0, n_animals=0, n_animals_with_both=0))
    m = (tab["cls"] == "TARGET_UP") & (tab["shell_lo"] == shell[0]) & np.isfinite(tab["difference"])
    t = _sel(tab, m)

    def stat(s):
        vals, ws = [], []
        for a in np.unique(s["animal"]):
            k = s["animal"] == a
            vals.append(s["difference"][k].mean()); ws.append(k.sum())
        return float(np.average(vals, weights=ws)) if vals else np.nan
    est, ci = _boot(t, stat, seed=seed)
    ctrl = _sel(tab, (tab["cls"] == "TARGET_FLAT") & (tab["shell_lo"] == shell[0]) & np.isfinite(tab["difference"]))
    by_shell = {}
    for lo in np.unique(tab["shell_lo"]):
        u = _sel(tab, (tab["cls"] == "TARGET_UP") & (tab["shell_lo"] == lo) & np.isfinite(tab["difference"]))
        by_shell[str(float(lo))] = dict(estimate=stat(u) if len(u["animal"]) else None, nucleus_median=float(np.median(u["nucleus_median"])) if len(u["animal"]) else None,
                                        background_median=float(np.median(u["background_median"])) if len(u["animal"]) else None)
    na = int(len(_animals(t)))
    return dict(estimate=est, ci=ci, coverage=dict(n_obs=int(m.sum()), n_animals=na, n_animals_with_both=na),
                target_flat_control=stat(ctrl) if len(ctrl["animal"]) else None, by_shell_target_up=by_shell)


REGISTRY = {
    "T0_proximity_effect": proximity_effect,
    "T1_distance_matched_topology": distance_matched_topology,
    "T2_topology_matched_distance": topology_matched_distance,
    "T3_onset_latency": onset_latency,
    "T4_offset_directionality": offset_directionality,
    "T5_control_channel": control_channel,
    "T6_target_flat_nearfield": target_flat_nearfield,
    "T7_path_length": lambda tab, seed=0: distance_matched_topology(tab, topo_col="path_short", seed=seed),
    "T8_sham_nearfield": sham_nearfield,
    "T9_animal_replication": animal_replication,
    "T10_farfield_breadth": farfield_breadth,
    "T11_baseline_covariance": baseline_covariance,
    "T11b_baseline_covariance_plane_matched": baseline_covariance_plane_matched,
    "T12_nucleus_vs_background_voxel": nucleus_vs_background_voxel,
}


def to_outcome(test: MechanismTestSpec, raw: dict, *, stage: str, partition: str, units: list[str], grade: str,
               resolution_s: float | None = None, derived_from: list[str] = ()) -> TestOutcome:
    cov = raw.get("coverage", {})
    n_units = int(cov.get("n_animals_with_both", cov.get("n_animals", 0)))
    ci = raw.get("ci")
    est = raw.get("estimate")
    status, call = OutcomeStatus.DECIDED, Call.INCONCLUSIVE
    floor = test.min_effect
    if n_units < test.min_units or ci is None or not np.all(np.isfinite(ci)):
        status = OutcomeStatus.UNDERPOWERED
    else:
        call = call_from_ci(ci, floor)
        if call == Call.INCONCLUSIVE:
            status = OutcomeStatus.INCONCLUSIVE
        if test.test_id.startswith("T3") and resolution_s is not None and np.isfinite(raw.get("resolution_s", np.nan)):
            # a NULL latency within the acquisition resolution cannot separate simultaneous from fast propagation
            raw = dict(raw, resolution_s=raw["resolution_s"])
    details = {k: v for k, v in raw.items() if k not in ("estimate", "ci")}
    o = TestOutcome(test_id=test.test_id, test_spec_hash=test.spec_hash, stage=stage, partition=partition,
                    status=status, call=call if status == OutcomeStatus.DECIDED else (Call.INCONCLUSIVE),
                    estimate=None if est is None or not np.isfinite(est) else float(est),
                    ci=None if ci is None or not np.all(np.isfinite(ci)) else [float(ci[0]), float(ci[1])],
                    min_effect=floor, n_units=n_units, n_observations=int(cov.get("n_obs", 0)), unit_ids=sorted(set(units)),
                    feature_family=test.statistic_family, contamination_grade=grade, details=_jsonable(details),
                    derived_from=list(derived_from))
    return o.finalize()


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x
