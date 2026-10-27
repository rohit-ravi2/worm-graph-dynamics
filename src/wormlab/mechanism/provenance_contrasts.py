"""Engine contrast wrappers for the g3 signal-provenance / unmixing families (registered into contrasts.REGISTRY).

Each takes the provider container {animal, rows, spont, unmix, halo, measurement} and returns the raw dict expected by
contrasts.to_outcome (estimate, ci, coverage, details)."""
from __future__ import annotations

import numpy as np

from . import contrasts
from . import signal_provenance as SP
from . import unmixing as U


def _cov(res: dict, n_animals: int, n_obs: int) -> dict:
    return dict(res, coverage=dict(n_animals=int(n_animals), n_obs=int(n_obs)))


def t13(tab, *, seed=0):
    r = SP.t13_optical_copy_subtracted(tab["rows"], seed)
    return _cov(r, r.get("n_animals", 0), r.get("n_rows", 0))


def t14(tab, *, seed=0):
    r = SP.t14_brightness_elasticity(tab["rows"], seed)
    return _cov(r, r.get("n_animals", 0), r.get("n_near", 0))


def t16(tab, *, seed=0):
    r = SP.t16_spontaneous_excess(tab["spont"], seed)
    return _cov(r, r.get("n_animals", 0), r.get("n_near", 0))


def t17(tab, *, seed=0):
    r = U.t17_unmixed_proximity(tab["unmix"], seed)
    r["halo_profile"] = tab.get("halo")
    return _cov(r, r.get("n_animals", 0), r.get("n_rows", 0))


def t6u(tab, *, seed=0, near=(0.0, 6.0), mid=(9.0, 15.0)):
    u = tab["unmix"]
    m = (np.abs(u["box_target_z"]) < 1.0) & (((u["dist_um"] >= near[0]) & (u["dist_um"] < near[1])) | ((u["dist_um"] >= mid[0]) & (u["dist_um"] < mid[1])))
    t = SP._sel(u, m)
    if not len(t["animal"]):
        return _cov(dict(estimate=np.nan, ci=[np.nan, np.nan]), 0, 0)
    est, ci, _ = SP._boot(t, lambda s: SP._mh_rd(s["resp"], s["dist_um"] < near[1], s["animal"]), seed)
    return _cov(dict(estimate=float(est), ci=ci), len(np.unique(t["animal"])), len(t["animal"]))


def t18(tab, *, seed=0, max_um=15.0, bin_um=2.0):
    u = tab["unmix"]
    m = (u["box_target_z"] > SP.Z_RESP) & u["labeled"].astype(bool) & (u["dist_um"] <= max_um)
    t = SP._sel(u, m)
    if not len(t["animal"]):
        return _cov(dict(estimate=np.nan, ci=[np.nan, np.nan]), 0, 0)
    t["_x"] = t["gap"].astype(bool)
    t["_s"] = np.char.add(t["animal"].astype(str), np.digitize(t["dist_um"], np.arange(bin_um, max_um + bin_um, bin_um)).astype(str))
    est, ci, _ = SP._boot(t, lambda s: SP._mh_rd(s["resp"], s["_x"], s["_s"]), seed)
    both = sum(1 for a in np.unique(t["animal"]) if t["_x"][t["animal"] == a].any() and (~t["_x"][t["animal"] == a]).any())
    return dict(estimate=float(est), ci=ci, n_gap=int(t["_x"].sum()), coverage=dict(n_animals_with_both=int(both), n_obs=int(len(t["animal"]))))


def t19(tab, *, seed=0, max_um=8.0, min_offset_px=1.0):
    u = tab["unmix"]
    m = (u["box_target_z"] > SP.Z_RESP) & (u["dist_um"] < max_um) & np.isfinite(u["cos_offset"]) & (u["offset_px"] >= min_offset_px) & (np.abs(u["cos_offset"]) > 0.5)
    t = SP._sel(u, m)
    if not len(t["animal"]):
        return _cov(dict(estimate=np.nan, ci=[np.nan, np.nan]), 0, 0)
    t["_x"] = t["cos_offset"] > 0.5
    t["_s"] = np.char.add(t["animal"].astype(str), np.digitize(t["dist_um"], np.arange(2.0, max_um + 2.0, 2.0)).astype(str))
    est, ci, _ = SP._boot(t, lambda s: SP._mh_rd(s["resp"], s["_x"], s["_s"]), seed)
    both = sum(1 for a in np.unique(t["animal"]) if t["_x"][t["animal"] == a].any() and (~t["_x"][t["animal"] == a]).any())
    return dict(estimate=float(est), ci=ci, coverage=dict(n_animals_with_both=int(both), n_obs=int(len(t["animal"]))))


def t20(tab, *, seed=0):
    r = U.t20_axial_anisotropy(tab["unmix"], seed, resp_key="box_resp")
    return _cov(r, r.get("n_animals", 0), r.get("n_ring_axial", 0) + r.get("n_ring_lateral", 0))


def t20u(tab, *, seed=0):
    r = U.t20_axial_anisotropy(tab["unmix"], seed, resp_key="resp")
    return _cov(r, r.get("n_animals", 0), r.get("n_ring_axial", 0) + r.get("n_ring_lateral", 0))


G3_REGISTRY = {
    "T13_optical_copy_subtracted_proximity": t13,
    "T14_brightness_elasticity": t14,
    "T16_spontaneous_excess": t16,
    "T17_unmixed_proximity": t17,
    "T6u_unmixed_target_flat": t6u,
    "T18_unmixed_gap_topology": t18,
    "T19_unmixed_spot_side": t19,
    "T20_axial_lateral_anisotropy": t20,
    "T20u_axial_lateral_anisotropy_unmixed": t20u,
}


def register():
    contrasts.REGISTRY.update(G3_REGISTRY)
