"""Descriptive audit families for the Deep Mechanism Campaign (§6-21): exploratory maps computed from the g3 container,
never adjudicated. Each function returns a JSON-able dict and states the operationalization it closes.

Container: randi.provider_g3.assemble output (+ per-animal descriptive products under "desc").
"""
from __future__ import annotations

import numpy as np

from . import signal_provenance as SP
from . import unmixing as U

NEAR, MID, FAR = (0.0, 6.0), (9.0, 15.0), (15.0, 40.0)


def _rd(t, resp, near_mask, strata, seed=0, boot=True):
    if not len(t["animal"]) or not near_mask.any() or near_mask.all():
        return dict(estimate=np.nan, ci=[np.nan, np.nan], n=int(len(t["animal"])))
    t = dict(t, _r=resp, _x=near_mask, _s=strata)
    if not boot:
        return dict(estimate=float(SP._mh_rd(t["_r"], t["_x"], t["_s"])), n=int(len(t["animal"])))
    est, ci, _ = SP._boot(t, lambda s: SP._mh_rd(s["_r"], s["_x"], s["_s"]), seed, n=200)
    return dict(estimate=float(est), ci=ci, n=int(len(t["animal"])), n_near=int(near_mask.sum()))


def p1_distance_curve(c: dict, bins=(0, 2, 3, 4, 5, 6, 8, 10, 12, 15)) -> dict:
    u = c["unmix"]
    up = u["box_target_z"] > SP.Z_RESP
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = up & (u["dist_um"] >= lo) & (u["dist_um"] < hi)
        if m.sum() < 10:
            continue
        out.append(dict(shell=[lo, hi], n=int(m.sum()), box_rate=float(u["box_resp"][m].mean()), unmixed_rate=float(u["resp"][m].mean())))
    return dict(unmixed_and_box=out, kappa_residual=SP.w9_spatial_copy_curve(c["rows"]),
                closes="response-rate shells on TARGET-UP events, identical rows for box vs unmixed")


def p2_anisotropy(c: dict) -> dict:
    u = c["unmix"]
    up = u["box_target_z"] > SP.Z_RESP
    dz, dxy = u.get("dz_um"), u.get("dxy_um")
    if dz is None:
        return dict(note="dz/dxy not available")
    grid = []
    for zlo, zhi in ((0, 1.5), (1.5, 3.0), (3.0, 6.0)):
        for rlo, rhi in ((0, 2), (2, 4), (4, 6), (6, 10)):
            m = up & (dz >= zlo) & (dz < zhi) & (dxy >= rlo) & (dxy < rhi)
            if m.sum() >= 10:
                grid.append(dict(dz_um=[zlo, zhi], dxy_um=[rlo, rhi], n=int(m.sum()), box_rate=float(u["box_resp"][m].mean()), unmixed_rate=float(u["resp"][m].mean())))
    return dict(grid=grid, closes="axial vs lateral separation grid; optics predict axial elongation of the BOX effect only")


def _near_mid(u, extra_mask):
    m = extra_mask & (((u["dist_um"] >= NEAR[0]) & (u["dist_um"] < NEAR[1])) | ((u["dist_um"] >= MID[0]) & (u["dist_um"] < MID[1])))
    t = SP._sel(u, m)
    return t, t["dist_um"] < NEAR[1]


def p3_animal_heterogeneity(c: dict) -> dict:
    u = c["unmix"]
    rows = []
    for a in np.unique(u["animal"]):
        t, nm = _near_mid(u, (u["animal"] == a) & (u["box_target_z"] > SP.Z_RESP))
        if nm.sum() < 5 or (~nm).sum() < 5:
            continue
        q = c["measurement"].get(a, {})
        rows.append(dict(animal=a, rd_box=float(t["box_resp"][nm].mean() - t["box_resp"][~nm].mean()),
                         rd_unmixed=float(t["resp"][nm].mean() - t["resp"][~nm].mean()), **{k: q.get(k) for k in ("sigma_xy_px", "sigma_z_planes", "r2_support", "n_nuclei")}))
    out = dict(per_animal=rows)
    for k in ("sigma_xy_px", "sigma_z_planes", "r2_support", "n_nuclei"):
        for v in ("rd_box", "rd_unmixed"):
            x = np.array([r[k] for r in rows], float)
            y = np.array([r[v] for r in rows], float)
            ok = np.isfinite(x) & np.isfinite(y)
            out[f"spearman_{v}_vs_{k}"] = SP._spearman(x[ok], y[ok]) if ok.sum() >= 6 else None
    out["closes"] = "per-animal near-mid response difference vs image-quality metrics (Spearman across animals)"
    return out


def p4_p5_target_brightness_density(c: dict) -> dict:
    u = c["unmix"]
    out = {}
    for key in ("target_A", "target_density"):
        if key not in u:
            continue
        up = u["box_target_z"] > SP.Z_RESP
        x = u[key]
        qs = np.nanpercentile(x[up], [33.3, 66.7])
        res = []
        for i, (lo, hi) in enumerate(((-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], np.inf))):
            t, nm = _near_mid(u, up & (x >= lo) & (x < hi))
            res.append(dict(tertile=i, box=_rd(t, t["box_resp"], nm, t["animal"]), unmixed=_rd(t, t["resp"], nm, t["animal"])))
        out[key] = res
    out["closes"] = "near-mid RD by target brightness (light-model amplitude) and local density (nuclei within 6 um) tertiles"
    return out


def p6_roi_geometry(c: dict) -> dict:
    return dict(roi_type_rd=c.get("desc_roi_rd"), closes="near-far response RD with core / box / wide ROIs (optics predict wide > box > core)")


def o1_target_flat_thresholds(c: dict) -> dict:
    u = c["unmix"]
    out = []
    for thr in (0.5, 1.0, 1.5, 2.0):
        t, nm = _near_mid(u, np.abs(u["box_target_z"]) < thr)
        out.append(dict(abs_target_z_below=thr, box=_rd(t, t["box_resp"], nm, t["animal"]), unmixed=_rd(t, t["resp"], nm, t["animal"])))
    return dict(thresholds=out, closes="target-flat near-mid RD across flatness thresholds")


def o2_effectiveness_latent(c: dict) -> dict:
    u = c["unmix"]
    up = u["box_target_z"] > SP.Z_RESP
    out = {}
    for name, rng in (("near", NEAR), ("mid", MID)):
        m = up & (u["dist_um"] >= rng[0]) & (u["dist_um"] < rng[1]) & np.isfinite(u["dff"]) & np.isfinite(u.get("target_dff", u["dff"] * np.nan))
        if m.sum() < 20:
            continue
        x, y = u["target_dff"][m], u["dff"][m]
        out[name] = dict(n=int(m.sum()), spearman_receiver_vs_target_amplitude=SP._spearman(x, y))
    out["closes"] = ("graded dependence of receiver unmixed dF/F on target amplitude among responding targets: mediation predicts a graded "
                     "near-specific slope; a binary effectiveness latent predicts none beyond response/no-response")
    return out


def o4_displacement(c: dict) -> dict:
    return dict(leifer=c.get("desc_displacement"), closes="event-locked displacement of tracked neurons (Leifer OSF coordinates), post vs pre")


def e1_e2_topology(c: dict) -> dict:
    u = c["unmix"]
    up = (u["box_target_z"] > SP.Z_RESP) & u["labeled"].astype(bool) & (u["dist_um"] <= 15)
    t = SP._sel(u, up)
    strata = np.char.add(t["animal"].astype(str), np.digitize(t["dist_um"], np.arange(2, 17, 2)).astype(str))
    out = dict(gap_unmixed=_rd(t, t["resp"], t["gap"].astype(bool), strata), gap_box=_rd(t, t["box_resp"], t["gap"].astype(bool), strata))
    if "chem" in t:
        out["chem_unmixed"] = _rd(t, t["resp"], t["chem"].astype(bool), strata)
    if "gap_weight" in t:
        res = []
        for lo, hi in ((0, 1), (1, 3), (3, 1e9)):
            m = (t["gap_weight"] >= lo) & (t["gap_weight"] < hi)
            res.append(dict(gap_weight=[lo, hi], n=int(m.sum()), unmixed_rate=float(t["resp"][m].mean()) if m.any() else None))
        out["gap_weight_dose"] = res
    out["closes"] = "labeled pairs <= 15 um, TARGET-UP, strata animal x 2-um bin"
    return out


def e4_network_features(c: dict) -> dict:
    """E4: does the near-mid response difference scale with the gap-junction degree of the target (labeled targets) or of
    the receiver (labeled receivers)? Electrical coupling predicts larger near effects for highly coupled neurons; optics
    and light do not."""
    u = c["unmix"]
    if "gap_degree_target" not in u:
        return dict(note="labels not available")
    up = u["box_target_z"] > SP.Z_RESP
    out = {}
    for key in ("gap_degree_target", "gap_degree_receiver"):
        x = u[key]
        ok = up & np.isfinite(x)
        if ok.sum() < 30:
            out[key] = dict(n=int(ok.sum()))
            continue
        qs = np.nanpercentile(x[ok], [50])
        res = []
        for name, m in (("low", ok & (x <= qs[0])), ("high", ok & (x > qs[0]))):
            t, nm = _near_mid(u, m)
            res.append(dict(split=name, degree_threshold=float(qs[0]), box=_rd(t, t["box_resp"], nm, t["animal"]), unmixed=_rd(t, t["resp"], nm, t["animal"])))
        out[key] = res
    out["closes"] = "near-mid RD split at the median gap-junction degree (wormneuroatlas) of labeled targets / receivers"
    return out


def h7_h10_local_state(c: dict) -> dict:
    sp = c["spont"]
    se = np.sqrt(np.maximum(sp["var_t"] * sp["var_r"], 1e-12) / np.maximum(sp["n_samples"], 1))
    zx = (sp["cov1_obs"] - sp["cov1_pred_h1"]) / se
    out = []
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 15), (15, 25), (25, 40)):
        m = (sp["dist_um"] >= lo) & (sp["dist_um"] < hi)
        if m.sum() >= 10:
            out.append(dict(shell=[lo, hi], n=int(m.sum()), mean_z_excess=float(np.mean(zx[m])), mean_cov1=float(np.mean(sp["cov1_obs"][m]))))
    return dict(spontaneous_excess_by_distance=out, closes="spontaneous lag-1 covariance excess over optics by distance (global = flat; local = decays)")


def p2b_anisotropy_homologs(c: dict) -> dict:
    """Axial vs lateral near (3-6 um) response rates among TARGET-UP events, split by named-pair status and L/R homology."""
    u = c["unmix"]
    if "lr_homolog" not in u:
        return dict(note="labels not available")
    up = (u["box_target_z"] > SP.Z_RESP) & (u["dist_um"] >= 3) & (u["dist_um"] < 6)
    ax = u["dz_um"] >= u["dxy_um"]
    out = {}
    for name, m in (("all", up), ("both_named", up & u["both_named"].astype(bool)), ("lr_homolog", up & u["lr_homolog"].astype(bool)),
                    ("named_non_homolog", up & u["both_named"].astype(bool) & ~u["lr_homolog"].astype(bool)), ("exclude_homolog", up & ~u["lr_homolog"].astype(bool))):
        out[name] = dict(axial_n=int((m & ax).sum()), lateral_n=int((m & ~ax).sum()),
                         axial_box=float(u["box_resp"][m & ax].mean()) if (m & ax).any() else None, lateral_box=float(u["box_resp"][m & ~ax].mean()) if (m & ~ax).any() else None,
                         axial_unmixed=float(u["resp"][m & ax].mean()) if (m & ax).any() else None, lateral_unmixed=float(u["resp"][m & ~ax].mean()) if (m & ~ax).any() else None)
    resp_ax = up & ax & u["box_resp"].astype(bool)
    out["homolog_fraction_of_axial_named_responders"] = float(u["lr_homolog"][resp_ax & u["both_named"].astype(bool)].mean()) if (resp_ax & u["both_named"].astype(bool)).any() else None
    out["closes"] = "does bilateral homology (left-right = imaging axis in side-mounted worms) account for the axial near-field excess?"
    return out


def m23_registration_mc(c: dict, draws: int = 20, levels=((0.21, 0.37), (0.42, 0.73)), seed: int = 7) -> dict:
    """§23: perturb centroid positions (isotropic lateral, axial) for target and receiver independently; recompute distance
    classes from separation magnitudes (exact in distribution for isotropic lateral error) and the near-mid RD (box and
    unmixed) on TARGET-UP events. levels = (sigma_xy_um, sigma_z_um): about 0.5 px / 0.3 plane and 1 px / 0.6 plane."""
    u = c["unmix"]
    up = u["box_target_z"] > SP.Z_RESP
    rng = np.random.default_rng(seed)
    out = []
    for sxy, sz in levels:
        est_b, est_u = [], []
        for _ in range(draws):
            ex = rng.normal(0, sxy * np.sqrt(2), (len(u["dist_um"]), 2))
            ez = rng.normal(0, sz * np.sqrt(2), len(u["dist_um"]))
            dxy = np.hypot(u["dxy_um"] + ex[:, 0], ex[:, 1])
            dz = np.abs(u["dz_um"] + ez)
            d = np.sqrt(dxy ** 2 + dz ** 2)
            m = up & ((d < NEAR[1]) | ((d >= MID[0]) & (d < MID[1])))
            t = SP._sel(dict(u, _d=d), m)
            nm = t["_d"] < NEAR[1]
            est_b.append(SP._mh_rd(t["box_resp"], nm, t["animal"]))
            est_u.append(SP._mh_rd(t["resp"], nm, t["animal"]))
        out.append(dict(sigma_xy_um=sxy, sigma_z_um=sz, box_rd=[float(np.percentile(est_b, q)) for q in (2.5, 50, 97.5)],
                        unmixed_rd=[float(np.percentile(est_u, q)) for q in (2.5, 50, 97.5)]))
    return dict(levels=out, closes="registration Monte Carlo on distance classes; spread over draws (not sampling CI)")


def r24_raw_image_strata(c: dict) -> dict:
    """§24 stratified raw-voxel image metrics (unmixing.IMG_FIELDS) with animal-cluster bootstrap CIs."""
    u = c["unmix"]
    if "shift_to_target_um" not in u:
        return dict(note="image metrics not available")
    up = u["box_target_z"] > SP.Z_RESP
    near = u["dist_um"] < NEAR[1]
    mid = (u["dist_um"] >= MID[0]) & (u["dist_um"] < MID[1])
    strata = dict(strong_near_responders=up & near & (u["box_z"] > 5), weak_near_responders=up & near & (u["box_z"] > SP.Z_RESP) & (u["box_z"] <= 5),
                  near_nonresponders=up & near & (u["box_z"] <= 1), mid_responders=up & mid & (u["box_z"] > SP.Z_RESP),
                  target_flat_near=(np.abs(u["box_target_z"]) < 1) & near,
                  unmixed_own_near_responders=up & near & (u["z"] > SP.Z_RESP))
    if "gap" in u:
        lab = u["labeled"].astype(bool) & up & (u["dist_um"] < 10)
        strata["gap_connected_pairs"] = lab & u["gap"].astype(bool)
        strata["unconnected_matched_pairs"] = lab & ~u["gap"].astype(bool)
    from .unmixing import IMG_FIELDS
    out = {}
    for name, m in strata.items():
        res = dict(n=int(m.sum()))
        if m.sum() >= 5:
            t = SP._sel(u, m)
            for f in IMG_FIELDS:
                vals = t[f]
                ok = np.isfinite(vals)
                if ok.sum() < 5:
                    continue
                tt = dict(animal=t["animal"][ok], v=vals[ok])
                est, ci, _ = SP._boot(tt, lambda s: float(np.mean(s["v"])), 3, n=200)
                res[f] = dict(mean=float(est), ci=ci)
        out[name] = res
    out["closes"] = ("contamination predicts receiver dF shifted toward the target, correlated with target light and a gradient toward the target; "
                     "own responses predict dF centred on the receiver and correlated with its own static shape")
    return out


def all_descriptives(c: dict) -> dict:
    fns = dict(P1=p1_distance_curve, P2=p2_anisotropy, P2b=p2b_anisotropy_homologs, P3=p3_animal_heterogeneity, P4_P5=p4_p5_target_brightness_density, P6=p6_roi_geometry,
               O1=o1_target_flat_thresholds, O2=o2_effectiveness_latent, O4=o4_displacement, E1_E2=e1_e2_topology, E4=e4_network_features, H7_H10=h7_h10_local_state,
               M23=m23_registration_mc, R24=r24_raw_image_strata, T20=lambda c: dict(box=U.t20_axial_anisotropy(c["unmix"], 0, resp_key="box_resp"),
                                                                              unmixed=U.t20_axial_anisotropy(c["unmix"], 0, resp_key="resp")))
    out = {}
    for k, f in fns.items():
        try:
            out[k] = f(c)
        except Exception as e:     # recorded, not hidden
            out[k] = dict(error=repr(e))
    out["W"] = c.get("desc_W")
    return out
