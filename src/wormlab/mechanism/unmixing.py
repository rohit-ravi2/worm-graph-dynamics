"""Evoked-volume unmixing: separate each receiver's own evoked amplitude from the target's light (compact + halo).

For one stim event with target t, crop voxels x (event_crops): dF(x) = post(x) - pre(x)
    dF(x) = sum_i a_i G_i(x) + sum_k h_k B_k(r_t(x)) + b
* G_i  : unit-amplitude static light shape of every detected nucleus whose centre lies in (or within a margin of) the
         crop - the same anisotropic Gaussian model (shared sigma, sub-voxel centres) fitted on the animal's mean volume
         in signal_provenance.fit_light_model; never fitted on outcomes.
* B_k  : piecewise-linear radial hat functions of the anisotropic (um) distance from the target centre, knots from
         HALO_START_UM outward: a FREE activity-dependent halo around the target of arbitrary radial shape.
* b    : constant.
Weighted least squares with approximate shot-noise weights 1 / (pre_counts/n_pre + post_counts/n_post).
The same design without halo on the PRE crop gives each nucleus' local baseline amplitude A_i, so the receiver's own
unmixed dF/F is a_i / A_i.

Contamination of any compact (static-kernel) or radially symmetric (halo) form is absorbed by a_t and h; a receiver's
own indicator response is absorbed by a_n. Residual confound (stated, not solved): an anisotropic activity-dependent
structure of the target (e.g. a neurite) passing through a receiver's light footprint.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse

from . import signal_provenance as SP

HALO_START_UM = 1.0
HALO_KNOTS_UM = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 17.0)   # selected on synthetic worlds only: halo_basis_variants.json
MARGIN = (2, 5, 5)
OFFSET = 100.0
VERSION = "unmixing/1.0"


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _hat_basis(r, knots):
    B = np.zeros((len(r), len(knots)))
    for k, c in enumerate(knots):
        lo = knots[k - 1] if k > 0 else c
        hi = knots[k + 1] if k + 1 < len(knots) else c
        left = (r >= lo) & (r <= c)
        right = (r > c) & (r <= hi)
        if c > lo:
            B[left, k] = (r[left] - lo) / (c - lo)
        else:
            B[r == c, k] = 1.0
        if hi > c:
            B[right, k] = (hi - r[right]) / (hi - c)
    B[r < knots[0], :] = 0.0
    return B


HALO_MODE = "both"         # isotropic + PSF-anisotropic radial families (most conservative against own-indicator claims; synthetic selection)


def design(shape, origin, centres, sigma_xy, sigma_z, target_centre, um=(1.22, 0.42, 0.42), halo=True, knots=None, mode=None):
    """Sparse design matrix (voxels x [nuclei..., halo..., const]) for a crop; returns (X, nucleus_index)."""
    P, Y, X = shape
    o = np.asarray(origin, float)
    lo = o - np.array(MARGIN)
    hi = o + np.array(shape) + np.array(MARGIN)
    inside = np.where(np.all(centres >= lo, 1) & np.all(centres < hi, 1))[0]
    cols, rows, vals = [], [], []
    for j, i in enumerate(inside):
        r, v = SP._gauss_col(centres[i] - o, shape, sigma_xy, sigma_z)
        rows.append(r); vals.append(v); cols.append(np.full(len(r), j))
    nvox = P * Y * X
    parts = []
    if inside.size:
        parts.append(sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nvox, len(inside))))
    if halo:
        zz, yy, xx = np.meshgrid(np.arange(P), np.arange(Y), np.arange(X), indexing="ij")
        tc = np.asarray(target_centre, float) - o
        knots = np.array(knots if knots is not None else HALO_KNOTS_UM)
        mode = mode or HALO_MODE
        dz2 = ((zz.ravel() - tc[0]) * um[0]) ** 2
        dxy2 = ((yy.ravel() - tc[1]) * um[1]) ** 2 + ((xx.ravel() - tc[2]) * um[2]) ** 2
        if mode in ("iso", "both"):
            parts.append(sparse.csr_matrix(_hat_basis(np.sqrt(dz2 + dxy2), knots)))
        if mode in ("aniso", "both"):
            a = (sigma_z * um[0]) / (sigma_xy * um[1])          # axial elongation of the imaging kernel
            parts.append(sparse.csr_matrix(_hat_basis(np.sqrt(dz2 / a ** 2 + dxy2), knots)))
    parts.append(sparse.csr_matrix(np.ones((nvox, 1))))
    return sparse.hstack(parts).tocsr(), inside


def _wls(Xm, y, w, ridge=1e-6):
    sw = np.sqrt(w)
    Xw = Xm.multiply(sw[:, None]).tocsr()
    A = (Xw.T @ Xw).toarray()
    bvec = Xw.T @ (y * sw)
    A[np.diag_indices_from(A)] += ridge * max(np.trace(A) / len(A), 1e-9)
    return np.linalg.solve(A, bvec)


def unmix_event(pre, post, n_pre, n_post, origin, light, target_idx, *, halo=True, um=(1.22, 0.42, 0.42), knots=None, mode=None) -> dict:
    shape = pre.shape
    cen = light["centres"]
    Xh, inside = design(shape, origin, cen, light["sigma_xy_px"], light["sigma_z_planes"], cen[target_idx], um=um, halo=halo, knots=knots, mode=mode)
    Xs, _ = design(shape, origin, cen, light["sigma_xy_px"], light["sigma_z_planes"], cen[target_idx], um=um, halo=False)
    p = np.maximum(pre.ravel().astype(float) - OFFSET, 1.0)
    q = np.maximum(post.ravel().astype(float) - OFFSET, 1.0)
    w = 1.0 / (p / max(n_pre, 1) + q / max(n_post, 1))
    dF = post.ravel().astype(float) - pre.ravel().astype(float)
    coef = _wls(Xh, dF, w)
    base = _wls(Xs, pre.ravel().astype(float) - OFFSET, 1.0 / p)
    n = len(inside)
    n_halo = Xh.shape[1] - n - 1
    # conventional 3x5x5 box dF and pre (counts above offset) at each nucleus' integer centre, for identical-row comparison
    box_dF = np.full(n, np.nan)
    box_pre = np.full(n, np.nan)
    D3 = (post.astype(float) - pre.astype(float))
    for j, i in enumerate(inside):
        c = np.round(cen[i]).astype(int) - np.asarray(origin).astype(int)
        if np.all(c - np.array([1, 2, 2]) >= 0) and np.all(c + np.array([2, 3, 3]) <= np.array(shape)):
            sl = (slice(c[0] - 1, c[0] + 2), slice(c[1] - 2, c[1] + 3), slice(c[2] - 2, c[2] + 3))
            box_dF[j] = D3[sl].mean()
            box_pre[j] = pre[sl].mean() - OFFSET
    img = image_metrics(pre, post, origin, cen, inside, target_idx, light["sigma_xy_px"], light["sigma_z_planes"], um)
    return dict(nuclei=inside, a=coef[:n], halo=coef[n:n + n_halo], const=float(coef[-1]), A=base[:n],
                box_dF=box_dF, box_pre=box_pre, img=img)


IMG_FIELDS = ("shift_to_target_um", "corr_own_shape", "corr_target_light", "local_bg_dF", "grad_cos_to_target", "static_shift_um")


def image_metrics(pre, post, origin, cen, inside, target_idx, sxy, sz, um=(1.22, 0.42, 0.42), half=(1, 4, 4)) -> np.ndarray:
    """§24 raw-voxel metrics for every nucleus in the crop relative to the target (rows aligned with ``inside``):
    shift_to_target_um   dF(+)-weighted centroid offset of the receiver-local patch projected on the unit vector to the target
    corr_own_shape       Pearson(dF patch, pre patch - local background): own response follows the receiver's own light shape
    corr_target_light    Pearson(dF patch, unit target light model restricted to the patch): contamination follows target light
    local_bg_dF          mean dF of patch voxels > 1.5 sigma from every nucleus centre (inter-nuclear haze)
    grad_cos_to_target   cosine between the patch-mean dF spatial gradient (um^-1) and the direction to the target
    static_shift_um      centroid shift of static brightness (post vs pre) in the patch (local motion)"""
    P, Y, X = pre.shape
    o = np.asarray(origin, float)
    dF = post.astype(float) - pre.astype(float)
    t_c = cen[target_idx] - o
    out = np.full((len(inside), len(IMG_FIELDS)), np.nan)
    sc = np.array(um)
    cloc = cen[inside] - o
    for j, i in enumerate(inside):
        if i == target_idx:
            continue
        c = np.round(cen[i] - o).astype(int)
        lo = c - np.array(half)
        hi = c + np.array(half) + 1
        if np.any(lo < 0) or np.any(hi > np.array([P, Y, X])):
            continue
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        d, pr, po = dF[sl], pre[sl].astype(float), post[sl].astype(float)
        zz, yy, xx = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), np.arange(lo[2], hi[2]), indexing="ij")
        coords = np.stack([zz, yy, xx], -1).astype(float)
        u = (t_c - (cen[i] - o)) * sc
        un = np.linalg.norm(u)
        if un <= 0:
            continue
        u = u / un
        w = np.clip(d, 0, None)
        if w.sum() > 0:
            cm = (coords * w[..., None]).sum((0, 1, 2)) / w.sum()
            out[j, 0] = float(((cm - (cen[i] - o)) * sc) @ u)
        bgv = np.percentile(pr, 20)
        if np.std(d) > 0 and np.std(pr) > 0:
            out[j, 1] = float(np.corrcoef(d.ravel(), (pr - bgv).ravel())[0, 1])
        gt = np.exp(-((zz - t_c[0]) ** 2) / (2 * sz ** 2) - ((yy - t_c[1]) ** 2 + (xx - t_c[2]) ** 2) / (2 * sxy ** 2))
        if np.std(d) > 0 and np.std(gt) > 0:
            out[j, 2] = float(np.corrcoef(d.ravel(), gt.ravel())[0, 1])
        dmin = np.full(d.shape, np.inf)
        for k in range(len(cloc)):
            dz_, dy_, dx_ = (zz - cloc[k, 0]) / max(sz, 1e-6), (yy - cloc[k, 1]) / max(sxy, 1e-6), (xx - cloc[k, 2]) / max(sxy, 1e-6)
            dmin = np.minimum(dmin, np.sqrt(dz_ ** 2 + dy_ ** 2 + dx_ ** 2))
        bgm = dmin > 1.5
        if bgm.sum() >= 5:
            out[j, 3] = float(d[bgm].mean())
        gz, gy, gx = np.gradient(d, *sc)
        g = np.array([gz.mean(), gy.mean(), gx.mean()])
        if np.linalg.norm(g) > 0:
            out[j, 4] = float(g @ u / np.linalg.norm(g))
        wp, wq = np.clip(pr - bgv, 0, None), np.clip(po - bgv, 0, None)
        if wp.sum() > 0 and wq.sum() > 0:
            cp = (coords * wp[..., None]).sum((0, 1, 2)) / wp.sum()
            cq = (coords * wq[..., None]).sum((0, 1, 2)) / wq.sum()
            out[j, 5] = float(np.linalg.norm((cq - cp) * sc))
    return out


def unmix_animal(animal: str, prov: dict, crops: dict, light: dict | None = None, *, halo: bool = True, knots=None, mode=None) -> dict:
    """Rows per (event, receiver in crop): unmixed a, local baseline A, distance; plus per-event halo profiles."""
    light = light or SP.fit_light_model(prov["mean_volume"], prov["nuclei"])
    um = (prov["meta"]["um_per_plane"], prov["meta"]["um_per_px"], prov["meta"]["um_per_px"])
    P_um = light["centres"] * np.array(um)
    evs = [e for e in SP.event_windows(prov) if f"e{e['event']}_pre" in crops]
    recs, halos = [], []
    for e in evs:
        k = f"e{e['event']}"
        t = int(e["target_nucleus"])
        n_pre, n_post = crops[k + "_n"]
        r = unmix_event(crops[k + "_pre"], crops[k + "_post"], n_pre, n_post, crops[k + "_origin"], light, t, halo=halo, um=um, knots=knots, mode=mode)
        ti = np.where(r["nuclei"] == t)[0]
        if not len(ti):
            continue
        a_t, A_t = r["a"][ti[0]], r["A"][ti[0]]
        halos.append(dict(event=int(e["event"]), target=t, a_t=float(a_t), A_t=float(A_t), halo=r["halo"].tolist()))
        dv = P_um[r["nuclei"]] - P_um[t]
        d = np.linalg.norm(dv, axis=1)
        for j, i in enumerate(r["nuclei"]):
            recs.append((int(e["event"]), t, int(i), float(d[j]), float(r["a"][j]), float(r["A"][j]), float(a_t), float(A_t),
                         float(r["box_dF"][j]), float(r["box_pre"][j]), float(abs(dv[j, 0])), float(np.hypot(dv[j, 1], dv[j, 2])), *[float(x) for x in r["img"][j]]))
    arr = np.array(recs, dtype=[("event", "i8"), ("target", "i8"), ("receiver", "i8"), ("dist_um", "f8"), ("a", "f8"), ("A", "f8"), ("a_t", "f8"),
                                ("A_t", "f8"), ("box_dF", "f8"), ("box_pre", "f8"), ("dz_um", "f8"), ("dxy_um", "f8")] + [(f, "f8") for f in IMG_FIELDS])
    return dict(animal=animal, rows=arr, halos=halos, sigma=(light["sigma_xy_px"], light["sigma_z_planes"]))


def response_table(unmixed: list[dict], min_A: float = 5.0) -> dict:
    """Per animal: unmixed dF/F = a / A; robust z per receiver across the events whose crops contain it; target z the
    same way. Returns concatenated rows (dict of arrays)."""
    out = {k: [] for k in ("animal", "event", "target", "receiver", "dist_um", "dff", "z", "resp", "target_z", "A", "box_z", "box_resp", "box_target_z",
                           "target_dff", "target_A", "dz_um", "dxy_um", "box_dF_t") + IMG_FIELDS}
    for u in unmixed:
        R = u["rows"]
        if not len(R):
            continue
        ev = np.unique(R["event"])
        nuc = np.unique(np.concatenate([R["receiver"], R["target"]]))
        ei = {e: i for i, e in enumerate(ev)}
        ni = {n: i for i, n in enumerate(nuc)}
        M = np.full((len(ev), len(nuc)), np.nan)
        Mb = np.full((len(ev), len(nuc)), np.nan)
        for r in R:
            if r["A"] > min_A:
                M[ei[r["event"]], ni[r["receiver"]]] = r["a"] / r["A"]
            if np.isfinite(r["box_pre"]) and r["box_pre"] > min_A:
                Mb[ei[r["event"]], ni[r["receiver"]]] = r["box_dF"] / r["box_pre"]
        Z = SP.robust_z(M)
        Zb = SP.robust_z(Mb)
        for r in R:
            if r["receiver"] == r["target"] or not np.isfinite(Z[ei[r["event"]], ni[r["receiver"]]]):
                continue
            tz = Z[ei[r["event"]], ni[r["target"]]]
            out["animal"].append(u["animal"]); out["event"].append(int(r["event"])); out["target"].append(int(r["target"]))
            out["receiver"].append(int(r["receiver"])); out["dist_um"].append(float(r["dist_um"]))
            out["dff"].append(float(M[ei[r["event"]], ni[r["receiver"]]])); z = float(Z[ei[r["event"]], ni[r["receiver"]]])
            out["z"].append(z); out["resp"].append(z > SP.Z_RESP); out["target_z"].append(float(tz)); out["A"].append(float(r["A"]))
            bz = Zb[ei[r["event"]], ni[r["receiver"]]]
            out["box_z"].append(float(bz)); out["box_resp"].append(bool(bz > SP.Z_RESP)); out["box_target_z"].append(float(Zb[ei[r["event"]], ni[r["target"]]]))
            out["target_dff"].append(float(M[ei[r["event"]], ni[r["target"]]])); out["target_A"].append(float(r["A_t"]))
            out["dz_um"].append(float(r["dz_um"])); out["dxy_um"].append(float(r["dxy_um"]))
            tb = R[(R["event"] == r["event"]) & (R["receiver"] == r["target"])]
            out["box_dF_t"].append(float(tb["box_dF"][0]) if len(tb) else np.nan)
            for f in IMG_FIELDS:
                out[f].append(float(r[f]))
    return {k: np.array(v) for k, v in out.items()}


def t17_unmixed_proximity(tab: dict, seed: int = 0, near=(0.0, 6.0), mid=(9.0, 15.0)) -> dict:
    # event selection by the conventional box target z (identical rows for unmixed and box statistics)
    m = (tab["box_target_z"] > SP.Z_RESP) & np.isfinite(tab["box_z"]) & (((tab["dist_um"] >= near[0]) & (tab["dist_um"] < near[1])) | ((tab["dist_um"] >= mid[0]) & (tab["dist_um"] < mid[1])))
    t = SP._sel(tab, m)
    if not len(t["animal"]):
        return dict(estimate=np.nan, ci=[np.nan, np.nan], n_rows=0)

    def stat(s):
        nm = s["dist_um"] < near[1]
        u = SP._mh_rd(s["resp"], nm, s["animal"])
        b = SP._mh_rd(s["box_resp"], nm, s["animal"])
        return np.array([u, b, 1 - u / b if np.isfinite(b) and abs(b) > 1e-9 else np.nan])

    est, _, vals = SP._boot(t, stat, seed)
    vals = np.array(vals) if len(vals) else np.full((1, 3), np.nan)
    pc = lambda k: [float(np.nanpercentile(vals[:, k], 2.5)), float(np.nanpercentile(vals[:, k], 97.5))] if np.isfinite(vals[:, k]).any() else [np.nan, np.nan]
    nm = t["dist_um"] < near[1]
    return dict(estimate=float(est[0]), ci=pc(0), rd_box=float(est[1]), rd_box_ci=pc(1), explained_fraction=float(est[2]), explained_fraction_ci=pc(2),
                n_rows=int(len(t["animal"])), n_near=int(nm.sum()), n_animals=int(len(np.unique(t["animal"]))),
                near_rate=float(t["resp"][nm].mean()) if nm.any() else np.nan, mid_rate=float(t["resp"][~nm].mean()) if (~nm).any() else np.nan,
                near_rate_box=float(t["box_resp"][nm].mean()) if nm.any() else np.nan, mid_rate_box=float(t["box_resp"][~nm].mean()) if (~nm).any() else np.nan)


def halo_profile(unmixed: list[dict], min_target_z_proxy: float = 0.0) -> dict:
    """Mean halo coefficient per knot normalised by the target's unmixed amplitude, over events with a_t > 0; and the
    fraction of the target's evoked light carried by the halo (knot >= HALO_START_UM)."""
    H, W = [], []
    for u in unmixed:
        for h in u["halos"]:
            if h["a_t"] > 0 and h["A_t"] > 0 and h["a_t"] / h["A_t"] > min_target_z_proxy:
                H.append(np.array(h["halo"]) / h["a_t"])
                W.append(u["animal"])
    if not H:
        return dict(n=0)
    H = np.array(H)
    return dict(n=int(len(H)), knots_um=list(HALO_KNOTS_UM), median_halo_over_target=np.median(H, 0).tolist(), mean_halo_over_target=H.mean(0).tolist())


def t20_axial_anisotropy(tab: dict, seed: int = 0, resp_key: str = "box_resp", ring=(3.0, 6.0), control=(9.0, 15.0), bin_um=1.0,
                         exclude_lr_homologs: bool = True) -> dict:
    """Axial-vs-lateral double difference: MH RD of response for AXIAL (dz >= dxy) vs LATERAL receivers within the 3-6 um
    ring (strata animal x 1-um 3D-distance bin), minus the same contrast in the 9-15 um control ring (anatomical
    orientation baseline). Optical contamination follows the axially elongated imaging kernel and extraction box (POS);
    biological coupling is indifferent to the imaging axis (NULL) EXCEPT for left/right homolog pairs, which lie along the
    imaging axis in side-mounted worms and are often electrically coupled: labeled L/R homolog pairs are excluded
    (unlabeled homologs cannot be; residual confound stated)."""
    base = tab["box_target_z"] > SP.Z_RESP
    if exclude_lr_homologs and "lr_homolog" in tab:
        base = base & ~tab["lr_homolog"].astype(bool)    # bilateral homologs lie along the imaging (left-right) axis in side-mounted worms
    if resp_key == "resp":
        base = base & np.isfinite(tab["z"])
    else:
        base = base & np.isfinite(tab["box_z"])
    m = base & (((tab["dist_um"] >= ring[0]) & (tab["dist_um"] < ring[1])) | ((tab["dist_um"] >= control[0]) & (tab["dist_um"] < control[1])))
    t = SP._sel(tab, m)
    if not len(t["animal"]):
        return dict(estimate=np.nan, ci=[np.nan, np.nan], n_rows=0)
    t["_ax"] = t["dz_um"] >= t["dxy_um"]
    t["_in"] = t["dist_um"] < ring[1]
    t["_s"] = np.char.add(t["animal"].astype(str), np.floor(t["dist_um"] / bin_um).astype(int).astype(str))

    def stat(s):
        a = SP._mh_rd(s[resp_key][s["_in"]], s["_ax"][s["_in"]], s["_s"][s["_in"]])
        b = SP._mh_rd(s[resp_key][~s["_in"]], s["_ax"][~s["_in"]], s["_s"][~s["_in"]])
        return np.array([a - b, a, b])

    est, _, vals = SP._boot(t, stat, seed)
    vals = np.array(vals) if len(vals) else np.full((1, 3), np.nan)
    pc = lambda k: [float(np.nanpercentile(vals[:, k], 2.5)), float(np.nanpercentile(vals[:, k], 97.5))] if np.isfinite(vals[:, k]).any() else [np.nan, np.nan]
    inr = t["_in"]
    return dict(estimate=float(est[0]), ci=pc(0), ring_rd=float(est[1]), ring_rd_ci=pc(1), control_rd=float(est[2]), control_rd_ci=pc(2),
                n_ring_axial=int((inr & t["_ax"]).sum()), n_ring_lateral=int((inr & ~t["_ax"]).sum()), n_animals=int(len(np.unique(t["animal"]))),
                resp_key=resp_key)
