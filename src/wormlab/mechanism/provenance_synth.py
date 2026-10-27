"""Physically rendered synthetic worlds for the signal-provenance campaign (hostile fixtures S14-S24 and baselines).

Unlike generators.py (which writes traces directly), this module renders VOLUMES: nuclei are anisotropic Gaussian
light distributions (nucleus convolved with the imaging PSF), every plane is sampled at its own acquisition time under
the alternating sweep, photon/read/frame noise is applied, and the SAME frozen code measures it:
raw_traces._segment on the mean volume -> roi_extract operators -> provenance-format payload. Contamination therefore
arises from optics and ROI geometry, never from a hand-written copy term (except where a fixture explicitly adds a
non-optical measurement artefact, e.g. S14's low-frequency haze).

Ground truth (true neuron per detected nucleus, mechanism edges) is returned separately and never read by analyses.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import sparse
from scipy.signal import lfilter

from . import roi_extract as RX
from .randi import raw_traces

FRAME_S = 0.02
PLANES = 25


@dataclass
class WorldParams:
    # geometry / optics
    shape: tuple = (25, 96, 96)
    n_neurons: int = 120
    min_sep_um: float = 2.6
    sigma_xy_um: float = 0.85          # effective (nucleus * PSF) lateral sigma
    sigma_z_um: float = 1.35           # effective axial sigma
    halo_frac: float = 0.0             # fraction of each neuron's light in a broad halo (static AND dynamic)
    halo_sigma_um: float = 2.5
    evoked_halo_frac: float = 0.0      # halo carrying dF only (absent from the static image): static-kappa misspecification
    neurite_halo_frac: float = 0.0     # v3 hostile: activity-dependent ELONGATED (neurite-like) halo per neuron, random orientation, dF only
    neurite_sigma_long_um: float = 4.0
    neurite_sigma_short_um: float = 0.8
    psf_tail_frac: float = 0.0         # v3: fraction of each nucleus' light in a broad non-Gaussian axial/lateral tail (static AND dynamic)
    psf_tail_sigma_xy_um: float = 2.5
    psf_tail_sigma_z_um: float = 6.0
    nucleus_size_logsd: float = 0.0    # v3: per-nucleus lognormal scale of the light-distribution width (model misspecification)
    halo_lowpass_tau_s: float = 0.0    # S14: halo dynamics low-pass filtered (low-frequency-only contamination)
    f0_median: float = 250.0           # counts above offset at nucleus peak
    f0_logsd: float = 0.5
    offset: float = 100.0
    haze: float = 3.0
    gain: float = 1.0                  # photons per count
    read_sd: float = 1.5
    frame_add_sd: float = 0.0          # frame-common additive noise (counts)
    frame_mult_sd: float = 0.0         # frame-common multiplicative noise
    motion_sd_um: float = 0.0          # AR(1) rigid drift sd
    motion_rho: float = 0.95
    seg_jitter_px: float = 0.0         # static registration error of analysed nucleus positions (applied to extraction)
    # timing / activity
    n_volumes: int = 1800
    event_every_s: float = 32.0
    first_event_s: float = 40.0
    rise_s: float = 0.35
    decay_s: float = 2.5
    spont_rate_hz: float = 0.01
    spont_amp: float = 0.6
    p_target: float = 0.7
    target_amp: float = 1.2
    target_delay_s: float = 0.3
    # mechanisms
    p_gap_near: float = 0.0            # H3: prob of a gap edge at d -> 0, decays exp(-d/lam_gap)
    lam_gap_um: float = 4.0
    gap_coef: float = 0.5
    gap_latency_s: float = 0.02
    gap_threshold: float = 0.0         # receiver responds only if coef*amp > threshold (sub-threshold coupling: S19)
    p_offtarget: float = 0.0           # H2: direct light activation prob scaled by light profile
    stim_sigma_xy_um: float = 1.32     # 3.1 um FWHM
    stim_sigma_z_um: float = 3.0
    offtarget_amp: float = 0.8
    spillover_latent: bool = False     # S16: stim effectiveness gates target AND off-target together
    p_chem: float = 0.0                # random (distance-independent) chemical edges
    chem_coef: float = 0.6
    chem_latency_s: float = 0.8
    chem_topology_near_bias: float = 0.0   # S21: make chemical/gap edges more likely between near pairs (edge label only)
    local_field_amp: float = 0.0       # H10/S17: target activation spreads to all neurons with kernel exp(-d^2/2lam^2)
    local_field_lam_um: float = 4.0
    local_field_spont: float = 0.0     # spatially local slow spontaneous field amplitude
    global_amp: float = 0.0
    lf_contam: float = 0.0             # S14: non-optical low-frequency copy of target counts into ROIs within lf_radius
    lf_contam_radius_um: float = 6.0
    lf_contam_tau_s: float = 20.0
    seed: int = 0
    # cross-rig deposits (DANDI 000692/000541/000981) are ~8-bit: q = clip(floor((V - offset - compress_floor) / compress_scale), 0, compress_max).
    # The renderer re-expands q*scale + floor + offset so downstream code is unchanged. None = no compression (default; all
    # recorded worlds unchanged, and no RNG draws are added).
    compress_floor: float | None = None
    compress_scale: float = 1.0
    compress_max: int = 255


def _poisson_disk(rng, n, shape_um, min_sep):
    pts = []
    tries = 0
    c = np.array(shape_um) / 2
    rad = np.array(shape_um) * np.array([0.42, 0.45, 0.45])
    while len(pts) < n and tries < 200000:
        tries += 1
        p = c + (rng.random(3) * 2 - 1) * rad
        if np.sum(((p - c) / rad) ** 2) > 1:
            continue
        if pts and np.min(np.linalg.norm(np.array(pts) - p, axis=1)) < min_sep:
            continue
        pts.append(p)
    return np.array(pts)


def _blob_matrix(pos_um, shape, sxy, sz, um_px, um_pl, trunc=3.5, scales=None):
    """Sparse (voxels x neurons) peak-normalised anisotropic Gaussians (optional per-neuron width scale)."""
    P, Y, X = shape
    rows, cols, vals = [], [], []
    sxy0, sz0 = sxy, sz
    for i, p in enumerate(pos_um):
        if scales is not None:
            sxy, sz = sxy0 * scales[i], sz0 * scales[i]
        rz, rxy = int(np.ceil(trunc * sz / um_pl)), int(np.ceil(trunc * sxy / um_px))
        cz, cy, cx = p[0] / um_pl, p[1] / um_px, p[2] / um_px
        zz = np.arange(max(int(cz) - rz, 0), min(int(cz) + rz + 2, P))
        yy = np.arange(max(int(cy) - rxy, 0), min(int(cy) + rxy + 2, Y))
        xx = np.arange(max(int(cx) - rxy, 0), min(int(cx) + rxy + 2, X))
        gz = np.exp(-((zz - cz) * um_pl) ** 2 / (2 * sz ** 2))
        gy = np.exp(-((yy - cy) * um_px) ** 2 / (2 * sxy ** 2))
        gx = np.exp(-((xx - cx) * um_px) ** 2 / (2 * sxy ** 2))
        G = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
        keep = G > 1e-3
        Z, Yi, Xi = np.nonzero(keep)
        rows.append(np.ravel_multi_index((zz[Z], yy[Yi], xx[Xi]), shape))
        cols.append(np.full(len(Z), i))
        vals.append(G[keep])
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(P * Y * X, len(pos_um)))


def _neurite_matrix(pos_um, shape, s_long, s_short, um_px, um_pl, rng, trunc=3.0):
    """Sparse (voxels x neurons) elongated Gaussians starting at each soma along a random 3D direction (peak-normalised)."""
    P, Y, X = shape
    rows, cols, vals = [], [], []
    for i, p in enumerate(pos_um):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        c = p + u * s_long            # centre displaced along the process
        r = int(np.ceil(trunc * s_long))
        zz = np.arange(max(int((c[0] - r) / um_pl), 0), min(int((c[0] + r) / um_pl) + 1, P))
        yy = np.arange(max(int((c[1] - r) / um_px), 0), min(int((c[1] + r) / um_px) + 1, Y))
        xx = np.arange(max(int((c[2] - r) / um_px), 0), min(int((c[2] + r) / um_px) + 1, X))
        if not (len(zz) and len(yy) and len(xx)):
            continue
        Z, Yg, Xg = np.meshgrid(zz * um_pl, yy * um_px, xx * um_px, indexing="ij")
        d = np.stack([Z - c[0], Yg - c[1], Xg - c[2]], -1)
        along = d @ u
        perp2 = np.sum(d ** 2, -1) - along ** 2
        G = np.exp(-along ** 2 / (2 * s_long ** 2) - perp2 / (2 * s_short ** 2))
        keep = G > 1e-3
        zi, yi, xi = np.nonzero(keep)
        rows.append(np.ravel_multi_index((zz[zi], yy[yi], xx[xi]), shape))
        cols.append(np.full(len(zi), i))
        vals.append(G[keep])
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(P * Y * X, len(pos_um)))


def _kernel(rise, decay, dt=FRAME_S, dur=None):
    dur = dur or 8 * decay
    t = np.arange(0, dur, dt)
    k = (1 - np.exp(-t / rise)) * np.exp(-t / decay)
    return k / k.max()


def plane_times(n_volumes: int) -> np.ndarray:
    """(V, 25) acquisition time of each depth-sorted plane under alternating sweeps, 20 ms frames."""
    v = np.arange(n_volumes)[:, None]
    p = np.arange(PLANES)[None, :]
    idx = np.where(v % 2 == 0, p, PLANES - 1 - p)
    return (v * PLANES + idx) * FRAME_S


def simulate(params: WorldParams, animal: str = "syn-0", with_crops: bool = False) -> tuple[dict, dict]:
    w = params
    rng = np.random.default_rng(w.seed)
    um_px, um_pl = raw_traces.UM_PER_PX, raw_traces.UM_PER_PLANE
    shape = tuple(w.shape)
    shape_um = (shape[0] * um_pl, shape[1] * um_px, shape[2] * um_px)
    pos = _poisson_disk(rng, w.n_neurons, shape_um, w.min_sep_um)
    N = len(pos)
    F0 = w.f0_median * np.exp(rng.normal(0, w.f0_logsd, N))
    D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    size_sc = np.exp(rng.normal(0, w.nucleus_size_logsd, N)) if w.nucleus_size_logsd else None
    B = _blob_matrix(pos, shape, w.sigma_xy_um, w.sigma_z_um, um_px, um_pl, scales=size_sc)
    if w.psf_tail_frac:
        Bt = _blob_matrix(pos, shape, w.psf_tail_sigma_xy_um, w.psf_tail_sigma_z_um, um_px, um_pl, trunc=3.0)
        B = (B * (1 - w.psf_tail_frac) + Bt * w.psf_tail_frac).tocsr()
    B_halo = _blob_matrix(pos, shape, w.halo_sigma_um, w.halo_sigma_um * 1.3, um_px, um_pl) if (w.halo_frac or w.evoked_halo_frac) else None
    B_neur = _neurite_matrix(pos, shape, w.neurite_sigma_long_um, w.neurite_sigma_short_um, um_px, um_pl, rng) if w.neurite_halo_frac else None

    T = w.n_volumes
    PT = plane_times(T)
    n_fine = T * PLANES + 1
    tgrid = np.arange(n_fine) * FRAME_S
    # ---- drives on the fine (frame) grid ----
    drive = np.zeros((N, n_fine))
    spk = rng.random((N, n_fine)) < w.spont_rate_hz * FRAME_S
    drive += spk * rng.lognormal(np.log(w.spont_amp), 0.4, (N, n_fine))
    if w.global_amp:
        g = (rng.random(n_fine) < 0.02 * FRAME_S) * w.global_amp
        drive += g[None, :]
    # edges
    gap = np.zeros((N, N), bool)
    chem = np.zeros((N, N), bool)
    if w.p_gap_near:
        pg = w.p_gap_near * np.exp(-D / w.lam_gap_um)
        gap = np.triu(rng.random((N, N)) < pg, 1)
        gap = gap | gap.T
    if w.p_chem:
        pc = np.full((N, N), w.p_chem * 0.05)
        if w.chem_topology_near_bias:
            pc = pc * (1 + w.chem_topology_near_bias * np.exp(-D / 4.0))
        chem = rng.random((N, N)) < np.clip(pc, 0, 1)
        np.fill_diagonal(chem, False)
    label_edges = gap | chem
    if w.chem_topology_near_bias and not w.p_chem:
        # S21: connectome LABELS enriched among near pairs, with no functional effect
        pe = np.clip(0.05 * (1 + w.chem_topology_near_bias * np.exp(-D / 4.0)), 0, 1)
        label_edges = np.triu(rng.random((N, N)) < pe, 1)
        label_edges = label_edges | label_edges.T

    stim_field = np.zeros((N, n_fine))
    events = []
    t_ev = w.first_event_s
    k = 0
    vt = PT.mean(1)
    stim_mult = []
    while t_ev < T * PLANES * FRAME_S - 12:
        tgt = int(rng.integers(N))
        eff = (rng.random() < w.p_target)
        i0 = int(round((t_ev + w.target_delay_s) / FRAME_S))
        if eff:
            a = w.target_amp * rng.lognormal(0, 0.35)
            drive[tgt, i0] += a
            if w.local_field_amp:
                ker = w.local_field_amp * np.exp(-D[tgt] ** 2 / (2 * w.local_field_lam_um ** 2))
                ker[tgt] = 0
                stim_field[:, i0 + 5] += ker * a
        if w.p_offtarget:
            dxy = np.hypot(pos[:, 1] - pos[tgt, 1], pos[:, 2] - pos[tgt, 2])
            light = np.exp(-dxy ** 2 / (2 * w.stim_sigma_xy_um ** 2) - (pos[:, 0] - pos[tgt, 0]) ** 2 / (2 * w.stim_sigma_z_um ** 2))
            light[tgt] = 0
            gate = eff if w.spillover_latent else True
            if gate:
                hit = rng.random(N) < w.p_offtarget * light
                drive[hit, i0] += w.offtarget_amp * rng.lognormal(0, 0.35, hit.sum())
        off_px = rng.normal(0, 0.8, 2)
        events.append(dict(kind="stim", event=k, t_on=float(t_ev), t_off=float(t_ev + 0.5), power_W=0.0012, true_target=tgt, effective=bool(eff),
                           spot_x=float(pos[tgt, 2] / um_px + off_px[0]), spot_y=float(pos[tgt, 1] / um_px + off_px[1])))
        k += 1
        t_ev += w.event_every_s + rng.uniform(-2, 2)

    # one-hop propagation of ALL activity (spontaneous and evoked) through gap / chemical edges
    src = drive.copy()
    if gap.any():
        lat = int(round(w.gap_latency_s / FRAME_S))
        prop = np.zeros_like(src)
        prop[:, lat:] = (gap.astype(float) @ src[:, :n_fine - lat]) * w.gap_coef
        if w.gap_threshold:
            prop = np.where(prop > w.gap_threshold, prop, 0.0)
        drive += prop
    if chem.any():
        lat = int(round(w.chem_latency_s / FRAME_S))
        prop = np.zeros_like(src)
        prop[:, lat:] = (chem.astype(float).T @ src[:, :n_fine - lat]) * w.chem_coef
        drive += prop
    drive += stim_field
    ker = _kernel(w.rise_s, w.decay_s)
    dff = lfilter(ker, [1.0], drive, axis=1)
    if w.local_field_spont:
        # spatially smooth slow field: 6 latent sources with Gaussian footprints
        src = pos[rng.choice(N, 6, replace=False)]
        lat = lfilter([1.0], [1, -0.999], rng.normal(0, 1, (6, n_fine)), axis=1)
        lat = lat / lat.std(1, keepdims=True)
        foot = np.exp(-np.linalg.norm(pos[:, None] - src[None], axis=-1) ** 2 / (2 * w.local_field_lam_um ** 2))
        dff += w.local_field_spont * np.clip(foot @ lat, -1.5, None) * 0.2
    dff = np.clip(dff, -0.5, None)

    # ---- motion (rigid drift, um) ----
    shift = np.zeros((T, 3))
    if w.motion_sd_um:
        e = rng.normal(0, w.motion_sd_um * np.sqrt(1 - w.motion_rho ** 2), (T, 3))
        for v in range(1, T):
            shift[v] = w.motion_rho * shift[v - 1] + e[v]
        shift[:, 0] *= 0.5

    P, Y, X = shape
    npl = Y * X
    Bp = [B[p * npl:(p + 1) * npl] for p in range(P)]
    Hp = [B_halo[p * npl:(p + 1) * npl] for p in range(P)] if B_halo is not None else None
    Np = [B_neur[p * npl:(p + 1) * npl] for p in range(P)] if B_neur is not None else None
    frame_add = rng.normal(0, w.frame_add_sd, (T, P)) if w.frame_add_sd else np.zeros((T, P))
    frame_mult = rng.normal(0, w.frame_mult_sd, (T, P)) if w.frame_mult_sd else np.zeros((T, P))

    # static-registration error of the ANALYSED positions is applied by jittering the rendered world instead of the ROI
    # operators, so segmentation and extraction stay untouched.
    halo_dff = dff
    if w.halo_lowpass_tau_s:
        a_lp = np.exp(-FRAME_S / w.halo_lowpass_tau_s)
        halo_dff = lfilter([1 - a_lp], [1, -a_lp], dff, axis=1)
    lf = None
    if w.lf_contam:
        a = np.exp(-FRAME_S / w.lf_contam_tau_s)
        lf = lfilter([1 - a], [1, -a], dff * F0[:, None], axis=1)

    vols_R, FS, acc, nacc = [], [], np.zeros(shape, np.float64), 0
    motion_cache = {}

    def render(v):
        V = np.empty(shape, np.float32)
        fi = np.clip(np.round(PT[v] / FRAME_S).astype(int), 0, n_fine - 1)
        for p in range(P):
            amp = F0 * (1 + dff[:, fi[p]])
            if w.motion_sd_um and np.any(shift[v] != 0):
                key = tuple(np.round(shift[v] / np.array([um_pl / 2, um_px / 2, um_px / 2])).astype(int))
                if key not in motion_cache:
                    if len(motion_cache) > 400:
                        motion_cache.clear()
                    Bs = _blob_matrix(pos + np.array(key) * np.array([um_pl / 2, um_px / 2, um_px / 2]), shape, w.sigma_xy_um, w.sigma_z_um, um_px, um_pl)
                    motion_cache[key] = [Bs[q * npl:(q + 1) * npl] for q in range(P)]
                img = motion_cache[key][p] @ amp
            else:
                img = Bp[p] @ amp
            if Hp is not None:
                img = img * (1 - w.halo_frac) + Hp[p] @ (w.halo_frac * amp + w.evoked_halo_frac * F0 * halo_dff[:, fi[p]])
            if Np is not None:
                img = img + Np[p] @ (w.neurite_halo_frac * F0 * dff[:, fi[p]])
            sig = np.maximum(img * (1 + frame_mult[v, p]) + w.haze + frame_add[v, p], 0)
            ph = rng.poisson(sig * w.gain) / w.gain
            V[p] = (ph + w.offset + rng.normal(0, w.read_sd, sig.shape)).reshape(Y, X)
            if w.compress_floor is not None:
                q = np.clip(np.floor((V[p] - w.offset - w.compress_floor) / w.compress_scale), 0, w.compress_max)
                V[p] = q * w.compress_scale + w.compress_floor + w.offset
        return V

    # mean volume for segmentation: 200 evenly spaced volumes (same role as raw_traces' 400)
    for v in np.linspace(5, T - 5, 200).astype(int):
        acc += render(v)
        nacc += 1
    M = (acc / nacc).astype(np.float32)
    nuc, thr = raw_traces._segment(M)
    if w.seg_jitter_px:
        jit = np.round(rng.normal(0, w.seg_jitter_px, (len(nuc), 2))).astype(int)
        nuc_an = nuc.copy()
        nuc_an[:, 1:] = np.clip(nuc_an[:, 1:] + jit, 3, np.array([Y - 4, X - 4]))
    else:
        nuc_an = nuc
    ops = RX.roi_operators(nuc_an, shape)
    R = np.full((T, len(nuc_an), len(RX.ROI_NAMES)), np.nan, np.float32)
    FSa = np.full((T, P, len(RX.FRAME_STATS)), np.nan, np.float32)
    nuc_um = nuc_an * np.array([um_pl, um_px, um_px])
    if lf is not None:
        dd = np.linalg.norm(nuc_um[:, None] - pos[None], axis=-1)
        lf_w = w.lf_contam * np.exp(-dd / w.lf_contam_radius_um) * (dd > 1.5)
    for v in range(T):
        V = render(v)
        R[v] = RX.extract_volume(V, ops)
        FSa[v] = RX.frame_stats(V)
        if lf is not None:
            fi = int(round(PT[v, 12] / FRAME_S))
            R[v] += (lf_w @ lf[:, fi]).astype(np.float32)[:, None]

    # detected nucleus <-> true neuron
    dd = np.linalg.norm((nuc * np.array([um_pl, um_px, um_px]))[:, None] - pos[None], axis=-1)
    true_of = np.where(dd.min(1) < 1.2, dd.argmin(1), -1)
    ev_out = []
    for e in events:
        tgt = e.pop("true_target")
        eff = e.pop("effective")
        v = int(np.searchsorted(vt, e["t_on"]))
        dxy = np.hypot(nuc[:, 2] - e["spot_x"], nuc[:, 1] - e["spot_y"])
        zc = pos[tgt, 0] / um_pl
        cand = np.where((dxy <= 4) & (np.abs(nuc[:, 0] - zc) <= 2))[0]
        ti = int(cand[np.argmin(dxy[cand])]) if len(cand) else -1
        e.update(vol=v, target_nucleus=ti, spot_offset_px=[float(e["spot_x"] - nuc[ti, 2]), float(e["spot_y"] - nuc[ti, 1])] if ti >= 0 else [np.nan, np.nan],
                 target_roi_known=True)
        ev_out.append((e, tgt, eff))
    payload = dict(nuclei=nuc, rois=R, frame_stats=FSa, plane_times=PT, plane_depth_um=np.tile(np.arange(P) * um_pl, (T, 1)).astype(np.float32),
                   mean_volume=M, vol_time=vt,
                   meta=dict(sub=animal, version="provenance_synth/1.0", roi_names=RX.ROI_NAMES, frame_stats=RX.FRAME_STATS,
                             n_volumes=T, um_per_px=um_px, um_per_plane=um_pl, events=[e for e, _, _ in ev_out], synthetic=True))
    if with_crops:
        from .randi import event_crops as EC
        payload["crops"] = EC.crops_from_source(render, T, payload["meta"]["events"], nuc, shape)
    truth = dict(params=asdict(w), true_pos_um=pos, F0=F0, true_of_nucleus=true_of, gap=gap, chem=chem, label_edges=label_edges,
                 event_true_target=[t for _, t, _ in ev_out], event_effective=[f for _, _, f in ev_out], seg_threshold=thr)
    return payload, truth


def cohort(base: WorldParams, n_animals: int, seed: int = 0, **overrides) -> list[tuple[str, dict, dict]]:
    out = []
    for a in range(n_animals):
        p = WorldParams(**{**asdict(base), **overrides, "seed": seed * 1000 + a})
        pl, tr = simulate(p, f"syn{seed}-{a}")
        out.append((f"syn{seed}-{a}", pl, tr))
    return out
