"""g3 provider: provenance_traces/1.0 + event_crops/1.0 (sealed per partition) -> signal-provenance and unmixing tables.

Payloads are opened only inside ``table()`` with a granted broker entry; every file's sha256 is checked against its
extraction SEAL.jsonl first. Per-animal processing runs in worker processes; results live in memory only.
Prior g0-g2 outcomes are imported (CONFIRMATION/RESERVE UNTOUCHED regraded PARTIAL, as in g2).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import observations as O
from .. import signal_provenance as SP
from .. import unmixing as U
from ..engine import DataProvider
from ..spec import TestOutcome
from . import event_crops as EC
from . import partitions, provenance_traces as PT
from .provider import RandiProvider, SealViolation, _sha

PROV_ADDENDUM = partitions.BENCH_DIR / "g3_provenance_manifest.json"
CROPS_ADDENDUM = partitions.BENCH_DIR / "g3_event_crops_manifest.json"


def _base(part):
    return partitions.BENCH_DIR / ("discovery" if part == "DISCOVERY" else f"sealed/{part.lower()}")


def _seals(d):
    f = d / "SEAL.jsonl"
    if not f.exists():
        return {}
    return {r["sub"]: r["sha256"] for r in (json.loads(l) for l in f.read_text().splitlines()) if r["status"] in ("ok", "exists")}


def process_animal(args) -> dict:
    """All g3 per-animal products (worker process)."""
    sub, part, conn = args
    base = _base(part)
    pv = PT.load(base / "provenance" / f"{sub}.npz")
    cr = EC.load(base / "event_crops" / f"{sub}.npz")
    payload = O.load_payload(base / "raw_traces" / f"{sub}.npz")
    if pv["meta"]["extractor_sha256"] != json.loads(PROV_ADDENDUM.read_text())["extractor"]["source_sha256"]:
        raise SealViolation(f"{sub}: provenance produced by a different extractor")
    if cr["meta"]["extractor_sha256"] != json.loads(CROPS_ADDENDUM.read_text())["extractor"]["source_sha256"]:
        raise SealViolation(f"{sub}: event_crops produced by a different extractor")
    light = SP.fit_light_model(pv["mean_volume"], pv["nuclei"])
    ap = SP.prepare_animal(sub, pv, light=light)
    rows = SP.pair_rows(ap)
    spont = SP.spontaneous_pairs(ap)
    crops = {k: v for k, v in cr.items() if k.startswith("e")}
    um = unmix = U.unmix_animal(sub, pv, crops, light=light)
    # labels/connectome and spot offsets for unmixed rows
    labels = [str(x) for x in payload["labels"]]
    evs = {int(e["event"]): e for e in pv["meta"]["events"] if e.get("kind") == "stim"}
    extra = []
    um_s = np.array([pv["meta"]["um_per_plane"], pv["meta"]["um_per_px"], pv["meta"]["um_per_px"]])
    P_um = pv["nuclei"] * um_s
    density = np.array([(np.linalg.norm(P_um - P_um[i], axis=1) < 6.0).sum() - 1 for i in range(len(P_um))])
    for r in unmix["rows"]:
        t, n = int(r["target"]), int(r["receiver"])
        lt, ln = labels[t], labels[n]
        lab = bool(conn is not None and lt in conn.index and ln in conn.index)
        gw = float(conn.gap[conn.index[ln], conn.index[lt]] + conn.gap[conn.index[lt], conn.index[ln]]) if lab else np.nan
        gap = bool(lab and gw > 0)
        chem = bool(lab and (conn.chem[conn.index[ln], conn.index[lt]]) > 0)
        gdeg_t = float(((conn.gap[conn.index[lt]] + conn.gap[:, conn.index[lt]]) > 0).sum()) if (conn is not None and lt in conn.index) else np.nan
        gdeg_n = float(((conn.gap[conn.index[ln]] + conn.gap[:, conn.index[ln]]) > 0).sum()) if (conn is not None and ln in conn.index) else np.nan
        e = evs.get(int(r["event"]), {})
        off = np.array(e.get("spot_offset_px", [np.nan, np.nan]), float)
        v = (pv["nuclei"][n] - pv["nuclei"][t]).astype(float)
        dxy = np.hypot(v[1], v[2])
        offn = float(np.hypot(*off)) if np.all(np.isfinite(off)) else np.nan
        cos = float((v[2] * off[0] + v[1] * off[1]) / (dxy * offn)) if np.isfinite(offn) and offn > 0 and dxy > 0 else np.nan
        dv = (P_um[n] - P_um[t])
        homolog = bool(lt and ln and lt != ln and lt[-1] in "LR" and ln[-1] in "LR" and lt[:-1] == ln[:-1])
        extra.append((lab, gap, offn, cos, chem, gw, float(abs(dv[0])), float(np.hypot(dv[1], dv[2])), float(density[t]), bool(lt and ln), homolog,
                      gdeg_t, gdeg_n))
    unmix["extra"] = np.array(extra, dtype=[("labeled", "?"), ("gap", "?"), ("offset_px", "f8"), ("cos_offset", "f8"), ("chem", "?"),
                                            ("gap_weight", "f8"), ("dz_um", "f8"), ("dxy_um", "f8"), ("target_density", "f8"),
                                            ("both_named", "?"), ("lr_homolog", "?"), ("gap_degree_target", "f8"), ("gap_degree_receiver", "f8")])
    meas = dict(sigma_xy_px=light["sigma_xy_px"], sigma_z_planes=light["sigma_z_planes"], r2_support=light["r2_support"], n_isolated=light["n_isolated"],
                n_nuclei=int(len(pv["nuclei"])), n_events=len(ap.amps["events"]))
    light.pop("G", None)
    desc = animal_descriptives(sub, pv, ap, rows)
    return dict(sub=sub, rows=rows, spont=spont, unmix=unmix, measurement=meas, desc=desc)


def animal_descriptives(sub, pv, ap, rows) -> dict:
    """Per-animal W/P6/O4 products (exploratory)."""
    from .. import waveform_provenance as WP
    out = {}
    try:
        out["W1W2W8"] = WP.w1_w2_w8_pairs(ap)
        out["W3"] = WP.w3_waveforms(ap, rows)
        out["W6coh"] = WP.w6_coherence(ap)
    except Exception as e:
        out["W1W3W6_error"] = repr(e)
    try:
        out["W5_plane"] = SP.w5_lag(ap, rows)
        out["W5_naive"] = SP.w5_lag(ap, rows, use_plane_times=False)
        out["W6"] = SP.band_transfer(ap)
    except Exception as e:
        out["W_error"] = repr(e)
    # P6: near/far responders by ROI type (robust z within ROI type), TARGET-UP by box target z
    try:
        evs = ap.amps["events"]
        box = ap.amps["rois"]["box"]
        zbox = SP.robust_z((box["post"] - box["base"]) / box["base"])
        tg = np.array([int(e["target_nucleus"]) for e in evs])
        res = {}
        for roi in ("core", "box", "wide", "ring"):
            A = ap.amps["rois"][roi]
            z = SP.robust_z((A["post"] - A["base"]) / A["base"])
            cnt = dict(near_resp=0, near_n=0, far_resp=0, far_n=0)
            for i, t in enumerate(tg):
                if not zbox[i, t] > SP.Z_RESP:
                    continue
                d = ap.kap["dist"][:, t]
                nm = (d > 0) & (d < 6.0) & np.isfinite(z[i])
                fm = (d >= 15.0) & (d < 40.0) & np.isfinite(z[i])
                cnt["near_resp"] += int((z[i][nm] > SP.Z_RESP).sum()); cnt["near_n"] += int(nm.sum())
                cnt["far_resp"] += int((z[i][fm] > SP.Z_RESP).sum()); cnt["far_n"] += int(fm.sum())
            res[roi] = cnt
        out["P6_counts"] = res
    except Exception as e:
        out["P6_error"] = repr(e)
    # O4: Leifer tracked-coordinate displacement around events (post v+1..v+12 minus pre v-8..v-1), by distance from target
    try:
        from . import leifer_osf as LO
        if LO.folder_for(sub) is not None:
            o = LO.load(sub)
            l2r = LO.align_by_timestamps(sub, pv["vol_time"])
            r2l = {}
            for k, v in enumerate(l2r):
                r2l.setdefault(int(v), k)
            tr = o["tracked_raw_zyx"] * np.array([pv["meta"]["um_per_plane"], pv["meta"]["um_per_px"], pv["meta"]["um_per_px"]])
            med = np.nanmedian(o["tracked_raw_zyx"], 0)
            nuc = pv["nuclei"].astype(float)
            dd = np.linalg.norm((med[:, None, :] - nuc[None]) * np.array([1.22, 0.42, 0.42]), axis=-1)
            ref_nuc = np.where(dd.min(1) < 1.5, dd.argmin(1), -1)
            near_d, far_d, tgt_d = [], [], []
            for e in evs:
                v = int(e["vol"])
                if v not in r2l:
                    continue
                k = r2l[v]
                if k - 8 < 0 or k + 13 > len(tr):
                    continue
                disp = np.nanmean(tr[k + 1:k + 13], 0) - np.nanmean(tr[k - 8:k], 0)
                mag = np.linalg.norm(disp, axis=1)
                t = int(e["target_nucleus"])
                for j, nidx in enumerate(ref_nuc):
                    if nidx < 0 or not np.isfinite(mag[j]):
                        continue
                    dist = ap.kap["dist"][nidx, t]
                    (tgt_d if nidx == t else near_d if dist < 6 else far_d if dist >= 15 else []).append(float(mag[j]))
            out["O4"] = dict(n_ref_matched=int((ref_nuc >= 0).sum()), target_median_um=float(np.median(tgt_d)) if tgt_d else None,
                             near_median_um=float(np.median(near_d)) if near_d else None, far_median_um=float(np.median(far_d)) if far_d else None,
                             n_target=len(tgt_d), n_near=len(near_d), n_far=len(far_d),
                             tracked_jitter_p90_px=[float(np.nanpercentile(np.abs(o["tracked_raw_zyx"] - med)[..., i], 90)) for i in range(3)])
    except Exception as e:
        out["O4_error"] = repr(e)
    return out


class RandiProvenanceProvider(DataProvider):
    def __init__(self, manifest: dict, prior_outcomes: list[dict], workers: int = 6):
        self.manifest, self.prior, self.workers = manifest, prior_outcomes, workers
        self.base = RandiProvider(manifest)
        self._cache: dict = {}

    def units(self, partition: str) -> list[str]:
        if partition not in ("DISCOVERY", "CONFIRMATION", "RESERVE"):
            return []
        b = _base(partition)
        ps, cs = _seals(b / "provenance"), _seals(b / "event_crops")
        return [s for s in self.manifest["partitions"][partition] if s in ps and s in cs]

    def data_stream(self) -> str:
        p = json.loads(PROV_ADDENDUM.read_text())["extractor"]["source_sha256"][:12]
        c = json.loads(CROPS_ADDENDUM.read_text())["extractor"]["source_sha256"][:12]
        return f"provenance_traces/1.0:{p}+event_crops/1.0:{c}"

    def table(self, partition: str, grant: dict) -> dict:
        if not grant.get("granted") or grant.get("partition") != partition:
            raise PermissionError("table() requires a granted broker entry for this partition")
        if partition in self._cache:
            return self._cache[partition]
        b = _base(partition)
        ps, cs = _seals(b / "provenance"), _seals(b / "event_crops")
        subs = self.units(partition)
        for s in subs:
            if _sha(b / "provenance" / f"{s}.npz") != ps[s] or _sha(b / "event_crops" / f"{s}.npz") != cs[s]:
                raise SealViolation(f"{partition}/{s}: payload does not match its extraction seal")
        conn = O.Connectome.load_wormneuroatlas()
        with ProcessPoolExecutor(self.workers) as ex:
            res = list(ex.map(process_animal, [(s, partition, conn) for s in subs]))
        self._cache[partition] = assemble(res)
        return self._cache[partition]

    def imported_outcomes(self):
        out = [o for o in self.base.imported_outcomes()]
        for o in self.prior:
            d = dict(o)
            d["details"] = dict(d.get("details", {}), data_stream=d.get("details", {}).get("data_stream") or "raw_traces/1.0",
                                imported_from_generation=d.get("details", {}).get("imported_from_generation", "prior"),
                                regraded="inspected before this generation existed: PARTIAL at most")
            if d["partition"] in ("CONFIRMATION", "RESERVE") and d["contamination_grade"] == "UNTOUCHED":
                d["contamination_grade"] = "PARTIAL"
            d.pop("outcome_id", None)
            out.append(TestOutcome(**d).finalize())
        return out


def assemble(res: list[dict]) -> dict:
    rows = SP.concat_rows([r["rows"] for r in res])
    spont = SP.concat_rows([r["spont"] for r in res])
    unm = [r["unmix"] for r in res]
    tab = U.response_table(unm)
    # attach labels / spot geometry to the response table by (animal, event, receiver)
    key = {}
    for u in unm:
        for r, x in zip(u["rows"], u["extra"]):
            key[(u["animal"], int(r["event"]), int(r["receiver"]))] = x
    ex = [key.get((a, int(e), int(n))) for a, e, n in zip(tab.get("animal", []), tab.get("event", []), tab.get("receiver", []))]
    for f in ("labeled", "gap", "offset_px", "cos_offset"):
        tab[f] = np.array([x[f] if x is not None else (False if f in ("labeled", "gap") else np.nan) for x in ex])
    for f in ("chem", "gap_weight", "target_density", "both_named", "lr_homolog", "gap_degree_target", "gap_degree_receiver"):
        tab[f] = np.array([x[f] if x is not None else (False if f in ("chem", "both_named", "lr_homolog") else np.nan) for x in ex])
    desc = {r["sub"]: r.get("desc", {}) for r in res}
    roi_rd = {}
    for roi in ("core", "box", "wide", "ring"):
        c = [d.get("P6_counts", {}).get(roi) for d in desc.values() if d.get("P6_counts")]
        if c:
            nr, nn, fr, fn = (sum(x[k] for x in c) for k in ("near_resp", "near_n", "far_resp", "far_n"))
            roi_rd[roi] = dict(near_rate=nr / max(nn, 1), far_rate=fr / max(fn, 1), rd=nr / max(nn, 1) - fr / max(fn, 1), near_n=nn, far_n=fn)
    W = dict(W2=SP.w2_alpha_stability(rows) if rows else None, W7=SP.w7_amplitude_linearity(rows) if rows else None,
             W5_plane_median=_med([d.get("W5_plane", {}).get("best_lag_s") for d in desc.values()]),
             W5_naive_median=_med([d.get("W5_naive", {}).get("best_lag_s") for d in desc.values()]),
             W6_bands=[d.get("W6", {}).get("bands") for d in desc.values()],
             W10_reciprocal=reciprocal_pairs(rows) if rows else None,
             W1_W2_W8=_wp_summary([d["W1W2W8"] for d in desc.values() if d.get("W1W2W8")]),
             W3=_w3([p for d in desc.values() for p in d.get("W3", [])]),
             W6_coherence=[d.get("W6coh") for d in desc.values() if d.get("W6coh")],
             W4_shared_voxel=shared_voxel_noise(spont) if spont else None)
    disp = {k: d.get("O4") for k, d in desc.items() if d.get("O4")}
    return dict(animal=np.array([r["sub"] for r in res]), rows=rows, spont=spont, unmix=tab, halo=U.halo_profile(unm),
                measurement={r["sub"]: r["measurement"] for r in res}, desc_roi_rd=roi_rd, desc_W=W, desc_displacement=disp)


def _wp_summary(per):
    from .. import waveform_provenance as WP
    return WP.w1_summary(per) if per else None


def _w3(pairs):
    from .. import waveform_provenance as WP
    return WP.w3_summary(pairs) if pairs else None


def _med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else None


def reciprocal_pairs(rows: dict) -> dict:
    """W10: pairs (A, B) where A was a responding target with B near AND B was a responding target with A near."""
    m = (rows["target_z"] > SP.Z_RESP) & (rows["dist_um"] < 6.0) & (rows["dC_t"] > 0)
    d = {}
    for a, t, r, dn, dt, ka, fn, ft in zip(rows["animal"][m], rows["target"][m], rows["receiver"][m], rows["dC_n"][m], rows["dC_t"][m], rows["kappa"][m],
                                           rows["own_n"][m], rows["F0_t"][m]):
        d.setdefault((a, int(t), int(r)), []).append((dn / dt, ka, fn))
    out = []
    for (a, t, r), v in d.items():
        if t < r and (a, r, t) in d:
            ab, ba = np.median([x[0] for x in v]), np.median([x[0] for x in d[(a, r, t)]])
            out.append(dict(ratio_ab=float(ab), ratio_ba=float(ba), kappa_ab=float(v[0][1]), kappa_ba=float(d[(a, r, t)][0][1])))
    if not out:
        return dict(n_pairs=0)
    lr = [np.log(abs(x["ratio_ab"]) + 1e-6) - np.log(abs(x["ratio_ba"]) + 1e-6) for x in out]
    lk = [np.log(x["kappa_ab"] + 1e-6) - np.log(x["kappa_ba"] + 1e-6) for x in out]
    return dict(n_pairs=len(out), spearman_logratio_vs_logkappa=SP._spearman(np.array(lk), np.array(lr)) if len(out) >= 5 else None)


def shared_voxel_noise(sp: dict) -> dict:
    """W4: zero-lag minus lag-1 cross-covariance (white shared noise) vs exact ROI voxel overlap."""
    white = sp["cov0_obs"] - sp["cov1_obs"]
    ov = sp["overlap"] > 0
    return dict(n_overlap_pairs=int(ov.sum()), mean_white_overlap=float(np.mean(white[ov])) if ov.any() else None,
                mean_white_no_overlap_near=float(np.mean(white[(~ov) & (sp["dist_um"] < 6)])) if ((~ov) & (sp["dist_um"] < 6)).any() else None,
                mean_white_far=float(np.mean(white[sp["dist_um"] >= 15])) if (sp["dist_um"] >= 15).any() else None)
