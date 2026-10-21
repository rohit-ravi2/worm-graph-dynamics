"""Atanas, Kim et al. 2023, Cell 186:4134 (Flavell lab): freely moving whole-brain imaging, NLS-GCaMP7f x NeuroPAL.

Source: DANDI:000776, pinned to published version 0.241009.1509 (38 animals, ~27 GB each because the raw light-sheet
volumes are embedded). Only the processed groups are streamed (HTTP range reads; no full download):
calcium traces + raw fluorescence, ROI identity labels, per-ROI aligned coordinates (labelled ROIs only), and the
behaviour channels. Each extracted file records the DANDI asset sha256 it came from; the cache manifest records the
sha256 of every extracted file and of this module, so a cache can be verified or rebuilt.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import h5py
import numpy as np

from .. import identity as I
from ..config import load_config
from ..fetch import sha256_file
from ..schema import Recording

DATASET = "atanas2023"
DANDISET = "000776"
VERSION = "0.241009.1509"
API = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/"
EXTRACTOR_VERSION = "atanas2023-extract/1.0"
BEHAVIOR = ("velocity", "reversal_events", "head_curvature", "angular_velocity", "pumping", "body_curvature",
            "forwardness", "dorsalness", "feedingness")
VOXEL_UM = 0.54


def _dir() -> Path:
    return load_config().cache_dir / DATASET


def module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def asset_index(refresh: bool = False) -> list[dict]:
    """[{subject, path, asset_id, size, sha256}] for the pinned version (cached as JSON)."""
    f = _dir() / "asset_index.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text())
    out, url = [], API + "?page_size=100"
    while url:
        d = json.load(urllib.request.urlopen(url, timeout=120))
        for a in d["results"]:
            m = json.load(urllib.request.urlopen(API + a["asset_id"] + "/", timeout=120))
            out.append(dict(subject=a["path"].split("/")[0], path=a["path"], asset_id=a["asset_id"], size=a["size"],
                            sha256=m["digest"]["dandi:sha2-256"],
                            s3=[u for u in m["contentUrl"] if "s3" in u][0]))
        url = d.get("next")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1))
    return out


def _dec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def extract(asset: dict) -> Path:
    """Stream the processed groups of one asset into <cache>/atanas2023/<subject>.h5."""
    import remfile
    outp = _dir() / f"{asset['subject']}.h5"
    if outp.exists():
        return outp
    h = h5py.File(remfile.File(asset["s3"]), "r")
    ca = h["processing/CalciumActivity"]
    resp = ca["SignalCalciumImResponseSeries"]
    seg = ca["CalciumSeriesSegmentation/Aligned_neuron_coordinates"]
    vm = seg["voxel_mask"][:]
    tmp = outp.with_suffix(".h5.part")
    with h5py.File(tmp, "w") as o:
        o.attrs.update(dict(subject=asset["subject"], source=f"DANDI:{DANDISET}/{VERSION}/{asset['path']}",
                            source_sha256=asset["sha256"], extractor=EXTRACTOR_VERSION, extractor_sha256=module_sha256(),
                            trace_description=_dec(resp.attrs.get("description", "")), voxel_um=VOXEL_UM))
        o.create_dataset("traces", data=resp["data"][:], compression="gzip")
        o.create_dataset("raw_fluor", data=ca["SignalRawFluor/SignalCalciumImResponseSeries/data"][:], compression="gzip")
        o.create_dataset("t", data=resp["timestamps"][:])
        o.create_dataset("rois", data=resp["rois"][:])
        o.create_dataset("labels", data=np.array([_dec(x) for x in seg["ID_labels"][:]], dtype=h5py.string_dtype()))
        o.create_dataset("voxel_xyz", data=np.stack([vm["x"], vm["y"], vm["z"]], 1).astype(np.int32))
        o.create_dataset("voxel_weight", data=vm["weight"].astype(np.float32))
        bg = o.create_group("behavior")
        for k in BEHAVIOR:
            p = f"processing/Behavior/{k}/{k}"
            if p in h:
                gg = bg.create_group(k)
                gg.create_dataset("data", data=h[p]["data"][:])
                gg.create_dataset("timestamps", data=h[p]["timestamps"][:])
    h.close()
    tmp.rename(outp)
    return outp


def build_cache(workers: int = 4) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    idx = asset_index()
    with ThreadPoolExecutor(workers) as ex:
        paths = list(ex.map(extract, idx))
    man = dict(dataset=DATASET, dandiset=DANDISET, version=VERSION, extractor=EXTRACTOR_VERSION,
               extractor_sha256=module_sha256(),
               files={p.name: dict(sha256=sha256_file(p), source_sha256=a["sha256"]) for p, a in zip(paths, idx)})
    (_dir() / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return man


def verify_cache() -> bool:
    man = json.loads((_dir() / "MANIFEST.json").read_text())
    for name, rec in man["files"].items():
        if sha256_file(_dir() / name) != rec["sha256"]:
            raise RuntimeError(f"{name} changed since the manifest was written")
    return True


def _to_grid(t_src, x, t):
    """Behaviour on the neural time base: exact if identical, else nearest-sample (labelled in provenance)."""
    if len(t_src) == len(t) and np.allclose(t_src, t):
        return np.asarray(x, float), "identical"
    j = np.clip(np.searchsorted(t_src, t), 1, len(t_src) - 1)
    j = np.where(np.abs(t_src[j - 1] - t) <= np.abs(t_src[j] - t), j - 1, j)
    return np.asarray(x, float)[j], "nearest"


def load(subjects=None) -> list[Recording]:
    idx = asset_index()
    recs = []
    for a in idx:
        if subjects and a["subject"] not in subjects:
            continue
        p = extract(a)
        with h5py.File(p, "r") as f:
            X = f["traces"][:]
            t = f["t"][:]
            labs = [I.normalize(_dec(x)) for x in f["labels"][:]]
            w = f["voxel_weight"][:]
            pos = np.where(w[:, None] > 0, f["voxel_xyz"][:] * VOXEL_UM, np.nan).astype(float)
            beh, native, how = {}, {}, {}
            for k in f["behavior"]:
                d, ts = f[f"behavior/{k}/data"][:], f[f"behavior/{k}/timestamps"][:]
                native[k] = (ts, d)
                beh[k], how[k] = _to_grid(ts, d, t)
            prov = dict(f.attrs)
        mask = np.isfinite(X)
        X = np.where(mask, X, 0.0)
        prov = {k: (v.item() if hasattr(v, "item") else v) for k, v in prov.items()}
        prov["behavior_alignment"] = how
        recs.append(Recording(DATASET, a["subject"], "freely_moving", t, X, mask, labs,
                              "fluorescence as deposited (DANDI description: 'Raw calcium fluorescence activity')",
                              behavior=beh, behavior_native=native, positions_um=pos, provenance=prov).validate())
    return recs
