"""Kato et al. 2015, Cell 163:656 (Zimmer lab): immobilised whole-brain calcium imaging, NLS-GCaMP5K.

Source: OSF node 2395t ("Kato2015 whole brain imaging data"); sha256 values are OSF's own recorded hashes and were
matched against an independent local copy on 2026-10-08.

Arms: WT_NoStim (5 worms, 8 manually annotated states), WT_Stim (7 worms, O2 21%/4% binary steps), AVA_HisCl (5 worms,
AVA silenced by HisCl1 + 10 mM histamine for the whole session; a true node-ablation intervention).
Signal: bleach-corrected dF/F0 (``deltaFOverF_bc`` in WT_NoStim, ``traces`` in the other two arms, per the readme).
"""
from __future__ import annotations

import h5py
import numpy as np

from .. import identity as I
from ..fetch import ensure_file
from ..matv73 import cellstr, char
from ..schema import Event, Recording

DATASET = "kato2015"
OSF_NODE = "2395t"
MANIFEST = {
    "WT_NoStim.mat": ("https://osf.io/download/na4f9/", "7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a"),
    "WT_Stim.mat": ("https://osf.io/download/ncbrk/", "96ffc39afbc90b66ff1b4c98de199f8424391450d4a8050cbe2b65b3a2fdaa72"),
    "AVA_HisCl.mat": ("https://osf.io/download/4wh9u/", "54bc05fb4cca861e4f425d01c41f3c5afa098d8247e21bf8d026945a21b51805"),
    "readme_Kato2015.txt": ("https://osf.io/download/y8a6e/", "9fbc04b06a5c59d13a7ce6a90a428bc49e2d4ef5a2ff21bd2efa6f67655bc3d6"),
}
ARMS = ("WT_NoStim", "WT_Stim", "AVA_HisCl")
# readme order for the 4-state arms; AVA_HisCl carries its own States_key, which is checked against this
STATE_CODES_4 = ("FWD", "REV", "REVSUS", "TURN")


def _labels(names) -> list[I.Label]:
    out = []
    for nm in names:
        if isinstance(nm, tuple):
            out.append(I.normalize("", extra_candidates=nm))
        elif nm.strip().isdigit():
            out.append(I.Label(nm, "unknown", None))
        else:
            out.append(I.normalize(nm))
    return out


def _state_events(t, codes, names) -> tuple[list[Event], np.ndarray]:
    ev, cur, start = [], None, None
    for k, c in enumerate(list(codes) + [None]):
        if c != cur:
            if cur is not None and cur == cur:
                ev.append(Event("state", names[int(cur)], float(t[start]), float(t[k - 1])))
            cur, start = c, k
    return ev, np.asarray(codes)


def load(arm: str, *, download: bool = True) -> list[Recording]:
    assert arm in ARMS, arm
    url, sha = MANIFEST[arm + ".mat"]
    path = ensure_file(DATASET, arm + ".mat", url, sha, download=download)
    recs = []
    with h5py.File(path, "r") as h:
        g = h[arm]
        n = g["dataset"].shape[0]
        for i in range(n):
            get = lambda k: np.asarray(h[g[k][i, 0]])
            name = char(get("dataset"))
            prov = dict(source=f"OSF:{OSF_NODE}/{arm}.mat", sha256=sha, index=i, dataset_name=name)
            if arm == "WT_NoStim":
                X = get("deltaFOverF_bc").T
                t = get("tv").ravel()
                names = cellstr(h, h[g["NeuronNames"][i, 0]])
                st = h[g["States"][i, 0]]
                keys = sorted(st.keys())
                M = np.stack([np.asarray(st[k]).ravel() for k in keys], 1)
                codes = np.where(M.sum(1) > 0, M.argmax(1), -1).astype(float)
                codes[codes < 0] = np.nan
                state_names = [k.upper() for k in keys]
                prov["state_source"] = "one-hot indicator vectors (KNOWN)"
                signal = "dF/F0, bleach-corrected (deltaFOverF_bc)"
            else:
                X = get("traces").T
                t = get("timeVectorSeconds").ravel()
                names = cellstr(h, h[g["IDs"][i, 0]])
                codes = get("States").ravel().astype(float) - 1.0
                state_names = list(STATE_CODES_4)
                prov["state_source"] = ("readme order FWD,REV,REVSUS,TURN; INFERRED_FROM_DATA: AVA z by state matches WT_NoStim "
                                        "(FWD -1.04, REVSUS +1.0), checked 2026-10-08")
                if "States_key" in g:
                    prov["states_key_raw"] = cellstr(h, h[g["States_key"][i, 0]])
                    prov["state_source"] = ("CONFLICTED: States_key (0=REVERSAL,1=FORWARD,2=QUIESCENCE,3=TURN) contradicts "
                                            "reverse-module (RIM/AIB/AVE) activity under code k -> key[k-1]; readme order "
                                            "FWD,REV,REVSUS,TURN used because it reproduces the WT_Stim activity pattern. "
                                            "Do not use AVA_HisCl state labels as ground truth.")
                signal = "dF/F0, bleach-corrected (traces)"
            T = min(len(t), X.shape[0], len(codes))
            X, t, codes = X[:T], t[:T], codes[:T]
            mask = np.isfinite(X)
            X = np.where(mask, X, 0.0)
            ev, _ = _state_events(t, codes, state_names)
            if arm == "WT_Stim":
                s = h[g["stimulus"][i, 0]]
                sw = np.asarray(s["switchtimes"]).ravel()
                conc = np.asarray(s["conc"]).ravel()
                init = int(np.asarray(s["initialstate"]).ravel()[0]) - 1
                ident = char(s["identity"])
                level = init
                for k, ts in enumerate(sw):
                    level = 1 - level
                    stop = sw[k + 1] if k + 1 < len(sw) else float(t[-1])
                    ev.append(Event("stimulus", f"{ident}_{conc[level]:g}{char(s['concunits'])}", float(ts), float(stop),
                                    dict(identity=ident, conc=float(conc[level]))))
            if arm == "AVA_HisCl":
                ev.append(Event("intervention", "AVA_silenced_HisCl1_10mM_histamine", float(t[0]), float(t[-1]),
                                dict(target_class="AVA", kind="node_silencing", whole_session=True)))
            rec = Recording(DATASET, f"{arm}_{i}", arm, t, X, mask, _labels(names), signal,
                            behavior={"state_code": codes}, events=ev, provenance=dict(prov, state_names=state_names))
            recs.append(rec.validate())
    return recs
