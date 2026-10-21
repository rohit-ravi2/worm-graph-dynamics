"""Minimal MATLAB v7.3 (HDF5) cell/struct/char decoding used by source adapters."""
from __future__ import annotations

import h5py
import numpy as np


def char(a) -> str:
    a = np.asarray(a)
    return "".join(map(chr, a.ravel())) if a.dtype.kind in "ui" else ""


def deref(h: h5py.File, ref):
    return h[ref]


def cellstr(h: h5py.File, obj) -> list:
    """Decode a cell array whose elements are char arrays or nested cells of char arrays.
    Returns a list whose items are str or tuple[str, ...] (nested candidates)."""
    out = []
    for r in np.asarray(obj).ravel():
        o = h[r]
        a = np.asarray(o)
        if a.dtype.kind in "ui":
            out.append(char(a) if a.size and not (a.size == 2 and a.ravel().tolist() == [0, 0]) else "")
        elif a.dtype.kind == "O":
            vals = tuple(v for v in (char(h[x]) for x in a.ravel()) if v)
            out.append(vals[0] if len(vals) == 1 else vals)
        else:
            out.append("")
    return out
