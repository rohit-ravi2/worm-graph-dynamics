"""Source-agnostic ROI extraction core shared by real raw volumes and rendered synthetic movies.

Every ROI is a fixed linear operator on a depth-sorted volume (plane, y, x): a sparse row-normalised matrix, so a
trace is ``W @ V.ravel()``. Real data (randi.provenance_traces) and synthetic movies (provenance_synth) call the
same ``roi_operators`` / ``extract_volume`` functions, so hostile synthetic worlds pass through the identical
measurement path.

ROI set per nucleus centre c = (p, y, x) (all offsets clipped to the volume):
  plane_m1, plane_0, plane_p1 : 5x5 mean in planes p-1, p, p+1   (box = mean of the three)
  core                        : 3x3 mean in plane p
  q_ym, q_yp, q_xm, q_xp      : half-box means over planes p-1..p+1 (rows y-2..y-1 / y+1..y+2 x cols x-2..x+2; and
                                the transposed column halves)
  ring                        : plane-p annulus 4 <= r <= 6 px
  wide                        : 3x9x9 mean (planes p-1..p+1)
Per frame (plane): 10th percentile and median over a fixed 4-px grid subsample (frame-common background).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

ROI_NAMES = ("plane_m1", "plane_0", "plane_p1", "core", "q_ym", "q_yp", "q_xm", "q_xp", "ring", "wide")
FRAME_STATS = ("p10", "median")


def _offsets(name: str) -> list[tuple[int, int, int]]:
    r2, r1 = range(-2, 3), range(-1, 2)
    if name in ("plane_m1", "plane_0", "plane_p1"):
        dz = {"plane_m1": -1, "plane_0": 0, "plane_p1": 1}[name]
        return [(dz, a, b) for a in r2 for b in r2]
    if name == "core":
        return [(0, a, b) for a in r1 for b in r1]
    if name == "q_ym":
        return [(dz, a, b) for dz in r1 for a in (-2, -1) for b in r2]
    if name == "q_yp":
        return [(dz, a, b) for dz in r1 for a in (1, 2) for b in r2]
    if name == "q_xm":
        return [(dz, a, b) for dz in r1 for a in r2 for b in (-2, -1)]
    if name == "q_xp":
        return [(dz, a, b) for dz in r1 for a in r2 for b in (1, 2)]
    if name == "ring":
        return [(0, a, b) for a in range(-6, 7) for b in range(-6, 7) if 4 <= np.hypot(a, b) <= 6]
    if name == "wide":
        return [(dz, a, b) for dz in r1 for a in range(-4, 5) for b in range(-4, 5)]
    raise KeyError(name)


def roi_operators(nuclei: np.ndarray, shape: tuple[int, int, int]) -> dict[str, sparse.csr_matrix]:
    """Row-normalised sparse averaging operators, one per ROI name, shape (n_nuclei, prod(shape))."""
    P, Y, X = shape
    ops = {}
    for name in ROI_NAMES:
        off = np.array(_offsets(name))
        rows, cols = [], []
        for i, c in enumerate(np.asarray(nuclei, int)):
            q = c[None, :] + off
            ok = (q[:, 0] >= 0) & (q[:, 0] < P) & (q[:, 1] >= 0) & (q[:, 1] < Y) & (q[:, 2] >= 0) & (q[:, 2] < X)
            q = q[ok]
            rows.append(np.full(len(q), i))
            cols.append(np.ravel_multi_index((q[:, 0], q[:, 1], q[:, 2]), shape))
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        M = sparse.csr_matrix((np.ones(len(rows), np.float64), (rows, cols)), shape=(len(nuclei), P * Y * X))
        n = np.asarray(M.sum(1)).ravel()
        ops[name] = sparse.diags(1.0 / np.maximum(n, 1)) @ M
    return ops


def overlap_fraction(nuclei: np.ndarray, shape, name: str = "box") -> sparse.csr_matrix:
    """Fraction of receiver i's ROI voxels that also belong to source j's ROI (shared-voxel extraction overlap)."""
    if name == "box":
        W = sum(roi_operators(nuclei, shape)[k] for k in ("plane_m1", "plane_0", "plane_p1"))
    else:
        W = roi_operators(nuclei, shape)[name]
    B = (W > 0).astype(np.float64)
    inter = B @ B.T
    n = np.asarray(B.sum(1)).ravel()
    return sparse.diags(1.0 / np.maximum(n, 1)) @ inter


def extract_volume(V: np.ndarray, ops: dict) -> np.ndarray:
    """(n_nuclei, len(ROI_NAMES)) float32 ROI means for one depth-sorted volume."""
    v = V.ravel().astype(np.float64)
    return np.stack([ops[k] @ v for k in ROI_NAMES], 1).astype(np.float32)


def frame_stats(V: np.ndarray, step: int = 4) -> np.ndarray:
    """(planes, 2): 10th percentile and median per plane over a fixed grid subsample."""
    S = V[:, ::step, ::step].reshape(V.shape[0], -1)
    return np.stack([np.percentile(S, 10, axis=1), np.median(S, axis=1)], 1).astype(np.float32)


def box_trace(R: np.ndarray) -> np.ndarray:
    """3x5x5 box mean from the three plane ROIs (last axis ROI_NAMES order)."""
    return R[..., :3].mean(-1)
