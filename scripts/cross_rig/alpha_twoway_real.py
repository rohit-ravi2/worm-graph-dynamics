"""Corrected copy-ratio check on REAL DISCOVERY animals (exploratory; never reads sealed partitions).

The g3 W1/W2 estimator compared alpha = slope(D_n on D_t) with the one-way kernel kappa_nt. Under known pure contamination
that ratio is biased upward by reverse leakage (receiver light in the target box; scripts/cross_rig/alpha_bias_check.py).
Here alpha is compared with the two-way prediction (kappa_nt var_t + kappa_tn var_n) / var_t, using exactly the
g3 residuals (waveform_provenance._residuals) and the same pair selection (stimulated targets, receivers < 6 um,
kappa_nt > 0.02), per window. Ratios near 1 = measured transfer equals the static-kernel contamination prediction;
< 1 = less transfer than the kernel predicts; > 1 = excess transfer.
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from wormlab.mechanism import signal_provenance as SP, waveform_provenance as WP
from wormlab.mechanism.randi import partitions as P, provenance_traces as PT

REPO = Path(__file__).resolve().parents[2]


def base_dir(part):
    return P.BENCH_DIR / ("discovery" if part == "DISCOVERY" else f"sealed/{part.lower()}")


def one(sub, part="DISCOVERY"):
    pv = PT.load(base_dir(part) / "provenance" / f"{sub}.npz")
    ap = SP.prepare_animal(sub, pv)
    R = WP._residuals(ap)
    masks = WP._window_masks(ap.prov)
    K = ap.kap["kappa"]["box"]; dist = ap.kap["dist"]
    tg = sorted({int(e["target_nucleus"]) for e in SP.event_windows(ap.prov)})
    out = {w: [] for w in masks}
    for w, m in masks.items():
        D = WP._diff_pairs(R, m)
        if len(D) < 20:
            continue
        v = np.nanvar(D, 0)
        for t in tg:
            for n in np.where((dist[:, t] < 6) & (np.arange(len(dist)) != t))[0]:
                if K[n, t] <= 0.02 or not (v[t] > 0):
                    continue
                a, b = D[:, t], D[:, n]; ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 20:
                    continue
                alpha = float(np.dot(a[ok], b[ok]) / np.dot(a[ok], a[ok]))
                two = (K[n, t] * v[t] + K[t, n] * v[n]) / v[t]
                out[w].append((alpha / K[n, t], alpha / two))
    return sub, out


if __name__ == "__main__":
    subs = P.load()["partitions"]["DISCOVERY"]
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        res = list(ex.map(one, subs))
    rng = np.random.default_rng(0)
    summ = {}
    for w in ("spontaneous", "pre", "evoked", "post_quiet"):
        per = {s: np.array(o[w]) for s, o in res if len(o[w])}
        animals = list(per)
        def med(sel, k):
            x = np.concatenate([per[a][:, k] for a in sel]); return float(np.median(x))
        bo = {k: [med(rng.choice(animals, len(animals)), k) for _ in range(400)] for k in (0, 1)}
        summ[w] = dict(n_pairs=int(sum(len(v) for v in per.values())), n_animals=len(animals),
                       alpha_over_kappa_oneway=med(animals, 0), ci_oneway=[float(np.percentile(bo[0], 2.5)), float(np.percentile(bo[0], 97.5))],
                       alpha_over_twoway=med(animals, 1), ci_twoway=[float(np.percentile(bo[1], 2.5)), float(np.percentile(bo[1], 97.5))])
    p = REPO / "results/mechanism/cross_rig/alpha_twoway_discovery.json"; p.write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
