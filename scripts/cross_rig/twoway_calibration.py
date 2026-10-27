"""Calibration of the two-way copy-ratio estimator (synthetic only; gate for its held-out use).

Estimator (frozen in scripts/cross_rig/alpha_twoway_real.py): per near pair (< 6 um, kappa_nt > 0.02, stimulated
targets), alpha / [(kappa_nt var_t + kappa_tn var_n) / var_t] on first-differenced g3 residuals; median over pairs;
animal bootstrap. Worlds: pure static-kernel contamination at two brightness spreads, the calibrated realistic world
(PSF tails + size heterogeneity, i.e. kernel misspecified), and a coupling world (C_H3).
Declared reading: contamination worlds should give ~1 (within [0.7, 1.4]); the coupling world should exceed them if the
estimator is sensitive to coupling at all.
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from wormlab.mechanism import provenance_synth as S, signal_provenance as SP, waveform_provenance as WP

REPO = Path(__file__).resolve().parents[2]
C = dict(n_neurons=280, min_sep_um=2.2, sigma_xy_um=1.05, sigma_z_um=1.9, f0_logsd=1.1, psf_tail_frac=0.15, nucleus_size_logsd=0.15)
WORLDS = {
    "pure_equal_brightness": {**C, "psf_tail_frac": 0.0, "nucleus_size_logsd": 0.0, "f0_logsd": 0.0},
    "pure_hetero_brightness": {**C, "psf_tail_frac": 0.0, "nucleus_size_logsd": 0.0},
    "realistic_misspecified": C,
    "coupling_C_H3": {**C, "p_gap_near": 0.9, "lam_gap_um": 4.0, "gap_coef": 0.35},
}


def one(args):
    world, seed = args
    pl, _ = S.simulate(S.WorldParams(**WORLDS[world], n_volumes=1646, seed=seed))
    ap = SP.prepare_animal(f"{world}-{seed}", pl)
    R = WP._residuals(ap)
    masks = WP._window_masks(ap.prov)
    K = ap.kap["kappa"]["box"]; dist = ap.kap["dist"]
    tg = sorted({int(e["target_nucleus"]) for e in SP.event_windows(ap.prov)})
    out = {}
    for w in ("spontaneous", "evoked"):
        D = WP._diff_pairs(R, masks[w]); v = np.nanvar(D, 0); r = []
        for t in tg:
            for n in np.where((dist[:, t] < 6) & (np.arange(len(dist)) != t))[0]:
                if K[n, t] <= 0.02 or not (v[t] > 0):
                    continue
                a, b = D[:, t], D[:, n]; ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 20:
                    continue
                alpha = float(np.dot(a[ok], b[ok]) / np.dot(a[ok], a[ok]))
                r.append((alpha / K[n, t], alpha / ((K[n, t] * v[t] + K[t, n] * v[n]) / v[t])))
        out[w] = r
    return world, seed, out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    jobs = [(w, 3000 + k) for w in WORLDS for k in range(n)]
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as ex:
        res = list(ex.map(one, jobs))
    rng = np.random.default_rng(0); summ = {}
    for world in WORLDS:
        for w in ("spontaneous", "evoked"):
            per = [np.array(o[w]) for wd, s, o in res if wd == world and len(o[w])]
            if not per:
                continue
            med = lambda sel, k: float(np.median(np.concatenate([per[i][:, k] for i in sel])))
            idx = np.arange(len(per))
            bo = [med(rng.choice(idx, len(idx)), 1) for _ in range(400)]
            summ[f"{world}|{w}"] = dict(n_animals=len(per), n_pairs=int(sum(len(p) for p in per)), oneway=med(idx, 0), twoway=med(idx, 1),
                                        twoway_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))])
    p = REPO / f"results/mechanism/cross_rig/twoway_calibration_n{n}.json"; p.write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
