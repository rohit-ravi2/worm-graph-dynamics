"""Diagnose the W1/W2 copy-ratio estimator under KNOWN pure contamination (synthetic only).

Hypothesis: alpha = slope(D_n on D_t) estimates kappa_nt only if the receiver's light does not leak back into the
target box. With two-way leakage, E[alpha] ~ (kappa_nt var_t + kappa_tn var_n) / var_t, so alpha/kappa_nt > 1 and
grows with receiver/target variance ratio (brightness heterogeneity). Compares alpha/kappa_nt with alpha/alpha_2way.
"""
import sys, json
import numpy as np
from wormlab.mechanism import provenance_synth as S, signal_provenance as SP, waveform_provenance as WP

C = dict(n_neurons=280, min_sep_um=2.2, sigma_xy_um=1.05, sigma_z_um=1.9, f0_logsd=1.1, psf_tail_frac=0.0, nucleus_size_logsd=0.0)


def run(seed, f0_logsd):
    pl, _ = S.simulate(S.WorldParams(**dict(C, f0_logsd=f0_logsd), n_volumes=1646, seed=seed))
    ap = SP.prepare_animal(f"s{seed}", pl)
    R = WP._residuals(ap)
    D = WP._diff_pairs(R, WP._window_masks(ap.prov)["spontaneous"])
    K = ap.kap["kappa"]["box"]; dist = ap.kap["dist"]
    v = np.nanvar(D, 0)
    out = []
    tg = sorted({int(e["target_nucleus"]) for e in SP.event_windows(ap.prov)})
    for t in tg:
        for n in np.where((dist[:, t] < 6) & (np.arange(len(dist)) != t))[0]:
            if K[n, t] <= 0.02:
                continue
            a, b = D[:, t], D[:, n]; ok = np.isfinite(a) & np.isfinite(b)
            alpha = float(np.dot(a[ok], b[ok]) / np.dot(a[ok], a[ok]))
            two = (K[n, t] * v[t] + K[t, n] * v[n]) / v[t]
            out.append((alpha / K[n, t], alpha / two, v[n] / v[t]))
    o = np.array(out)
    return dict(n=len(o), alpha_over_kappa=float(np.median(o[:, 0])), alpha_over_twoway=float(np.median(o[:, 1])), var_ratio=float(np.median(o[:, 2])))


if __name__ == "__main__":
    res = {f"f0_logsd={s}": [run(2000 + k, s) for k in range(2)] for s in (0.0, 1.1)}
    print(json.dumps(res, indent=1))
