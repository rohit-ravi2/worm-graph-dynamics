"""T5 map: near-field (atlas-distance) response profile, WT vs unc-31, from the Leifer exports
(same windows and robust-z as build_responses; positions unavailable for unc-31 -> atlas distance).

Conditioning: target up (target z_g_full > 3), stratified by target dFF_full tercile (bleed scales
with the target's signal; unc-31 used 300 ms vs 500 ms stimulation). Readouts per genotype:
P(responder up | target up, atlas-distance bin, target-amplitude tercile); near-field excess
over each genotype's own far floor. Animal-bootstrap CIs.
"""
import glob, json, os
import numpy as np
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
RAW = os.environ.get('RANDI_EXPORT_DIR', 'data/randi_raw')
ATLAS = os.path.join(os.path.join(os.path.dirname(__import__('wormneuroatlas').__file__), 'data'), 'anatlas_neuron_positions.txt')
names = open(ATLAS).readline().lstrip('#').split()
P = np.loadtxt(ATLAS, comments='#'); apos = {n: P[i] for i, n in enumerate(names)}
PRE, FULL = 10, 20


def robust_z(A):
    med = np.nanmedian(A, 0); mad = 1.4826 * np.nanmedian(np.abs(A - med), 0)
    with np.errstate(all='ignore'):
        return (A - med) / mad


def rows_for(gdir):
    recs = []
    keys = sorted({os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(RAW, gdir, '*_gcamp.txt'))}, key=int)
    for ai, k in enumerate(keys):
        p = os.path.join(RAW, gdir, f'{k}_')
        G = np.loadtxt(p + 'gcamp.txt')
        if G.ndim == 1:
            continue
        lab = [l.strip() for l in open(p + 'labels.txt').read().split('\n')][:G.shape[1]]
        lab += [''] * (G.shape[1] - len(lab))
        sn = np.atleast_1d(np.loadtxt(p + 'stim_neurons.txt')).astype(int)
        sv = np.atleast_1d(np.loadtxt(p + 'stim_volume_i.txt')).astype(int)
        nS, nN = len(sv), G.shape[1]
        amp = np.full((nS, nN), np.nan)
        for s, v in enumerate(sv):
            with np.errstate(all='ignore'):
                g0 = np.nanmean(G[max(v - PRE, 0):v], 0); gf = np.nanmean(G[v:min(v + FULL, len(G))], 0)
                amp[s] = gf / g0 - 1
        z = robust_z(amp)
        cnt = {l: lab.count(l) for l in set(lab)}
        for s in range(nS):
            t = sn[s]
            if not (0 <= t < nN) or lab[t] not in apos or cnt[lab[t]] > 1 or not np.isfinite(z[s, t]):
                continue
            for j in range(nN):
                if j == t or lab[j] not in apos or cnt[lab[j]] > 1 or not np.isfinite(z[s, j]):
                    continue
                recs.append((ai, s, np.linalg.norm(apos[lab[t]] - apos[lab[j]]), z[s, j], z[s, t], amp[s, t]))
    return np.array(recs)


rng = np.random.default_rng(5)


def rate(R, m, B=500):
    an = R[:, 0].astype(int); A = np.unique(an[m]); y = R[:, 3] > 3
    num = np.bincount(an[m], weights=y[m].astype(float), minlength=an.max() + 1)
    den = np.bincount(an[m], minlength=an.max() + 1).astype(float)
    bs = [num[s].sum() / max(den[s].sum(), 1) for s in (rng.choice(A, len(A)) for _ in range(B))]
    return [round(float(num.sum() / max(den.sum(), 1)), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), int(den.sum())]


cache = os.path.join(OUT, 't5_rows.npz')
if os.path.exists(cache):
    c = np.load(cache); WT, UN = c['WT'], c['UN']
else:
    WT, UN = rows_for('exported_data'), rows_for('exported_data_unc31')
    np.savez_compressed(cache, WT=WT, UN=UN)
print('rows WT', len(WT), 'animals', len(np.unique(WT[:, 0])), '| unc-31', len(UN), 'animals', len(np.unique(UN[:, 0])))
qs = np.quantile(WT[:, 2], [0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0])
up_wt, up_un = WT[:, 4] > 3, UN[:, 4] > 3
terc = np.quantile(np.concatenate([WT[up_wt, 5], UN[up_un, 5]]), [0, 1 / 3, 2 / 3, 1])
print('target dFF of target-up events: WT median', np.median(WT[up_wt, 5]).round(3), 'unc-31 median', np.median(UN[up_un, 5]).round(3))
out = dict(atlas_bins=qs.tolist(), target_amp_terciles=terc.tolist(), rows=[])
for ti in range(3):
    for i in range(len(qs) - 1):
        r = dict(tercile=ti, atlas_bin=[round(qs[i], 3), round(qs[i + 1], 3)])
        for nm, R, up in [('WT', WT, up_wt), ('unc31', UN, up_un)]:
            m = up & (R[:, 5] >= terc[ti]) & (R[:, 5] <= terc[ti + 1]) & (R[:, 2] >= qs[i]) & (R[:, 2] < qs[i + 1] + (1e-9 if i == len(qs) - 2 else 0))
            r[nm] = rate(R, m)
        out['rows'].append(r); print(r)
for nm, R, up in [('WT', WT, up_wt), ('unc31', UN, up_un)]:
    near = rate(R, up & (R[:, 2] < qs[2])); far = rate(R, up & (R[:, 2] >= qs[4]))
    flat_near = rate(R, (np.abs(R[:, 4]) < 1) & (R[:, 2] < qs[2]))
    out[nm + '_near_vs_far'] = dict(near_lt_p5=near, far_ge_p20=far, near_target_flat=flat_near)
    print(nm, out[nm + '_near_vs_far'])
json.dump(out, open(os.path.join(OUT, 't5_genotype.json'), 'w'), indent=1)
