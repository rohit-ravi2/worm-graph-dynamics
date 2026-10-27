"""PQ07: re-score structural predictors on each regime's reproduced q map (and on power-matched control maps).

Declared before scoring (same as run-1 T6 plus two additions):
  dist      = -atlas distance (wormneuroatlas anatlas_neuron_positions.txt)
  conn      = W = log1p(chem[i,j] + gap[i,j])  (aconnectome_default.h5, chem[i,j] = synapses j -> i; i responder)
  hop2      = log1p(# directed 2-hop paths j -> k -> i on binary graph B = (chem>0)|(gap>0))
  lr_dist, lr_conn, lr_conn_dist = leave-one-source-out logistic regression on within-source z-scored features
Label: q < 0.05, and separately p <= p* (fixed per-pair evidence threshold from A_geo, see pq06) (off-diagonal, testable pairs with atlas positions). Metric: AUROC within each source with >=1
significant and >=1 non-significant pair, averaged over sources; 95% CI by source bootstrap (2000, seeded).
Usage: pq07_predictors.py <null>
"""
import json, os, sys
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')  # tiny BLAS problems: multithreading made LOSO fits ~100x slower
import numpy as np, h5py
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(__file__))
from pq02_features import FA, IDS

OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
WNA = os.path.dirname(FA) + '/'
names = open(WNA + 'anatlas_neuron_positions.txt').readline().lstrip('#').split()
Ppos = np.loadtxt(WNA + 'anatlas_neuron_positions.txt', comments='#')
apos = {n: Ppos[i] for i, n in enumerate(names)}
with h5py.File(WNA + 'aconnectome_default.h5') as h:
    chem, gap = h['chem'][:], h['gap'][:]
HAS = np.array([n in apos for n in IDS])
POS = np.array([apos[n] if n in apos else np.full(3, np.nan) for n in IDS])
DIST = np.linalg.norm(POS[:, None, :] - POS[None, :, :], axis=-1)
W = np.log1p(chem + gap)
B = ((chem > 0) | (gap > 0)).astype(float)
HOP2 = np.log1p(B @ B)
OFFD = ~np.eye(300, dtype=bool)
PRED = {'dist': -DIST, 'conn': W, 'hop2': HOP2}


def auroc(score, lab):
    pos = lab.sum(); neg = len(lab) - pos
    if pos == 0 or neg == 0:
        return np.nan
    r = rankdata(score)
    return (r[lab].sum() - pos * (pos + 1) / 2) / (pos * neg)


def zwithin(x, src):
    z = np.zeros_like(x, float)
    for s in np.unique(src):
        m = src == s; sd = x[m].std()
        z[m] = (x[m] - x[m].mean()) / sd if sd > 0 else 0.0
    return z


def score_map(sig, testable, rng, n_boot=2000, with_lr=True, ci=True):
    universe = testable & OFFD & HAS[:, None] & HAS[None, :]
    I, J = np.where(universe)
    y = sig[I, J]
    feats = {k: v[I, J] for k, v in PRED.items()}
    per = {}
    srcs = [s for s in np.unique(J) if 0 < y[J == s].sum() < (J == s).sum()]
    if with_lr:
        Z = {k: zwithin(feats[k], J) for k in ['dist', 'conn']}
        lr_sets = {'lr_dist': ['dist'], 'lr_conn': ['conn'], 'lr_conn_dist': ['conn', 'dist']}
        lr_scores = {k: np.full(len(y), np.nan) for k in lr_sets}
        for s in srcs:
            te = J == s; tr = ~te
            for k, cols in lr_sets.items():
                X = np.column_stack([Z[c] for c in cols])
                clf = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr])
                lr_scores[k][te] = clf.decision_function(X[te])
        feats.update(lr_scores)
    keys = list(feats)
    for s in srcs:
        m = J == s
        per[s] = {k: auroc(feats[k][m], y[m]) for k in keys}
    out = dict(n_pairs=int(len(y)), n_sig=int(y.sum()), n_sources_scored=len(srcs))
    if not srcs:
        return out
    vals = {k: np.array([per[s][k] for s in srcs]) for k in keys}
    for k in keys:
        v = vals[k]; d = dict(mean=round(float(np.nanmean(v)), 4))
        if ci:
            bs = [np.nanmean(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
            d['ci95'] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        out[k] = d
    diffs = {'dist_minus_conn': vals['dist'] - vals['conn']}
    if with_lr:
        diffs['lr_conn_dist_minus_lr_dist'] = vals['lr_conn_dist'] - vals['lr_dist']
    for k, v in diffs.items():
        d = dict(mean=round(float(np.nanmean(v)), 4))
        if ci:
            bs = [np.nanmean(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
            d['ci95'] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        out[k] = d
    return out


def main():
    null = sys.argv[1] if len(sys.argv) > 1 else 'wtqgt0.5'
    Mz = np.load(f'{OUT}/pq06_{null}_maps.npz')
    with h5py.File(FA, 'r') as f:
        qpub = f['wt/q'][:]
    rng = np.random.default_rng(7)
    res = dict(null=null, universe_note='off-diagonal testable pairs with both neurons in anatlas positions',
               declared_predictors=list(PRED) + ['lr_dist', 'lr_conn', 'lr_conn_dist'])
    res['published'] = score_map(qpub < 0.05, np.isfinite(qpub), rng)
    pstar = json.load(open(f'{OUT}/pq06_{null}.json'))['pstar_from_A_geo']
    for name in ['A', 'A_geo', 'B', 'C']:
        Q = Mz[f'Q_{name}']; P = Mz[f'P_{name}']
        res[name] = score_map(Q < 0.05, np.isfinite(Q), rng)
        res[f'{name}_p_leq_pstar'] = score_map(P <= pstar, np.isfinite(P), rng)
        print(name, json.dumps({k: v for k, v in res[name].items()}), flush=True)
    for name in ['B', 'C']:
        for ctrl, label in [('rand', 'q'), ('pairmatch', 'q'), ('rand', 'p'), ('pairmatch', 'p')]:
            S = Mz[f'{name}_{ctrl}_sig' if label == 'q' else f'{name}_{ctrl}_psig']; Tm = Mz[f'{name}_{ctrl}_testable']
            rows = []
            for d in range(len(S)):
                sig = np.unpackbits(S[d])[:90000].reshape(300, 300).astype(bool)
                tst = np.unpackbits(Tm[d])[:90000].reshape(300, 300).astype(bool)
                rows.append(score_map(sig, tst, rng, with_lr=False, ci=False))  # LOSO logistic only on regime maps (cost)
            agg = {}
            for k in ['dist', 'conn', 'hop2', 'lr_dist', 'lr_conn', 'lr_conn_dist', 'dist_minus_conn', 'lr_conn_dist_minus_lr_dist']:
                v = np.array([r[k]['mean'] for r in rows if k in r])
                if len(v):
                    agg[k] = dict(mean_over_draws=round(float(v.mean()), 4), p2_5=round(float(np.percentile(v, 2.5)), 4),
                                  p97_5=round(float(np.percentile(v, 97.5)), 4))
            agg['n_sources_scored_mean'] = round(float(np.mean([r['n_sources_scored'] for r in rows])), 1)
            agg['n_draws'] = len(rows)
            res[f'{name}_ctrl_{ctrl}' + ('' if label == 'q' else '_p_leq_pstar')] = agg
            print(name, ctrl, label, json.dumps(agg), flush=True)
    json.dump(res, open(f'{OUT}/pq07_{null}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
