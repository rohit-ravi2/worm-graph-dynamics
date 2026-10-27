"""T2 map: does neuron IDENTITY carry response information beyond MEASURED per-animal distance?

Rows: targeted stims whose target label and responder label are both in the canonical atlas
(wormneuroatlas anatlas, 303 L/R-resolved names) and unique within the animal; non-target ROIs.
Distances:
  d_box   = sqrt((dx/13)^2 + (dy/13)^2 + (dz/7)^2), measured in this animal, in extraction-box units
  d_atlas = Euclidean distance between canonical atlas positions (atlas units)
Identity prior (leave-animal-out): response rate of the same ordered (target label, responder label)
  pair in all OTHER animals (NaN if unseen elsewhere); plus its leave-animal-out mean d_box.
Readouts (response = z_g_full > 3), overall and restricted to NON-overlapping boxes:
  (1) within-event AUROC of -d_box vs -d_atlas (events with >=1 responder and >=1 non-responder)
  (2) within d_box bins: AUROC of -d_atlas and of identity prior
  (3) within d_atlas bins: AUROC of -d_box
Animal-bootstrap CIs.
"""
import json, os
import numpy as np
from collections import defaultdict
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
ATLAS = os.path.join(os.path.join(os.path.dirname(__import__('wormneuroatlas').__file__), 'data'), 'anatlas_neuron_positions.txt')
names = open(ATLAS).readline().lstrip('#').split()
P = np.loadtxt(ATLAS, comments='#')
apos = {n: P[i] for i, n in enumerate(names)}

T = np.load(os.path.join(OUT, 'responses_wt.npz'))
animal, stim = T['animal'], T['stim']
rl, tl = T['resp_label'], T['target_label']
# uniqueness of labels within animal
dup = set()
cnt = defaultdict(int)
for a, l in set(zip(animal[stim == 0], rl[stim == 0])):
    pass
for a in np.unique(animal):
    m = (animal == a) & (stim == 0)
    labs, c = np.unique(rl[m], return_counts=True)
    for l, k in zip(labs, c):
        if k > 1:
            dup.add((a, l))
ok = (T['target'] >= 0) & ~T['is_target'] & np.isin(rl, names) & np.isin(tl, names) & (rl != tl)
ok &= np.array([(a, r) not in dup and (a, t) not in dup for a, r, t in zip(animal, rl, tl)]) if ok.any() else ok
ok &= np.isfinite(T['z_g_full']) & np.isfinite(T['dz_target'])
idx = np.where(ok)[0]
a_, s_ = animal[idx], stim[idx]
y = (T['z_g_full'][idx] > 3).astype(int)
dx, dy, dz = T['dx_target'][idx], T['dy_target'][idx], T['dz_target'][idx]
d_box = np.sqrt((dx / 13) ** 2 + (dy / 13) ** 2 + (dz / 7) ** 2)
overlap = (np.abs(dx) < 13) & (np.abs(dy) < 13) & (np.abs(dz) < 7)
d_atl = np.array([np.linalg.norm(apos[t] - apos[r]) for t, r in zip(tl[idx], rl[idx])])
pair = np.array([t + '>' + r for t, r in zip(tl[idx], rl[idx])])
# leave-animal-out identity prior
sum_y = defaultdict(float); n_y = defaultdict(int); sum_d = defaultdict(float)
per_an = defaultdict(lambda: [0.0, 0, 0.0])
for p, a, yy, db in zip(pair, a_, y, d_box):
    sum_y[p] += yy; n_y[p] += 1; sum_d[p] += db
    v = per_an[(p, a)]; v[0] += yy; v[1] += 1; v[2] += db
prior = np.full(len(idx), np.nan); prior_d = np.full(len(idx), np.nan)
for i, (p, a) in enumerate(zip(pair, a_)):
    v = per_an[(p, a)]
    n = n_y[p] - v[1]
    if n > 0:
        prior[i] = (sum_y[p] - v[0]) / n
        prior_d[i] = (sum_d[p] - v[2]) / n
print('rows', len(idx), 'animals', len(np.unique(a_)), 'responders', y.sum(), 'with prior', np.isfinite(prior).sum())


def auroc(score, lab):
    m = np.isfinite(score)
    score, lab = score[m], lab[m]
    pos, neg = lab == 1, lab == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    r = np.argsort(np.argsort(score, kind='mergesort'), kind='mergesort').astype(float)
    # average ranks for ties
    from scipy.stats import rankdata
    r = rankdata(score)
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def within_event(score_list, mask):
    ev = a_.astype(np.int64) * 100000 + s_
    res = defaultdict(list); evan = []
    for e in np.unique(ev[mask]):
        m = mask & (ev == e)
        if y[m].min() == y[m].max():
            continue
        for k, sc in score_list.items():
            res[k].append(auroc(sc[m], y[m]))
        evan.append(a_[m][0])
    return {k: np.array(v) for k, v in res.items()}, np.array(evan)


rng = np.random.default_rng(3)


def boot_mean(vals, groups, B=1000):
    A = np.unique(groups)
    est = np.nanmean(vals)
    bs = []
    for _ in range(B):
        s = rng.choice(A, len(A))
        bs.append(np.nanmean(np.concatenate([vals[groups == g] for g in s])))
    return [round(float(est), 4), round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]


out = {}
for nm, mask in [('all', np.ones(len(idx), bool)), ('no_overlap', ~overlap)]:
    scores = {'neg_d_box': -d_box, 'neg_d_atlas': -d_atl, 'identity_prior': prior}
    res, evan = within_event(scores, mask & np.isfinite(prior))
    r = {k: boot_mean(v, evan) for k, v in res.items()}
    r['diff_box_minus_atlas'] = boot_mean(res['neg_d_box'] - res['neg_d_atlas'], evan)
    r['diff_prior_minus_box'] = boot_mean(res['identity_prior'] - res['neg_d_box'], evan)
    r['n_events'] = int(len(evan))
    # binned conditional AUROCs (pooled within bin, bootstrap by animal)
    cond = []
    for lo, hi in [(0, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 4), (4, 100)]:
        m = mask & (d_box >= lo) & (d_box < hi) & np.isfinite(prior)
        if y[m].sum() < 10:
            continue
        def stat(sel):
            return [auroc(-d_atl[sel], y[sel]), auroc(prior[sel], y[sel]), auroc(-prior_d[sel], y[sel])]
        est = stat(m)
        A = np.unique(a_[m]); bs = []
        for _ in range(300):
            s = rng.choice(A, len(A)); sel = np.concatenate([np.where(m & (a_ == g))[0] for g in s])
            bs.append(stat(sel))
        bs = np.array(bs)
        cond.append(dict(d_box=[lo, hi], n=int(m.sum()), resp_rate=round(float(y[m].mean()), 4),
                         auroc_neg_d_atlas=[round(est[0], 4)] + np.nanpercentile(bs[:, 0], [2.5, 97.5]).round(4).tolist(),
                         auroc_identity_prior=[round(est[1], 4)] + np.nanpercentile(bs[:, 1], [2.5, 97.5]).round(4).tolist(),
                         auroc_neg_prior_mean_dbox=[round(est[2], 4)] + np.nanpercentile(bs[:, 2], [2.5, 97.5]).round(4).tolist()))
    r['conditional_on_d_box_bins'] = cond
    out[nm] = r
    print(nm, json.dumps(r, indent=1))
json.dump(out, open(os.path.join(OUT, 't2_identity_vs_distance.json'), 'w'), indent=1)
