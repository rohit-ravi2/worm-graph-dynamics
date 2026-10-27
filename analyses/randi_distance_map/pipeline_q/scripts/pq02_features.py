"""PQ02: per-event Randi significance features for every (dataset, labeled stim, labeled responder ROI).

For each exported WT dataset k: atlas indices from exported labels via Funatlas.ids_to_i semantics
(no merging; AWCOFF->AWCL, AWCON->AWCR; '_' stripped; exact match else -1; Funatlas.py:325-482).
Stim target atlas index aj = ai[stim_neurons[ie]] for stim_neurons >= 0 (Funatlas.py:253-255). -3 targets
(manually located, complement labels not exported) are kept with aj=-3 and resolved in pq03 by exact value
matching against funatlas.h5 dFF_all.
Windows per Fconn.py:242-247; features per Funatlas.get_significance_features (randi_q_lib).
Output: out/events_wt.npz with one row per event.
Null (spont): pseudo-stimulus at every volume ee in [SHIFT_VOL, T-120] for every ROI of the 2 spontaneous
recordings -> out/null_spont.npz.
"""
import glob, json, os, sys
from multiprocessing import Pool
import numpy as np
import h5py
sys.path.insert(0, os.path.dirname(__file__))
import randi_q_lib as L

BASE = (os.environ.get('RANDI_EXPORT_DIR', 'data/randi_raw') + '/')
PREP = os.path.join(os.path.dirname(__file__), '..', 'out', 'prep')
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
FA = os.path.join(os.path.join(os.path.dirname(__import__('wormneuroatlas').__file__), 'data'), 'funatlas.h5')


def atlas_ids():
    with h5py.File(FA, 'r') as f:
        ids = [x.decode() for x in f['neuron_ids'][:]]
    # funatlas.h5 truncates to 5 chars: AWCOF/AWCON are atlas AWCL/AWCR (Funatlas.py:69-70, 369-377)
    ids[ids.index('AWCOF')] = 'AWCL'; ids[ids.index('AWCON')] = 'AWCR'
    return ids


IDS = atlas_ids()
IX = {n: i for i, n in enumerate(IDS)}


def ai_of(label):
    lab = {'AWCOFF': 'AWCL', 'AWCON': 'AWCR'}.get(label, label).replace('_', '')
    return IX.get(lab, -1)


def features_for_events(Y, M, events):
    """events: list of (ee, roi). Returns dict of arrays."""
    T, N = Y.shape
    convs = {}
    rows = {k: [] for k in ['dff', 'sd', 'dFF', 'pre_zero', 'corrected', 'nanrun', 'seglen']}
    for ee, roi in events:
        if roi not in convs:
            convs[roi] = L.neuron_convs(Y[:, roi])
        i0, sv, i1 = L.stim_windows(ee, T)
        f = L.significance_features(Y[:, roi], convs[roi], i0, i1)
        for k in ['dff', 'sd', 'dFF', 'pre_zero', 'corrected']:
            rows[k].append(f[k])
        rows['nanrun'].append(L.max_nan_run(M[i0:i1, roi]))
        rows['seglen'].append(i1 - i0)
    return rows


def work_wt(k):
    d = BASE + 'exported_data/'
    P = np.load(f'{PREP}/wt_{k}.npz')
    Y, M = P['Y'], P['M']
    T, N = Y.shape
    labels = open(f'{d}{k}_labels.txt').read().split('\n')
    ai = np.array([ai_of(labels[i]) if i < len(labels) else -1 for i in range(N)])
    sn = np.atleast_1d(np.loadtxt(f'{d}{k}_stim_neurons.txt')).astype(int)
    sv = np.atleast_1d(np.loadtxt(f'{d}{k}_stim_volume_i.txt')).astype(int)
    ev, meta = [], []
    for ie, (s, ee) in enumerate(zip(sn, sv)):
        if s == -3:  # manually located target, label not exported: aj resolved later by value matching (pq03)
            aj = -3
        elif s < 0 or ai[s] < 0:
            continue
        else:
            aj = ai[s]
        for roi in np.where(ai >= 0)[0]:
            ev.append((ee, roi)); meta.append((int(k), ie, roi, ai[roi], aj, s))
    rows = features_for_events(Y, M, ev)
    meta = np.array(meta, dtype=int).reshape(-1, 6)
    out = dict(ds=meta[:, 0], stim=meta[:, 1], roi=meta[:, 2], ai=meta[:, 3], aj=meta[:, 4], target_roi=meta[:, 5])
    out.update({kk: np.array(v) for kk, v in rows.items()})
    return out


def work_spont(k):
    P = np.load(f'{PREP}/spont_{k}.npz')
    Y, M = P['Y'], P['M']
    T, N = Y.shape
    ev = [(ee, roi) for roi in range(N) for ee in range(L.SHIFT_VOL, T - 120 + 1)]
    rows = features_for_events(Y, M, ev)
    out = {kk: np.array(v) for kk, v in rows.items()}
    out['ds'] = np.full(len(ev), int(k)); out['roi'] = np.array([e[1] for e in ev]); out['ee'] = np.array([e[0] for e in ev])
    return out


def cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'wt'
    procs = int(os.environ.get('PQ_PROCS', 12))
    if which == 'wt':
        keys = sorted([os.path.basename(p)[3:-4] for p in glob.glob(f'{PREP}/wt_*.npz')], key=int)
        if len(sys.argv) > 2:
            keys = keys[:int(sys.argv[2])]
        with Pool(procs) as pool:
            parts = pool.map(work_wt, keys)
        E = cat(parts)
        np.savez_compressed(f'{OUT}/events_wt.npz', **E)
        print('events', len(E['ds']), 'datasets', len(keys))
    else:
        keys = sorted([os.path.basename(p)[6:-4] for p in glob.glob(f'{PREP}/spont_*.npz')], key=int)
        with Pool(procs) as pool:
            parts = pool.map(work_spont, keys)
        E = cat(parts)
        np.savez_compressed(f'{OUT}/null_spont.npz', **E)
        print('null events', len(E['ds']))
