"""PQ01: per-dataset signal + NaN mask exactly as used by the published funatlas.h5.

EMPIRICAL FINDING (reproduced in pq03_validate_dff.py): the published per-trial dF/F values
(funatlas.h5 wt/dFF_all) are reproduced to floating-point precision when the significance features are
computed on the Leifer-exported gcamp traces AS-IS (no additional photobleaching correction, spike removal,
or causal SG smoothing); applying the wormdatamodel preprocessing chain (signal.py:178-185) on top of the
export moves values by 1e-3..1e-2. The export is Funatlas.sig[ds].data (Funatlas.export_to_txt,
Funatlas.py:279-309), i.e. the same array the published statistics read.

NaN mask (Signal.nan_mask = isnan(raw) before interpolation, signal.py:104): taken from DANDI:001075
BaseGreenSignal for every column whose finite values equal the export exactly. Columns that do not match
(export substituted, e.g. 'matchless' traces, signal.py:293-324) and the 3 exported datasets with no DANDI
subject get an all-False mask (SUBSTITUTION; for matchless-substituted columns this is also what the code
would produce when the matchless trace has no NaNs).
Output: out/prep/{set}_{k}.npz (Y, M, mask_src) and out/prep_index_{set}.json.
"""
import glob, json, os, re, sys
import numpy as np

BASE = (os.environ.get('RANDI_EXPORT_DIR', 'data/randi_raw') + '/')
SETS = {'wt': BASE + 'exported_data', 'spont': BASE + 'spont_data', 'unc31': BASE + 'exported_data_unc31'}
RAW_DIR = os.environ.get('RANDI_RAW_DIR', 'data/randi2023_dandi001075')
CACHE = os.path.join(os.path.dirname(__file__), '..', '..', 'out', 'cache')
OUT = os.path.join(os.path.dirname(__file__), '..', 'out', 'prep')
os.makedirs(OUT, exist_ok=True)


def dandi_sub_for_session():
    ses = {}
    for f in os.listdir(RAW_DIR):
        m = re.match(r'(sub-\d+)_ses-(\d{8})-(\d{6})_desc-imaging', f)
        if m:
            ses[f'{m.group(2)}_{m.group(3)}'] = m.group(1)
    return ses


def work(setname, k, sub):
    E = np.loadtxt(f'{SETS[setname]}/{k}_gcamp.txt')
    if E.ndim == 1:
        E = E[:, None]
    T, N = E.shape
    M = np.zeros_like(E, dtype=bool)
    src = np.array(['none'] * N, dtype='<U8')
    if sub is not None and os.path.exists(f'{CACHE}/{sub}.npz'):
        G = np.load(f'{CACHE}/{sub}.npz')['G']
        if G.shape[0] == T:
            for j in range(min(N, G.shape[1])):
                fin = np.isfinite(G[:, j])
                if np.array_equal(G[fin, j], E[fin, j]):
                    M[:, j] = ~fin; src[j] = 'dandi'
    np.savez_compressed(f'{OUT}/{setname}_{k}.npz', Y=E, M=M, mask_src=src)
    return dict(set=setname, k=k, sub=sub, T=T, N=N, n_cols_mask_from_dandi=int((src == 'dandi').sum()),
                nan_frac=float(M.mean()))


def main():
    setname = sys.argv[1] if len(sys.argv) > 1 else 'wt'
    d = SETS[setname]
    keys = sorted([os.path.basename(p).split('_')[0] for p in glob.glob(f'{d}/*_gcamp.txt')], key=int)
    ses = dandi_sub_for_session()
    res = []
    for k in keys:
        m = re.search(r'pumpprobe_(\d{8}_\d{6})', open(f'{d}/{k}_ds_name.txt').read())
        sub = ses.get(m.group(1)) if (m and setname == 'wt') else None
        r = work(setname, k, sub); res.append(r); print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(OUT, '..', f'prep_index_{setname}.json'), 'w'), indent=1)
    if setname == 'wt':
        tot = sum(r['N'] for r in res); dn = sum(r['n_cols_mask_from_dandi'] for r in res)
        print(f'columns with DANDI NaN mask: {dn}/{tot}; datasets without DANDI: {[r["k"] for r in res if r["sub"] is None]}')


if __name__ == '__main__':
    main()
