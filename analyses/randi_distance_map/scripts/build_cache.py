"""Build per-animal caches from DANDI:001075 segmentation NWBs + Leifer exported_data.

For each WT animal: green/red base signals (T x N, NaN kept), timestamps, ROI centroids
(x_px, y_px, z_plane), NeuroPAL labels, stim table (volume index, target ROI index from the
Leifer export, stim-spot pixel xy). Output: out/cache/<sub>.npz + out/cache/index.json.

Exported recordings are matched to DANDI subjects by session datetime
(exported ds_name 'pumpprobe_YYYYMMDD_HHMMSS' <-> raw filename 'ses-YYYYMMDD-HHMMSS').
"""
import glob, json, os, re, sys
import h5py
import numpy as np

SEG_DIR = os.environ.get('RANDI_SEG_DIR', 'data/randi2023_dandi001075_seg')
RAW_DIR = os.environ.get('RANDI_RAW_DIR', 'data/randi2023_dandi001075')
EXP_DIR = os.path.join(os.environ.get('RANDI_EXPORT_DIR', 'data/randi_raw'), 'exported_data')
OUT = os.path.join(os.path.dirname(__file__), '..', 'out', 'cache')
os.makedirs(OUT, exist_ok=True)


def session_of_subject():
    ses = {}
    for f in os.listdir(RAW_DIR):
        m = re.match(r'(sub-\d+)_ses-(\d{8})-(\d{6})_desc-imaging', f)
        if m:
            ses[m.group(1)] = f'{m.group(2)}_{m.group(3)}'
    return ses


def exported_by_session():
    out = {}
    for f in glob.glob(os.path.join(EXP_DIR, '*_ds_name.txt')):
        k = os.path.basename(f).split('_')[0]
        m = re.search(r'pumpprobe_(\d{8}_\d{6})', open(f).read())
        if m:
            out[m.group(1)] = k
    return out


def load_exported(k):
    p = os.path.join(EXP_DIR, f'{k}_')
    labels = open(p + 'labels.txt').read().split('\n')
    sn = np.atleast_1d(np.loadtxt(p + 'stim_neurons.txt')).astype(int)
    sv = np.atleast_1d(np.loadtxt(p + 'stim_volume_i.txt')).astype(int)
    return labels, sn, sv


def build(sub, ses, exp_key):
    f = os.path.join(SEG_DIR, f'{sub}_seg.nwb')
    with h5py.File(f, 'r') as h:
        ph = h['processing/ophys']
        G = ph['GreenSignals/BaseGreenSignal/data'][:]
        R = ph['RedSignals/BaseRedSignal/data'][:]
        ts = ph['GreenSignals/BaseGreenSignal/timestamps'][:]
        C = ph['PumpProbeGreenSegmentations/PumpProbeGreenPlaneSegmentation/centroids'][:]
        npid = ph['PumpProbeGreenSegmentations/PumpProbeGreenPlaneSegmentation/neuropal_ids'][:]
        np_labels = ph['NeuroPALSegmentations/NeuroPALPlaneSegmentation/labels'][:]
        np_ids = ph['NeuroPALSegmentations/NeuroPALPlaneSegmentation/id'][:]
        S = h['intervals/OptogeneticStimulusTable']
        st = S['start_time'][:]
        T = ph['TargetedImageSegmentation/TargetPlaneSegmentation']
        pm = T['pixel_mask'][:]
        pmi = T['pixel_mask_index'][:]
        strain = h['general/subject/strain'][()].decode()
    # one pixel per stim target (pixel_mask_index is cumulative end index)
    starts = np.concatenate([[0], pmi[:-1]])
    spot_xy = np.array([[pm[s]['x'], pm[s]['y']] for s in starts], float)
    stim_vol = np.searchsorted(ts, st)
    # NeuroPAL labels per pumpprobe ROI via neuropal_ids
    id2lab = {int(i): l.decode() if isinstance(l, bytes) else str(l) for i, l in zip(np_ids, np_labels)}
    roi_label = []
    for x in npid:
        x = x.decode() if isinstance(x, bytes) else str(x)
        roi_label.append(id2lab.get(int(x), '') if x.strip().isdigit() else '')  # '' / 'merge' = unmatched
    rec = dict(sub=sub, ses=ses, strain=strain, n_roi=int(C.shape[0]), n_t=int(G.shape[0]),
               n_stim=int(len(st)), exported=exp_key)
    stim_target = np.full(len(st), -9, int)
    if exp_key is not None:
        labels, sn, sv = load_exported(exp_key)
        if len(sn) == len(st) and np.all(np.abs(sv - stim_vol) <= 2):
            stim_target = sn
            stim_vol = sv  # Leifer's own stim volume index is canonical
            exp_labels = [l.strip() for l in labels[:C.shape[0]]]
            both = [(a, b) for a, b in zip(exp_labels, roi_label) if a and b]
            rec['label_agree_both_labeled'] = float(np.mean([a == b for a, b in both])) if both else None
            rec['n_exp_labels'] = sum(bool(a) for a in exp_labels)
            if rec['n_exp_labels'] > 0:  # Leifer-curated labels (those behind funatlas) take precedence
                roi_label = exp_labels
        else:
            rec['exported_mismatch'] = [int(len(sn)), int(len(st))]
    np.savez_compressed(os.path.join(OUT, f'{sub}.npz'), G=G, R=R, ts=ts, C=C,
                        roi_label=np.array(roi_label), stim_vol=stim_vol, stim_target=stim_target,
                        spot_xy=spot_xy)
    return rec


def main(limit=None):
    ses = session_of_subject()
    exp = exported_by_session()
    subs = sorted([os.path.basename(p).replace('_seg.nwb', '') for p in glob.glob(os.path.join(SEG_DIR, 'sub-*_seg.nwb'))],
                  key=lambda s: int(s.split('-')[1]))
    if limit:
        subs = subs[:limit]
    index = []
    for sub in subs:
        s = ses.get(sub)
        try:
            rec = build(sub, s, exp.get(s))
        except Exception as e:  # record, don't hide
            rec = dict(sub=sub, error=repr(e))
        index.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(index, open(os.path.join(OUT, 'index.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
