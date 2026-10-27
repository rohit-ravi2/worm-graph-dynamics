"""Recon: align DANDI:001075 seg NWB (sub-0) with Leifer exported_data recording 0."""
import os
import h5py, numpy as np
SEG = os.path.join(os.environ.get('RANDI_SEG_DIR', 'data/randi2023_dandi001075_seg'), 'sub-0_seg.nwb')
EXP = os.path.join(os.environ.get('RANDI_EXPORT_DIR', 'data/randi_raw'), 'exported_data', '0_')

g = np.loadtxt(EXP + 'gcamp.txt')
lab = open(EXP + 'labels.txt').read().split('\n')
sn = np.loadtxt(EXP + 'stim_neurons.txt').astype(int)
sv = np.loadtxt(EXP + 'stim_volume_i.txt').astype(int)
print('exported', g.shape, 'labels', len(lab), 'stims', len(sn), len(sv))

with h5py.File(SEG, 'r') as h:
    G = h['processing/ophys/GreenSignals/BaseGreenSignal/data'][:]
    R = h['processing/ophys/RedSignals/BaseRedSignal/data'][:]
    ts = h['processing/ophys/GreenSignals/BaseGreenSignal/timestamps'][:]
    S = h['intervals/OptogeneticStimulusTable']
    st = S['start_time'][:]; tid = S['target_pumpprobe_id'][:]
    refs = S['targets'][:]
    T = h['processing/ophys/TargetedImageSegmentation/TargetPlaneSegmentation']
    pm = T['pixel_mask'][:]; pmi = T['pixel_mask_index'][:]; dep = T['depth_in_um'][:]
    print('first target ref ->', h[refs[0]].name, h[refs[0]].dtype if hasattr(h[refs[0]], 'dtype') else '')
    npid = h['processing/ophys/PumpProbeGreenSegmentations/PumpProbeGreenPlaneSegmentation/neuropal_ids'][:]
    C = h['processing/ophys/PumpProbeGreenSegmentations/PumpProbeGreenPlaneSegmentation/centroids'][:]
    NPl = h['processing/ophys/NeuroPALSegmentations/NeuroPALPlaneSegmentation/labels'][:]
print('seg G', G.shape, 'nan frac', np.isnan(G).mean().round(3), 'stims', len(st), 'pixel_mask rows', len(pm), 'index', pmi[:5])
stim_vol_seg = np.searchsorted(ts, st)
print('stim volume seg', stim_vol_seg[:8], 'exported', sv[:8])
print('tid', tid[:8], 'exported sn', sn[:8])
# does exported gcamp match green, or green/red ?
for name, M in [('green', G), ('red', R), ('g/r', G / R)]:
    ok = np.isfinite(M) & np.isfinite(g)
    cols = [np.corrcoef(M[:, j][ok[:, j]], g[:, j][ok[:, j]])[0, 1] for j in range(0, 114, 10)]
    print(name, np.round(cols, 3))
# labels: exported label for stim neuron vs neuropal label via neuropal_ids
for k in range(8):
    j = sn[k]
    if j >= 0:
        nid = npid[j]
        print(k, 'exp label', repr(lab[j]), 'npid', nid, 'centroid', C[j], 'pixel', pm[k], 'depth', dep[k])
