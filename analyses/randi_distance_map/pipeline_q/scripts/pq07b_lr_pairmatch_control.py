"""pq07b: logistic (LOSO) pair-matched control for the near-excluded connectome increment (lr_conn_dist - lr_dist).

pq07 scored the regime maps B/C (p <= p*) with the LOSO logistic and found connectome + distance > distance by +0.030
(primary null), but ran its 50 pair-matched control maps WITHOUT the logistic (cost). This script reuses pq07's
score_map unchanged, with_lr=True, on the pair-matched p <= p* control maps, and reports where the observed increment
falls in the control distribution. Exploratory (GENERATIVE) like the rest of pipeline_q.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pq07", sorted(HERE.glob("pq07_*.py"))[0])
pq07 = importlib.util.module_from_spec(spec); spec.loader.exec_module(pq07)

null = sys.argv[1] if len(sys.argv) > 1 else "wtqgt0.5"
OUT = pq07.OUT
Mz = np.load(f"{OUT}/pq06_{null}_maps.npz")
obs = json.load(open(f"{OUT}/pq07_{null}.json"))
rng = np.random.default_rng(11)
res = dict(null=null)
for name in ("B", "C"):
    o = obs[f"{name}_p_leq_pstar"]["lr_conn_dist_minus_lr_dist"]["mean"]
    S = Mz[f"{name}_pairmatch_psig"]; Tm = Mz[f"{name}_pairmatch_testable"]
    vals = []
    for d in range(len(S)):
        sig = np.unpackbits(S[d])[:90000].reshape(300, 300).astype(bool)
        tst = np.unpackbits(Tm[d])[:90000].reshape(300, 300).astype(bool)
        r = pq07.score_map(sig, tst, rng, with_lr=True, ci=False)
        if "lr_conn_dist_minus_lr_dist" in r:
            vals.append(r["lr_conn_dist_minus_lr_dist"]["mean"])
    v = np.array(vals)
    res[name] = dict(observed=o, control_mean=float(v.mean()), control_p2_5=float(np.percentile(v, 2.5)), control_p97_5=float(np.percentile(v, 97.5)),
                     frac_control_ge_observed=float((v >= o).mean()), n_draws=len(v), observed_minus_control_mean=float(o - v.mean()))
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(f"{OUT}/pq07b_{null}.json", "w"), indent=1)
