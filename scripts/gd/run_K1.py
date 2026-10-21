"""K1 (Kato 2015): held-out perturbation, AVA silencing (HisCl1 + histamine), predicted from wild-type dynamics.

Frozen spec (results/gd/benchmarks/K1_spec.json, write-once) before scoring:
  nodes: canonical neurons identified in >= 50% of WT recordings (NoStim + Stim, 12) and >= 3 of 5 HisCl recordings.
  models: 1-step source-set ridge (lags 5, alpha 100) per family: ar, var_dense, connectome, connectome_shuffled_s,
          random_k_s (s = 0..4), geometry_knn (Atanas median within-animal distances). Trained on all 12 WT recordings.
  intervention: free-running simulation (residual-sd Gaussian noise, 20000 steps, burn-in 1000, 3 seeds) with AVAL/AVAR
          clamped to 0 (their z-scored mean: fluctuations removed) vs unclamped.
  prediction: per non-AVA node, log(var_clamped / var_unclamped).
  observed: per non-AVA node, log(mean over HisCl animals of var(dF/F) / mean over WT_NoStim animals of var(dF/F))
          (both basal, unstimulated protocols).
  score: Spearman across nodes; connectome vs the mean of its matched nulls; manipulation check = AVA variance ratio.
Small n (5 vs 5 animals; ~20 nodes): a feasibility benchmark for the intervention API, not a confirmatory test.
"""
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_A2 as A2  # noqa: E402  (source-set construction and distances)
from wormlab.gd import graph as G, preprocess as P  # noqa: E402
from wormlab.gd.models.linear import SourceSetRidge, lagged  # noqa: E402
from wormlab.gd.sources import atanas2023 as A, kato2015 as K  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/gd/benchmarks"
LAGS, ALPHA, STEPS, BURN, SEEDS = 5, 100.0, 20000, 1000, (0, 1, 2)
CLAMP = ("AVAL", "AVAR")


def simulate(model, nodes, resid_sd, clamp_idx, seed):
    """x_t = W vec(x_{t-1..t-p}) + c + noise, with W assembled from the per-node source-set coefficients."""
    rng = np.random.default_rng(seed)
    K_, p = len(nodes), model.p
    W = np.zeros((K_, K_ * p)); c0 = np.zeros(K_)
    for i, c in enumerate(model.coef):
        cols = model._cols(i)[:-1]
        W[i, cols] = c[:-1]; c0[i] = c[-1]
    hist = np.zeros((K_, p))                                    # column l = lag l (0 = most recent)
    out = np.empty((STEPS, K_))
    noise = rng.normal(0, 1, (STEPS, K_)) * resid_sd
    for t in range(STEPS):
        x = W @ hist.ravel() + c0 + noise[t]
        x[clamp_idx] = 0.0
        if not np.all(np.isfinite(x)) or np.abs(x).max() > 1e3:
            return None
        hist = np.roll(hist, 1, axis=1); hist[:, 0] = x
        out[t] = x
    return out[BURN:]


def main():
    wt = [r for a in ("WT_NoStim", "WT_Stim") for r in K.load(a, download=False)]
    his = K.load("AVA_HisCl", download=False)
    nodes = [n for n in P.node_space(wt, 0.5) if sum(n in r.canonical_index() for r in his) >= 3]
    clamp_idx = [nodes.index(c) for c in CLAMP if c in nodes]
    at = [P.standardize(r) for r in A.load()]
    D = A2.median_pairwise_distance(at, nodes)
    C = G.load_connectome()
    A2.N_NULL = 5
    sets, k = A2.source_sets(nodes, C, D, np.random.default_rng(20261008))
    spec = dict(task="K1_kato_AVA_HisCl", nodes=nodes, clamp=CLAMP, lags=LAGS, alpha=ALPHA, steps=STEPS, burn=BURN, seeds=SEEDS,
                models=sorted(sets), train="WT_NoStim + WT_Stim (12)", observed_reference="WT_NoStim (5) vs AVA_HisCl (5)")
    p = OUT / "K1_spec.json"
    if not p.exists():
        p.write_text(json.dumps(dict(spec, frozen_utc=datetime.now(timezone.utc).isoformat()), indent=1)); os.chmod(p, 0o444)
    train = [P.to_nodes(P.standardize(r), nodes) for r in wt]

    # observed change (raw dF/F variance; basal protocols only)
    def var_by_node(recs):
        out = []
        for n in nodes:
            v = [np.var(r.neural[r.mask[:, r.canonical_index()[n]], r.canonical_index()[n]]) for r in recs if n in r.canonical_index()]
            out.append(np.mean(v) if v else np.nan)
        return np.array(out)
    obs = np.log(var_by_node(his) / var_by_node([r for r in wt if r.condition == "WT_NoStim"]))
    target = [i for i in range(len(nodes)) if i not in clamp_idx and np.isfinite(obs[i])]

    res = {}
    for name, S in sets.items():
        m = SourceSetRidge(S, LAGS, ALPHA, name).fit(train, 1)
        # residual sd on training data
        rs = []
        for X, M in train:
            Pd = m.predict(X, M); e = X[1:] - Pd[:-1]; e[~M[1:]] = np.nan; rs.append(e)
        resid_sd = np.nanstd(np.concatenate(rs), 0)
        preds = []
        for sd in SEEDS:
            a = simulate(m, nodes, resid_sd, [], sd); b = simulate(m, nodes, resid_sd, clamp_idx, sd)
            if a is None or b is None:
                preds = None; break
            preds.append(np.log(b.var(0) / a.var(0)))
        if preds is None:
            res[name] = dict(stable=False); continue
        pred = np.mean(preds, 0)
        rho = spearmanr(pred[target], obs[target]).correlation
        res[name] = dict(stable=True, spearman=float(rho), pred=pred.tolist())
    nulls = lambda pre: [res[k]["spearman"] for k in res if k.startswith(pre) and res[k].get("stable")]
    summary = dict(manipulation_check_AVA_log_var_ratio=[float(obs[i]) for i in clamp_idx], n_target_nodes=len(target),
                   spearman={k: v.get("spearman") for k, v in res.items()},
                   connectome_minus_mean_shuffled=res["connectome"]["spearman"] - float(np.mean(nulls("connectome_shuffled"))) if res["connectome"].get("stable") else None,
                   connectome_minus_mean_random=res["connectome"]["spearman"] - float(np.mean(nulls("random_k"))) if res["connectome"].get("stable") else None,
                   observed=obs.tolist())
    (OUT / "K1_results.json").write_text(json.dumps(dict(summary=summary, per_model=res), indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "observed"}, indent=1, default=float))


if __name__ == "__main__":
    main()
