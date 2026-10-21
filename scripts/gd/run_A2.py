"""A2 (Atanas 2023): neural next-state forecasting on held-out animals; does connectome structure help?

Frozen before scoring (results/gd/benchmarks/A2_spec.json, write-once): node space (canonical neurons identified in
>= 50% of animals), 5-fold animal split (seed 20261008), preprocessing version, horizons h = 1, 4, 10 frames
(0.6, 2.4, 6.0 s), p = 5 lags, ridge alpha grid chosen on an inner animal split, model list and nulls.

Models (source set S_i for node i; node-specific ridge shared across animals):
  persistence; ar (own history); var_dense (all nodes); connectome (own + chem/gap neighbours);
  connectome_shuffled_s (degree-preserving shuffles, s = 0..4); geometry_knn (own + k_i nearest by median within-animal
  distance, k_i = connectome neighbour count); random_k_s (own + k_i random nodes, s = 0..4).
Question: does `connectome` beat its matched-sparsity nulls (shuffled, geometry, random) on held-out animals?
"""
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from wormlab.gd import graph as G, preprocess as P
from wormlab.gd.bench import forecast as F, splits as SPL
from wormlab.gd.models.linear import Persistence, SourceSetRidge
from wormlab.gd.sources import atanas2023 as A

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/gd/benchmarks"
SPEC = OUT / "A2_spec.json"
H = (1, 4, 10); LAGS = 5; ALPHAS = (1.0, 10.0, 100.0, 1000.0); SEED = 20261008; N_NULL = 5


def median_pairwise_distance(recs, nodes):
    K = len(nodes); acc = [[[] for _ in range(K)] for _ in range(K)]
    for r in recs:
        ci = r.canonical_index()
        for a, na in enumerate(nodes):
            if na not in ci or not np.all(np.isfinite(r.positions_um[ci[na]])):
                continue
            for b, nb in enumerate(nodes):
                if nb in ci and np.all(np.isfinite(r.positions_um[ci[nb]])):
                    acc[a][b].append(np.linalg.norm(r.positions_um[ci[na]] - r.positions_um[ci[nb]]))
    D = np.array([[np.median(x) if x else np.nan for x in row] for row in acc])
    return D


def source_sets(nodes, C, D, rng):
    K = len(nodes)
    nb = C.sub(nodes).neighbors("any")
    k = np.array([len(n) for n in nb])
    sets = {"ar": [np.array([i]) for i in range(K)], "var_dense": [np.arange(K) for _ in range(K)],
            "connectome": [np.unique(np.r_[i, nb[i]]) for i in range(K)]}
    for s in range(N_NULL):
        nbs = G.shuffle_degree_preserving(C, seed=s).sub(nodes).neighbors("any")
        sets[f"connectome_shuffled_{s}"] = [np.unique(np.r_[i, nbs[i]]) for i in range(K)]
        sets[f"random_k_{s}"] = [np.unique(np.r_[i, rng.choice(np.setdiff1d(np.arange(K), [i]), k[i], replace=False)]) for i in range(K)]
    geo = []
    for i in range(K):
        d = D[i].copy(); d[i] = np.inf; d[~np.isfinite(d)] = np.inf
        geo.append(np.unique(np.r_[i, np.argsort(d)[: k[i]]]))
    sets["geometry_knn"] = geo
    return sets, k


def choose_alpha(sources, train, h, rng):
    idx = rng.permutation(len(train)); nv = max(1, len(train) // 4)
    va, tr = [train[i] for i in idx[:nv]], [train[i] for i in idx[nv:]]
    m = SourceSetRidge(sources, LAGS, ALPHAS[0]).fit(tr, h)
    pers = Persistence().fit(tr, h)
    best = None
    for a in ALPHAS:
        m.solve(a)
        s = [F.score_animal(X, M, m.predict(X, M), pers.predict(X, M), h) for X, M in va]
        r2 = np.mean([x["r2"] for x in s if x])
        if best is None or r2 > best[1]:
            best = (a, r2)
    return best[0]


def main():
    t0 = time.time()
    recs = [P.standardize(r) for r in A.load()]
    nodes = P.node_space(recs, 0.5)
    split = SPL.freeze("A2_atanas_5fold", [r.animal_id for r in recs], n_folds=5, seed=SEED)
    C = G.load_connectome()
    D = median_pairwise_distance(recs, nodes)
    rng = np.random.default_rng(SEED)
    sets, k = source_sets(nodes, C, D, rng)
    spec = dict(task="A2_atanas_neural_forecast", dataset=f"DANDI:{A.DANDISET}/{A.VERSION}", preprocess=P.PREPROCESS_VERSION,
                nodes=nodes, split_sha256=split["sha256"], horizons_frames=H, dt_s=0.6, lags=LAGS, alpha_grid=ALPHAS,
                models=sorted(sets) + ["persistence"], connectome=C.source, median_k=float(np.median(k)),
                metric="pooled R^2 per test animal; skill vs persistence; animal bootstrap; paired comparisons")
    OUT.mkdir(parents=True, exist_ok=True)
    if SPEC.exists():
        old = json.loads(SPEC.read_text())
        assert {k_: old[k_] for k_ in spec} == json.loads(json.dumps(spec)), "A2 spec differs from the frozen one"
    else:
        SPEC.write_text(json.dumps(dict(spec, frozen_utc=datetime.now(timezone.utc).isoformat()), indent=1)); os.chmod(SPEC, 0o444)
    data = {r.animal_id: P.to_nodes(r, nodes) for r in recs}
    res = {h: {m: {} for m in list(sets) + ["persistence"]} for h in H}
    alphas = {h: {} for h in H}
    for fold in range(split["n_folds"]):
        test = [a for a, f in split["fold_of"].items() if f == fold]
        train = [data[a] for a, f in split["fold_of"].items() if f != fold]
        for h in H:
            pers = Persistence().fit(train, h)
            for name, S in sets.items():
                a = choose_alpha(S, train, h, np.random.default_rng(SEED + fold))
                alphas[h].setdefault(name, []).append(a)
                m = SourceSetRidge(S, LAGS, a, name).fit(train, h)
                for an in test:
                    X, M = data[an]
                    res[h][name][an] = F.score_animal(X, M, m.predict(X, M), pers.predict(X, M), h)
            for an in test:
                X, M = data[an]
                Pp = pers.predict(X, M)
                res[h]["persistence"][an] = F.score_animal(X, M, Pp, Pp, h)
        print(f"fold {fold} done {time.time()-t0:.0f}s", flush=True)
    summary = {}
    for h in H:
        summ = {m: F.boot_mean([v["r2"] for v in d.values() if v]) for m, d in res[h].items()}
        animals = sorted(res[h]["connectome"])
        def paired(a, b):
            return F.boot_mean([res[h][a][x]["r2"] - res[h][b][x]["r2"] for x in animals if res[h][a][x] and res[h][b][x]])
        nulls_sh = [f"connectome_shuffled_{s}" for s in range(N_NULL)]; nulls_rk = [f"random_k_{s}" for s in range(N_NULL)]
        def paired_vs_mean(a, group):
            return F.boot_mean([res[h][a][x]["r2"] - np.mean([res[h][g][x]["r2"] for g in group]) for x in animals])
        summary[h] = dict(r2=summ, contrasts=dict(
            connectome_minus_ar=paired("connectome", "ar"), connectome_minus_shuffled=paired_vs_mean("connectome", nulls_sh),
            connectome_minus_random_k=paired_vs_mean("connectome", nulls_rk), connectome_minus_geometry=paired("connectome", "geometry_knn"),
            geometry_minus_random_k=paired_vs_mean("geometry_knn", nulls_rk), var_dense_minus_connectome=paired("var_dense", "connectome"),
            ar_minus_persistence=paired("ar", "persistence")))
    out = dict(spec_sha_file=str(SPEC.name), per_animal={str(h): res[h] for h in H}, summary={str(h): summary[h] for h in H},
               alphas={str(h): alphas[h] for h in H}, seconds=round(time.time() - t0, 1))
    (OUT / "A2_results.json").write_text(json.dumps(out, indent=1, default=float))
    for h in H:
        print(f"h={h}:", {m: round(v['mean'], 3) for m, v in summary[h]['r2'].items() if not m[-1].isdigit()})
        print("   contrasts", {c: (round(v['mean'], 4), [round(x, 4) for x in v['ci']]) for c, v in summary[h]['contrasts'].items()})


if __name__ == "__main__":
    main()
