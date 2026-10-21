"""A1 (Atanas 2023): behaviour from neural activity on held-out animals, strictly past-only.

Targets on the neural time base (dt 0.6 s): velocity, head_curvature (R^2), reversal state = velocity < 0 (AUROC).
Horizon h: predict y(t + h) using features from samples <= t only.
  h = 0  (concurrent): neural lags 0..p-1 (includes frame t); behaviour AR lags 1..p (excludes y(t)).
  h >= 1 (forecast):   neural lags 0..p-1; behaviour AR lags 0..p-1.
Models: ar (behaviour history only), neural (all nodes), ar_neural (both). Ridge alpha chosen on an inner animal split.
Key estimand: increment of ar_neural over ar, per animal, animal bootstrap. Same frozen 5-fold animal split as A2.
Replication check: the 2026-09-16 past-only audit found concurrent +0.012 and past-only -0.005 for velocity.
"""
import json, os, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from wormlab.gd import preprocess as P
from wormlab.gd.bench import forecast as F, splits as SPL
from wormlab.gd.sources import atanas2023 as A

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/gd/benchmarks"
H = (0, 1, 4, 10); LAGS = 5; ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0); SEED = 20261008
TARGETS = ("velocity", "head_curvature", "reversal")


def features(X, M, y, h, kind):
    """Rows indexed by t; returns design (T, d), validity (T,), target y(t+h)."""
    T = len(y)
    cols, ok = [], np.ones(T, bool)
    if kind in ("neural", "ar_neural"):
        for l in range(LAGS):
            sh = np.zeros_like(X); sh[l:] = X[: T - l]
            cols.append(sh)
            if l:
                ok[:l] = False
    if kind in ("ar", "ar_neural"):
        start = 1 if h == 0 else 0
        for l in range(start, start + LAGS):
            sh = np.zeros(T); sh[l:] = y[: T - l]
            cols.append(sh[:, None]); ok[:l] = False
    Z = np.concatenate(cols, 1)
    tgt = np.full(T, np.nan); tgt[: T - h if h else T] = y[h:]
    ok &= np.isfinite(tgt) & np.isfinite(Z).all(1)
    return Z, ok, tgt


def ridge(Z, y, a):
    Z1 = np.c_[Z, np.ones(len(Z))]; R = a * np.eye(Z1.shape[1]); R[-1, -1] = 0
    return np.linalg.solve(Z1.T @ Z1 + R, Z1.T @ y)


def predict(c, Z):
    return np.c_[Z, np.ones(len(Z))] @ c


def target(rec, name):
    v = rec.behavior["velocity"]
    if name == "reversal":
        return (v < 0).astype(float)
    return rec.behavior[name].astype(float)


def score(y, p, name):
    if name == "reversal":
        return roc_auc_score(y, p) if 0 < y.mean() < 1 else None
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fit_eval(train, test, h, kind, name, rng):
    def stack(items):
        Zs, ys = [], []
        for X, M, y in items:
            Z, ok, t = features(X, M, y, h, kind)
            Zs.append(Z[ok]); ys.append(t[ok])
        return np.concatenate(Zs), np.concatenate(ys)
    idx = rng.permutation(len(train)); nv = max(1, len(train) // 4)
    Ztr, ytr = stack([train[i] for i in idx[nv:]])
    best = None
    for a in ALPHAS:
        c = ridge(Ztr, ytr, a)
        s = [score(t[ok], predict(c, Z[ok]), name) for Z, ok, t in (features(X, M, y, h, kind) for X, M, y in (train[i] for i in idx[:nv]))]
        s = np.mean([x for x in s if x is not None])
        if best is None or s > best[1]:
            best = (a, s)
    c = ridge(*stack(train), best[0])
    out = {}
    for an, (X, M, y) in test.items():
        Z, ok, t = features(X, M, y, h, kind)
        out[an] = score(t[ok], predict(c, Z[ok]), name)
    return out, best[0]


def main():
    t0 = time.time()
    recs = [P.standardize(r) for r in A.load()]
    nodes = P.node_space(recs, 0.5)
    split = SPL.load("A2_atanas_5fold")
    spec = dict(task="A1_atanas_behaviour", preprocess=P.PREPROCESS_VERSION, nodes=nodes, split_sha256=split["sha256"],
                horizons_frames=H, lags=LAGS, alpha_grid=ALPHAS, targets=TARGETS, models=["ar", "neural", "ar_neural"])
    p = OUT / "A1_spec.json"
    if not p.exists():
        p.write_text(json.dumps(dict(spec, frozen_utc=datetime.now(timezone.utc).isoformat()), indent=1)); os.chmod(p, 0o444)
    data = {}
    for r in recs:
        X, M = P.to_nodes(r, nodes)
        for name in TARGETS:
            data[(r.animal_id, name)] = (X, M, target(r, name))
    res = {}
    for name in TARGETS:
        for h in H:
            for kind in ("ar", "neural", "ar_neural"):
                per = {}
                for fold in range(split["n_folds"]):
                    tr = [data[(a, name)] for a, f in split["fold_of"].items() if f != fold]
                    te = {a: data[(a, name)] for a, f in split["fold_of"].items() if f == fold}
                    o, _ = fit_eval(tr, te, h, kind, name, np.random.default_rng(SEED + fold))
                    per.update(o)
                res[(name, h, kind)] = per
        print(name, "done", round(time.time() - t0), "s", flush=True)
    summary = {}
    for name in TARGETS:
        for h in H:
            ar, an = res[(name, h, "ar")], res[(name, h, "ar_neural")]
            inc = [an[a] - ar[a] for a in ar if ar[a] is not None and an[a] is not None]
            summary[f"{name}|h={h}"] = dict(ar=F.boot_mean(list(ar.values())), neural=F.boot_mean(list(res[(name, h, 'neural')].values())),
                                           ar_neural=F.boot_mean(list(an.values())), increment=F.boot_mean(inc))
    (OUT / "A1_results.json").write_text(json.dumps(dict(summary=summary, per_animal={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}), indent=1, default=float))
    for k, v in summary.items():
        print(k, {m: round(v[m]["mean"], 3) for m in ("ar", "neural", "ar_neural")}, "increment", round(v["increment"]["mean"], 4), [round(x, 4) for x in v["increment"]["ci"]])


if __name__ == "__main__":
    main()
