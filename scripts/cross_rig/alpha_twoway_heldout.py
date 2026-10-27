"""Held-out (CONFIRMATION, RESERVE) replication of the two-way copy-ratio estimator. EXPLORATORY grade: the W
residual family was already viewed post-adjudication on these partitions in g3; only the estimator is new.

Order enforced here:
 1. calibration gate (results/mechanism/cross_rig/twoway_calibration_n*.json): both pure-contamination worlds'
    two-way medians in [0.7, 1.4] in both windows, else refuse;
 2. write-once prereg (twoway_heldout_prereg.json, chmod 444) pinning the estimator source sha256 and the decision rule;
 3. hash-chained access ledger entry per partition BEFORE any sealed read;
 4. every payload's sha256 checked against its extraction SEAL.jsonl.
Decision rule (declared in the prereg): REPLICATES on a partition iff, in all four windows, the median two-way ratio lies
in [0.7, 1.4] and its animal-bootstrap CI contains 1.
"""
import glob, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_twoway_real as A  # noqa: E402
from wormlab.mechanism.firewall import AccessLedger  # noqa: E402
from wormlab.mechanism.randi import partitions as P  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/mechanism/cross_rig"
PREREG = OUT / "twoway_heldout_prereg.json"
RULE = "REPLICATES iff in all four windows the median two-way ratio is in [0.7, 1.4] and its animal-bootstrap 95% CI contains 1"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gate():
    f = sorted(OUT.glob("twoway_calibration_n*.json"))
    if not f:
        raise SystemExit("no calibration result")
    c = json.loads(f[-1].read_text())
    ok = all(0.7 <= c[f"{w}|{x}"]["twoway"] <= 1.4 for w in ("pure_equal_brightness", "pure_hetero_brightness") for x in ("spontaneous", "evoked"))
    return ok, f[-1].name, c


def freeze():
    if PREREG.exists():
        return json.loads(PREREG.read_text())
    ok, cal, c = gate()
    if not ok:
        raise SystemExit(f"calibration gate failed ({cal}); estimator not licensed for held-out use")
    body = dict(family="copy_ratio_twoway", grade="EXPLORATORY (W residuals viewed post-adjudication in g3)", partitions=["CONFIRMATION", "RESERVE"],
                estimator=dict(file="scripts/cross_rig/alpha_twoway_real.py", sha256=sha(REPO / "scripts/cross_rig/alpha_twoway_real.py")),
                runner_sha256=sha(__file__), calibration=cal, discovery_result=json.loads((OUT / "alpha_twoway_discovery.json").read_text()),
                decision_rule=RULE, frozen_utc=datetime.now(timezone.utc).isoformat())
    PREREG.write_text(json.dumps(body, indent=1)); os.chmod(PREREG, 0o444)
    return body


def seals(part):
    f = A.base_dir(part) / "provenance" / "SEAL.jsonl"
    return {r["sub"]: r["sha256"] for r in map(json.loads, f.read_text().splitlines()) if r["status"] in ("ok", "exists")}


def one(args):
    sub, part, want = args
    p = A.base_dir(part) / "provenance" / f"{sub}.npz"
    if sha(p) != want:
        raise RuntimeError(f"{p} does not match its extraction seal")
    return A.one(sub, part)


if __name__ == "__main__":
    pre = freeze()
    assert pre["estimator"]["sha256"] == sha(REPO / "scripts/cross_rig/alpha_twoway_real.py"), "estimator changed after freeze"
    led = AccessLedger(OUT / "access_ledger.jsonl")
    m = P.load(); out = {}
    rng = np.random.default_rng(0)
    for part in ("CONFIRMATION", "RESERVE"):
        sl = seals(part)
        led.append(role="EXPLORATORY_JUDGE", partition=part, family="copy_ratio_twoway", prereg_sha256=sha(PREREG), n_animals=len(m["partitions"][part]))
        with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
            res = list(ex.map(one, [(s, part, sl[s]) for s in m["partitions"][part]]))
        summ = {}
        for w in ("spontaneous", "pre", "evoked", "post_quiet"):
            per = [np.array(o[w]) for s, o in res if len(o[w])]
            idx = np.arange(len(per))
            med = lambda sel, k: float(np.median(np.concatenate([per[i][:, k] for i in sel])))
            bo = [med(rng.choice(idx, len(idx)), 1) for _ in range(400)]
            ci = [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))]
            summ[w] = dict(n_animals=len(per), n_pairs=int(sum(len(x) for x in per)), oneway=med(idx, 0), twoway=med(idx, 1), twoway_ci=ci,
                           ok=bool(0.7 <= med(idx, 1) <= 1.4 and ci[0] <= 1 <= ci[1]))
        summ["REPLICATES"] = all(v["ok"] for v in summ.values() if isinstance(v, dict))
        out[part] = summ
        led.append(role="EXPLORATORY_JUDGE", partition=part, family="copy_ratio_twoway", result_sha256=hashlib.sha256(json.dumps(summ, sort_keys=True).encode()).hexdigest())
    (OUT / "twoway_heldout_result.json").write_text(json.dumps(dict(prereg=str(PREREG.name), rule=RULE, results=out), indent=1))
    print(json.dumps(out, indent=1))
