"""Extract provenance_traces/1.0 for one frozen partition (g3 signal-provenance campaign).

Order of guards: the g3 addendum must pin the current extractor hash; for CONFIRMATION/RESERVE the source raw_traces
payload must match its SEAL.jsonl hash; outputs of sealed partitions go to sealed/<part>/provenance with a SEAL.jsonl
that records only content hashes. No statistic is computed or printed."""
import hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from wormlab.mechanism.randi import partitions, provenance_traces as PT

ADDENDUM = partitions.BENCH_DIR / "g3_provenance_manifest.json"


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def freeze_addendum():
    if ADDENDUM.exists():
        raise FileExistsError(ADDENDUM)
    m = partitions.load()
    body = dict(generation="g3", purpose="signal-provenance campaign: continuous ROI traces + static mean volume",
                extractor=dict(module="wormlab.mechanism.randi.provenance_traces", version=PT.VERSION, source_sha256=PT.source_sha256()),
                parent_manifest_sha256=m["manifest_sha256"], partitions_reused=True,
                note="Families, statistics and call rules are frozen separately (g3 preregistration) before any JUDGE access.")
    body["addendum_sha256"] = partitions.canonical_sha(body)
    ADDENDUM.write_text(json.dumps(body, indent=1, sort_keys=True))
    ADDENDUM.chmod(0o444)
    return body


def _one(args):
    sub, payload, pseal, out = args
    t0 = time.time()
    if out.exists():
        return dict(sub=sub, status="exists", sha256=_sha(out))
    try:
        if pseal is not None and _sha(payload) != pseal:
            return dict(sub=sub, status="failed", error="source payload does not match raw_traces seal")
        tmp = out.with_name(out.stem + ".tmp.npz")
        PT.extract_animal(sub, payload, tmp)
        tmp.replace(out)
        return dict(sub=sub, status="ok", sha256=_sha(out), seconds=round(time.time() - t0, 1))
    except Exception as e:
        return dict(sub=sub, status="failed", error=repr(e))


def main(part, workers=8):
    add = json.loads(ADDENDUM.read_text())
    assert add["extractor"]["source_sha256"] == PT.source_sha256(), "provenance extractor changed since g3 addendum freeze"
    m = partitions.load()
    sealed = part in ("CONFIRMATION", "RESERVE")
    base = partitions.BENCH_DIR / (f"sealed/{part.lower()}" if sealed else part.lower())
    seals = {}
    if sealed:
        for line in (base / "raw_traces" / "SEAL.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("status") in ("ok", "exists"):
                seals[r["sub"]] = r["sha256"]
    d = base / "provenance"
    d.mkdir(parents=True, exist_ok=True)
    jobs = [(s, base / "raw_traces" / f"{s}.npz", seals.get(s) if sealed else None, d / f"{s}.npz") for s in m["partitions"][part]
            if (base / "raw_traces" / f"{s}.npz").exists()]
    with ProcessPoolExecutor(workers) as ex, open(d / "SEAL.jsonl", "a") as log:
        for r in ex.map(_one, jobs):
            r.update(partition=part, extractor_sha256=add["extractor"]["source_sha256"], addendum_sha256=add["addendum_sha256"])
            log.write(json.dumps(r) + "\n"); log.flush()
            print(r["sub"], r["status"], r.get("seconds"), r.get("error", ""), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        print(json.dumps(freeze_addendum(), indent=1))
    else:
        main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
