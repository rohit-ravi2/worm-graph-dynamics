"""Extract raw traces for one frozen partition. Confirmation outputs go to a sealed store: the driver
records only content hashes (SEAL.jsonl) and never computes or prints outcome statistics."""
import hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from wormlab.mechanism.randi import partitions, raw_traces

def _one(args):
    sub, out = args
    t0 = time.time()
    if out.exists():
        return dict(sub=sub, status="exists", sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    tmp = out.with_suffix(".tmp.npz")
    try:
        info = raw_traces.extract_animal(sub, tmp)
        tmp.replace(out)
        return dict(sub=sub, status="ok", sha256=hashlib.sha256(out.read_bytes()).hexdigest(), seconds=round(time.time() - t0, 1),
                    n_nuclei=info["n_nuclei"], n_events=info["n_events"])
    except Exception as e:  # recorded, not hidden
        return dict(sub=sub, status="failed", error=repr(e))

def _one_voxel(args):
    sub, payload, out = args
    from wormlab.mechanism.randi import voxel_x1
    t0 = time.time()
    if out.exists():
        return dict(sub=sub, status="exists", sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    try:
        tmp = out.with_suffix(".tmp")
        voxel_x1.extract_animal(sub, payload, tmp)
        tmp.replace(out)
        return dict(sub=sub, status="ok", sha256=hashlib.sha256(out.read_bytes()).hexdigest(), seconds=round(time.time() - t0, 1))
    except Exception as e:
        return dict(sub=sub, status="failed", error=repr(e))


def main_voxel(part, workers=3):
    from wormlab.mechanism.randi import voxel_x1
    add = json.loads((partitions.BENCH_DIR / "g2_addendum_manifest.json").read_text())
    assert add["voxel_extractor_sha256"] == voxel_x1.source_sha256(), "voxel extractor changed since g2 addendum freeze"
    sealed = part in ("CONFIRMATION", "RESERVE")
    base = partitions.BENCH_DIR / (f"sealed/{part.lower()}" if sealed else part.lower())
    d = base / "voxel_x1"
    d.mkdir(parents=True, exist_ok=True)
    m = partitions.load()
    jobs = [(s, base / "raw_traces" / f"{s}.npz", d / f"{s}.json") for s in m["partitions"][part] if (base / "raw_traces" / f"{s}.npz").exists()]
    with ProcessPoolExecutor(workers) as ex, open(d / "SEAL.jsonl", "a") as log:
        for r in ex.map(_one_voxel, jobs):
            r.update(partition=part, voxel_extractor_sha256=add["voxel_extractor_sha256"], addendum_sha256=add["addendum_sha256"])
            log.write(json.dumps(r) + "\n"); log.flush()
            print(r["sub"], r["status"], flush=True)


def main(part, workers=3):
    m = partitions.load()
    assert m["extractor"]["source_sha256"] == raw_traces.source_sha256(), "extractor source changed since partition freeze"
    sealed = part in ("CONFIRMATION", "RESERVE")
    d = partitions.BENCH_DIR / (f"sealed/{part.lower()}" if sealed else part.lower()) / "raw_traces"
    d.mkdir(parents=True, exist_ok=True)
    jobs = [(s, d / f"{s}.npz") for s in m["partitions"][part]]
    with ProcessPoolExecutor(workers) as ex, open(d / "SEAL.jsonl", "a") as log:
        for r in ex.map(_one, jobs):
            r.update(partition=part, extractor_sha256=m["extractor"]["source_sha256"], manifest_sha256=m["manifest_sha256"])
            if sealed:
                r = {k: v for k, v in r.items() if k in ("sub", "status", "sha256", "partition", "extractor_sha256", "manifest_sha256", "error", "seconds")}
            log.write(json.dumps(r) + "\n"); log.flush()
            print(r["sub"], r["status"], flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "voxel":
        main_voxel(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
