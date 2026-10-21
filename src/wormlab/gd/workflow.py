"""One command from public data to a verified evidence capsule.

    wormlab gd data            build / verify the Atanas and Kato caches (downloads if missing)
    wormlab gd run <BENCH>     verify data -> run the frozen benchmark if its results are missing (or --rerun)
                               -> freeze spec, split, results, data manifests, code, environment into a capsule -> verify
    wormlab gd verify <DIR>    re-verify a capsule (one-byte tamper fails)
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..evidence.capsule import (ROLE_CODE, ROLE_ENV, ROLE_INPUT_MANIFEST, ROLE_OUTPUT, ROLE_PLAN, ROLE_SPEC, ROLE_SUMMARY,
                                ArtifactSpec, create_capsule, verify)
from .fetch import ensure_file
from .sources import atanas2023, kato2015

REPO = Path(__file__).resolve().parents[3]
B = REPO / "results/gd/benchmarks"
BENCHMARKS = {
    "A1": dict(cmd=[["scripts/gd/run_A1.py"]], spec=B / "A1_spec.json", results=B / "A1_results.json", split="A2_atanas_5fold",
               data=("atanas2023",)),
    "A2": dict(cmd=[["scripts/gd/run_A2.py"]], spec=B / "A2_spec.json", results=B / "A2_results.json", split="A2_atanas_5fold",
               data=("atanas2023",)),
    "K1": dict(cmd=[["scripts/gd/run_K1.py"]], spec=B / "K1_spec.json", results=B / "K1_results.json", split=None,
               data=("kato2015", "atanas2023")),
    "L2": dict(cmd=[["scripts/gd/run_L2.py", str(f)] for f in range(5)] + [["scripts/gd/run_L2.py", "aggregate"]],
               spec=B / "L2/L2_spec.json", results=B / "L2/L2_results.json", split="A2_atanas_5fold", data=("atanas2023",)),
}


def data_manifests() -> list[Path]:
    """Verify every source and write its manifest; returns manifest paths."""
    out = REPO / "results/gd/data_manifests"; out.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in kato2015.MANIFEST.items():
        ensure_file(kato2015.DATASET, name, url, sha)
    (out / "kato2015.json").write_text(json.dumps(dict(source=f"OSF:{kato2015.OSF_NODE}", files=kato2015.MANIFEST), indent=1))
    if not (atanas2023._dir() / "MANIFEST.json").exists():
        atanas2023.build_cache()
    atanas2023.verify_cache()
    (out / "atanas2023.json").write_text((atanas2023._dir() / "MANIFEST.json").read_text())
    (out / "atanas2023_asset_index.json").write_text((atanas2023._dir() / "asset_index.json").read_text())
    return sorted(out.glob("*.json"))


def run(bench: str, rerun: bool = False) -> dict:
    b = BENCHMARKS[bench]
    manifests = data_manifests()
    if rerun or not b["results"].exists():
        for c in b["cmd"]:
            subprocess.run([sys.executable, *c], cwd=REPO, check=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cdir = REPO / "results/_capsules" / f"gd_{bench}_{ts}"
    env = REPO / "results/gd/logs" / f"env_{bench}_{ts}.txt"
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    plan = REPO / "results/gd/logs" / f"plan_{bench}_{ts}.json"
    plan.write_text(json.dumps(dict(benchmark=bench, commands=b["cmd"], git_head=head, python=sys.version), indent=1))
    code = [p for p in sorted((REPO / "src/wormlab/gd").rglob("*.py")) + sorted((REPO / "scripts/gd").glob("*.py")) if "__pycache__" not in str(p)]
    inputs = [ArtifactSpec(ROLE_INPUT_MANIFEST, str(m.relative_to(REPO)), m) for m in manifests]
    if b["split"]:
        sp = REPO / "results/gd/splits" / f"{b['split']}.json"
        inputs.append(ArtifactSpec(ROLE_INPUT_MANIFEST, str(sp.relative_to(REPO)), sp))
    extra = [p for p in sorted(b["results"].parent.glob("fold*.json"))] if bench == "L2" else []
    cap = create_capsule(
        cdir, capsule_id=f"gd_{bench}_{ts}", description=f"WormLab Graph Dynamics benchmark {bench}",
        replay_command=f"wormlab gd run {bench} --rerun",
        spec=ArtifactSpec(ROLE_SPEC, str(b["spec"].relative_to(REPO)), b["spec"]),
        execution_plan=ArtifactSpec(ROLE_PLAN, str(plan.relative_to(REPO)), plan),
        environment_lock=ArtifactSpec(ROLE_ENV, str(env.relative_to(REPO)), env),
        summary=ArtifactSpec(ROLE_SUMMARY, str(b["results"].relative_to(REPO)), b["results"]),
        input_manifests=inputs, code_snapshot=[ArtifactSpec(ROLE_CODE, str(c.relative_to(REPO)), c) for c in code],
        outputs=[ArtifactSpec(ROLE_OUTPUT, str(p.relative_to(REPO)), p) for p in extra])
    rep = verify(cdir)
    return dict(capsule=str(cdir), root=cap.root_hash, verified=bool(rep))
