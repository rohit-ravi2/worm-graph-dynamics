"""Freeze a finished mechanism run into an immutable EvidenceCapsule (copy-on-freeze, content-addressed)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ..evidence.capsule import ROLE_CODE, ROLE_DEVIATION, ROLE_ENV, ROLE_INPUT_MANIFEST, ROLE_OUTPUT, ROLE_PLAN, ROLE_SPEC, ROLE_SUMMARY, ArtifactSpec, create_capsule, verify

REPO = Path(__file__).resolve().parents[3]


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:  # recorded, not hidden
        return f"UNAVAILABLE: {e}"


def build(run, capsule_dir: Path, *, capsule_id: str, extra_inputs: list[Path] = (), payload_dirs: list[Path] = (), description: str = "",
          parent: Path | None = None) -> dict:
    out = run.run_dir / "outputs"
    paths = run.write_outputs()
    meta = dict(git_head=_git("rev-parse", "HEAD"), git_status_porcelain=_git("status", "--porcelain", "--", "src/wormlab/mechanism", "scripts"),
                python=sys.version, run_config=run.config.__dict__)
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=1, sort_keys=True, default=str))
    plan = dict(tests=[t.content() | {"spec_hash": t.spec_hash} for t in run.tests], compiled=run.compiled, config=run.config.__dict__,
                roles={"scout": run.config.scout_partition, "judge": run.config.judge_partition})
    (out / "execution_plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True, default=str))
    env = run.run_dir / "environment_lock.txt"
    env.write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    code = sorted((REPO / "src/wormlab/mechanism").rglob("*.py")) + [REPO / "scripts/mechanism_extract.py", REPO / "scripts/mechanism_run_randi.py", REPO / "src/wormlab/cli.py",
                                                                    REPO / "tests/test_mechanism_synthetic.py"]
    code = [c for c in code if c.exists() and "__pycache__" not in str(c)]
    skip = {"hypotheses.json", "execution_plan.json", "verdict.json", "deviations.json"}
    outputs = [ArtifactSpec(ROLE_OUTPUT, f"outputs/{p.name}", p) for p in sorted(out.iterdir()) if p.name not in skip]
    outputs += [ArtifactSpec(ROLE_OUTPUT, p.name, p) for p in (run.run_dir / "access_ledger.jsonl", run.run_dir / "journal.jsonl") if p.exists()]
    inputs = [ArtifactSpec(ROLE_INPUT_MANIFEST, str(Path(p).relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p), p) for p in extra_inputs]
    for d in payload_dirs:
        for f in sorted(Path(d).glob("*")):
            if f.is_file():
                inputs.append(ArtifactSpec(ROLE_INPUT_MANIFEST, str(f.relative_to(REPO)), f))
    cap = create_capsule(
        capsule_dir, capsule_id=capsule_id, description=description,
        replay_command="python scripts/mechanism_run_randi.py --replay <capsule>",
        spec=ArtifactSpec(ROLE_SPEC, "outputs/hypotheses.json", out / "hypotheses.json"),
        execution_plan=ArtifactSpec(ROLE_PLAN, "outputs/execution_plan.json", out / "execution_plan.json"),
        environment_lock=ArtifactSpec(ROLE_ENV, "environment_lock.txt", env),
        summary=ArtifactSpec(ROLE_SUMMARY, "outputs/verdict.json", out / "verdict.json"),
        input_manifests=inputs, code_snapshot=[ArtifactSpec(ROLE_CODE, str(c.relative_to(REPO)), c) for c in code],
        outputs=outputs, deviations=[ArtifactSpec(ROLE_DEVIATION, "outputs/deviations.json", out / "deviations.json")],
        parent=parent,
    )
    rep = verify(capsule_dir)
    return dict(root=cap.root_hash, verified=bool(rep), report=rep.__dict__ if hasattr(rep, "__dict__") else str(rep))
