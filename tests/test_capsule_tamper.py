"""A capsule verifies when untouched and fails after a one-byte change to any stored blob."""
from __future__ import annotations

import json

from wormlab.evidence import ArtifactSpec, create_capsule, verify
from wormlab.evidence.capsule import ROLE_CODE, ROLE_SPEC, ROLE_SUMMARY


def _capsule(tmp_path):
    spec = tmp_path / "spec.json"; spec.write_text(json.dumps({"task": "toy", "h": 4}))
    res = tmp_path / "results.json"; res.write_text(json.dumps({"r2": 0.5776}))
    code = tmp_path / "run.py"; code.write_text("print('run')\n")
    return create_capsule(
        tmp_path / "cap", capsule_id="toy", replay_command="python run.py",
        spec=ArtifactSpec(ROLE_SPEC, "spec.json", spec),
        summary=ArtifactSpec(ROLE_SUMMARY, "results.json", res),
        code_snapshot=[ArtifactSpec(ROLE_CODE, "run.py", code)])


def test_untouched_capsule_verifies(tmp_path):
    cap = _capsule(tmp_path)
    assert verify(cap.dir)


def test_one_byte_change_fails(tmp_path):
    cap = _capsule(tmp_path)
    blob = cap.blob_path(cap.artifacts_of(ROLE_SUMMARY)[0])
    blob.chmod(0o644)
    b = bytearray(blob.read_bytes()); b[0] ^= 1; blob.write_bytes(bytes(b))
    rep = verify(cap.dir)
    assert not rep and rep.failures


def test_editing_the_source_after_freeze_changes_nothing(tmp_path):
    cap = _capsule(tmp_path)
    (tmp_path / "results.json").write_text(json.dumps({"r2": 0.99}))
    assert verify(cap.dir)
