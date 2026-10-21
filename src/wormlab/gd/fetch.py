"""Content-verified retrieval of public files: cache -> mirrors -> download. Every file is checked by sha256."""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

from .config import load_config


class IntegrityError(RuntimeError):
    pass


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ensure_file(dataset: str, name: str, url: str, sha256: str, *, download: bool = True) -> Path:
    """Return a local path to ``name`` whose content hash is ``sha256``.

    Looks in <cache>/<dataset>/<name>, then in each configured mirror (by file name, recursively one level), then
    downloads to the cache. A mirror hit is linked into the cache (symlink) after verification, so the cache is the
    single place downstream code reads from.
    """
    cfg = load_config()
    target = cfg.cache_dir / dataset / name
    if target.exists():
        if sha256_file(target) == sha256:
            return target
        raise IntegrityError(f"{target} exists but its sha256 does not match the manifest")
    for m in cfg.mirrors:
        for cand in (m / name, *m.glob(f"*/{name}")):
            if cand.is_file() and sha256_file(cand) == sha256:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(cand.resolve(), target)
                return target
    if not download:
        raise FileNotFoundError(f"{dataset}/{name} not in cache or mirrors and download=False")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(part, "wb") as out:
        shutil.copyfileobj(r, out, 1 << 20)
    got = sha256_file(part)
    if got != sha256:
        part.unlink()
        raise IntegrityError(f"download of {url} has sha256 {got[:16]}..., manifest says {sha256[:16]}...")
    part.rename(target)
    return target
