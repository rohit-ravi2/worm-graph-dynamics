"""Where WormLab keeps data. Nothing here hard-codes a machine path.

Resolution order for the config file: $WORMLAB_CONFIG, ./wormlab.local.toml (gitignored), ~/.config/wormlab/config.toml.
Keys: ``cache_dir`` (default ~/.cache/wormlab) and ``mirrors`` (directories searched, by file name, for a file whose
sha256 matches the source manifest before anything is downloaded).
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    cache_dir: Path
    mirrors: tuple[Path, ...] = field(default_factory=tuple)
    source: str = "defaults"


def _candidates() -> list[Path]:
    out = []
    if os.environ.get("WORMLAB_CONFIG"):
        out.append(Path(os.environ["WORMLAB_CONFIG"]))
    out.append(Path.cwd() / "wormlab.local.toml")
    out.append(Path.home() / ".config" / "wormlab" / "config.toml")
    return out


def load_config() -> Config:
    for p in _candidates():
        if p.is_file():
            d = tomllib.loads(p.read_text())
            cache = Path(os.path.expanduser(d.get("cache_dir", "~/.cache/wormlab")))
            mirrors = tuple(Path(os.path.expanduser(m)) for m in d.get("mirrors", []))
            return Config(cache, mirrors, str(p))
    return Config(Path.home() / ".cache" / "wormlab")
