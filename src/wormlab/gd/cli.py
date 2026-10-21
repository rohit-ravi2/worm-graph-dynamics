"""``wormlab gd`` subcommands (registered by wormlab.cli)."""
from __future__ import annotations

import json


def add_parser(sub):
    p = sub.add_parser("gd", help="Graph Dynamics: portable data -> frozen benchmark -> verified capsule")
    g = p.add_subparsers(dest="gd_cmd", required=True)
    d = g.add_parser("data", help="build / verify the public-data caches")
    d.set_defaults(func=_data)
    r = g.add_parser("run", help="run a frozen benchmark and capsule it")
    r.add_argument("bench", choices=["A1", "A2", "K1", "L2"])
    r.add_argument("--rerun", action="store_true")
    r.set_defaults(func=_run)
    v = g.add_parser("verify", help="verify a capsule directory")
    v.add_argument("capsule")
    v.set_defaults(func=_verify)


def _data(args):
    from .workflow import data_manifests
    print(json.dumps([str(m) for m in data_manifests()], indent=1)); return 0


def _run(args):
    from .workflow import run
    print(json.dumps(run(args.bench, args.rerun), indent=1)); return 0


def _verify(args):
    from ..evidence.capsule import verify
    rep = verify(args.capsule)
    print(json.dumps(dict(ok=bool(rep)), indent=1)); return 0 if rep else 7
