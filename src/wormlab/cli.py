"""Command-line entry point: ``wormlab gd data | run <BENCH> | verify <DIR>``."""
from __future__ import annotations

import argparse
import sys

from .gd import cli as gd_cli


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wormlab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    gd_cli.add_parser(sub)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
