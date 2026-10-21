"""Canonical C. elegans neuron identities and normalisation of dataset labels.

Canonical names are the 300 neuron ids shipped with the ``wormneuroatlas`` package (White/Witvliet naming; the two
CAN cells are absent). A raw dataset label is normalised to one of:

* ``canonical`` -- an exact canonical neuron (e.g. ``AVAL``);
* ``class_only`` -- a bilateral class without a side (``AVA`` -> candidates AVAL/AVAR);
* ``ambiguous`` -- several explicit candidates (Kato cells, or ``AVAL/AVAR``);
* ``uncertain`` -- a single candidate marked with ``?`` by the annotator;
* ``unknown`` -- empty, numeric or unrecognised.
Only ``canonical`` labels enter graph models by default; the rest are kept so nothing is silently dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import h5py


@lru_cache(maxsize=1)
def canonical_names() -> tuple[str, ...]:
    with resources.as_file(resources.files("wormneuroatlas") / "data" / "funatlas.h5") as p, h5py.File(p) as h:
        return tuple(x.decode() for x in h["neuron_ids"][:])


@lru_cache(maxsize=1)
def _name_set() -> frozenset:
    return frozenset(canonical_names())


def homolog(name: str) -> str | None:
    """Left/right partner of a canonical neuron, if it exists."""
    if name and name[-1] in "LR":
        other = name[:-1] + ("R" if name[-1] == "L" else "L")
        if other in _name_set():
            return other
    return None


def neuron_class(name: str) -> str:
    """Bilateral class (AVAL -> AVA); unpaired or numbered cells are their own class."""
    return name[:-1] if homolog(name) else name


@dataclass(frozen=True)
class Label:
    raw: str
    status: str                    # canonical | class_only | ambiguous | uncertain | unknown
    name: str | None               # canonical neuron if status == canonical
    candidates: tuple[str, ...] = ()


def normalize(raw, extra_candidates=()) -> Label:
    s = "" if raw is None else str(raw).strip()
    names = _name_set()
    cands = [str(c).strip().upper() for c in extra_candidates if str(c).strip()]
    if not s and not cands:
        return Label("", "unknown", None)
    uncertain = "?" in s
    tokens = [t for t in re.split(r"[\s/,;|]+", s.replace("?", "").upper()) if t] + cands
    tokens = list(dict.fromkeys(tokens))
    exact = [t for t in tokens if t in names]
    if len(tokens) == 1 and exact:
        return Label(s, "uncertain" if uncertain else "canonical", None if uncertain else exact[0], tuple(exact))
    if len(exact) > 1:
        return Label(s, "ambiguous", None, tuple(exact))
    if len(tokens) == 1:
        t = tokens[0]
        side = tuple(x for x in (t + "L", t + "R") if x in names)
        if side:
            return Label(s, "uncertain" if uncertain else "class_only", None, side)
    if exact:
        return Label(s, "ambiguous", None, tuple(exact))
    return Label(s, "unknown", None)
