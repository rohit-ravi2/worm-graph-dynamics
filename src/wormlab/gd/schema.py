"""Typed recording: one animal, one session, multivariate and masked.

``neural`` is (T, N) on time base ``t``; ``mask`` marks valid samples. Each neuron carries an ``identity.Label``.
``behavior`` channels are resampled onto ``t`` by the source adapter (originals kept in ``behavior_native``).
``positions_um`` (N, 3) is optional per-animal geometry. ``events`` hold interval/impulse annotations (stimuli,
states, interventions). ``provenance`` records the exact source objects and their content hashes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .identity import Label


@dataclass
class Event:
    kind: str            # e.g. "stimulus", "state", "intervention"
    name: str            # e.g. "O2_up", "FWD", "AVA_HisCl"
    t_start: float
    t_stop: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Recording:
    dataset: str
    animal_id: str
    condition: str
    t: np.ndarray
    neural: np.ndarray
    mask: np.ndarray
    labels: list[Label]
    signal_kind: str                       # what ``neural`` holds, e.g. "dF/F0 (bleach-corrected)"
    behavior: dict[str, np.ndarray] = field(default_factory=dict)
    behavior_native: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    positions_um: np.ndarray | None = None
    events: list[Event] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ checks
    def validate(self) -> "Recording":
        T, N = self.neural.shape
        assert self.t.shape == (T,), "t must match neural rows"
        assert np.all(np.diff(self.t) > 0), "t must be strictly increasing"
        assert self.mask.shape == (T, N) and self.mask.dtype == bool
        assert len(self.labels) == N, "one Label per neuron"
        assert not np.any(np.isnan(self.neural[self.mask])), "masked-valid samples must be finite"
        for k, v in self.behavior.items():
            assert v.shape == (T,), f"behavior {k!r} must be on the neural time base"
        if self.positions_um is not None:
            assert self.positions_um.shape == (N, 3)
        return self

    # ------------------------------------------------------------- accessors
    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.t)))

    def canonical_index(self) -> dict[str, int]:
        """canonical neuron name -> column (only status == canonical; duplicates dropped, flagged in provenance)."""
        out, dup = {}, set()
        for j, lab in enumerate(self.labels):
            if lab.status == "canonical":
                if lab.name in out:
                    dup.add(lab.name)
                out[lab.name] = j
        for d in dup:
            out.pop(d, None)
        return out

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for a in (self.t, self.neural, self.mask):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(json.dumps([[l.raw, l.status, l.name] for l in self.labels]).encode())
        for k in sorted(self.behavior):
            h.update(k.encode()); h.update(np.ascontiguousarray(self.behavior[k]).tobytes())
        return h.hexdigest()

    def summary(self) -> dict:
        st = {}
        for l in self.labels:
            st[l.status] = st.get(l.status, 0) + 1
        return dict(dataset=self.dataset, animal=self.animal_id, condition=self.condition, T=int(self.neural.shape[0]),
                    N=int(self.neural.shape[1]), dt_s=round(self.dt, 4), label_status=st, behavior=sorted(self.behavior),
                    n_events=len(self.events), has_positions=self.positions_um is not None)
