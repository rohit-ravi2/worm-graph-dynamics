"""Measurement model + observation-time (scan-timing) correction.

``MeasurementModelSpec`` records how observations were generated. Every field carries a provenance
status; inferred or assumed geometry is never silently promoted to ground truth:

    KNOWN_FROM_METADATA  read from acquisition metadata / deposited files
    INFERRED_FROM_DATA   estimated from the data (must cite the estimating artifact)
    ASSUMED              adopted without direct evidence (must cite why)
    UNKNOWN              not available; tests requiring it cannot claim what it would support

Scan timing
-----------
In a volumetric scan each plane is imaged at its own time. A nucleus sampled late in a sweep cannot be
compared naively with one sampled early. Onsets are therefore estimated from per-sample acquisition
times and reported as INTERVALS: the onset lies in (t_last_below, t_first_above]. A receiver-minus-target
delay is the interval difference, and temporal precedence is only asserted when the whole delay interval
excludes zero (``delay_call``). Precedence below the acquisition resolution is never claimed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .spec import SpecError, canonical_sha

KNOWN = "KNOWN_FROM_METADATA"
INFERRED = "INFERRED_FROM_DATA"
ASSUMED = "ASSUMED"
UNKNOWN = "UNKNOWN"
STATUSES = (KNOWN, INFERRED, ASSUMED, UNKNOWN)


@dataclass
class MeasurementField:
    value: object
    status: str
    source: str = ""
    uncertainty: object = None
    note: str = ""

    def validate(self, name: str) -> None:
        if self.status not in STATUSES:
            raise SpecError(f"measurement field {name}: bad status {self.status!r}")
        if self.status in (INFERRED, ASSUMED, KNOWN) and not self.source:
            raise SpecError(f"measurement field {name}: status {self.status} requires a source")
        if self.status == UNKNOWN and self.value not in (None, "UNKNOWN"):
            raise SpecError(f"measurement field {name}: UNKNOWN fields must not carry a value")


@dataclass
class MeasurementModelSpec:
    model_id: str
    dataset: str
    fields: dict = field(default_factory=dict)      # name -> MeasurementField

    def validate(self) -> None:
        for k, f in self.fields.items():
            (f if isinstance(f, MeasurementField) else MeasurementField(**f)).validate(k)

    def get(self, name: str) -> MeasurementField:
        f = self.fields.get(name)
        if f is None:
            return MeasurementField(None, UNKNOWN, note="field not declared")
        return f if isinstance(f, MeasurementField) else MeasurementField(**f)

    def available(self, names: list[str]) -> dict:
        """Which required fields are usable. UNKNOWN blocks; ASSUMED/INFERRED are usable but flagged."""
        out = {}
        for n in names:
            f = self.get(n)
            out[n] = f.status
        return out

    def content(self) -> dict:
        return {"model_id": self.model_id, "dataset": self.dataset,
                "fields": {k: asdict(v) if isinstance(v, MeasurementField) else v for k, v in sorted(self.fields.items())}}

    @property
    def spec_hash(self) -> str:
        return canonical_sha(self.content())


# ----------------------------------------------------------------------------------------------
# onset estimation on irregular, plane-resolved sample times
# ----------------------------------------------------------------------------------------------
ONSET_K_SD = 2.0   # chosen on DISCOVERY detection coverage/specificity only (59% of top-30% amplitude targets detected,
                   # 0% of bottom-30%; k=3 detected 33%); never on a latency outcome. See docs/mechanism_engine/MEASUREMENT_MODEL.md


def onset_interval(t: np.ndarray, y: np.ndarray, t_stim: float, *, n_base: int, k_sd: float = ONSET_K_SD,
                   sustain: int = 2, t_max: float = 8.0) -> tuple[float, float] | None:
    """Onset interval (t_last_below, t_first_above] after ``t_stim``, or None if no onset.

    ``t`` are per-sample acquisition times (seconds) for THIS signal, ``y`` its values. Baseline = samples
    acquired strictly before the stimulus among the first ``n_base + 2``. Onset = first post-stimulus sample that
    starts a run of ``sustain`` samples above baseline mean + k_sd * baseline SD.
    """
    t = np.asarray(t, float); y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    pre = ok & (t < t_stim)
    pre[n_base + 2:] = False
    base = y[pre]
    if base.size < 3:
        return None
    thr = base.mean() + k_sd * max(base.std(ddof=1), 1e-9)
    post = np.where(ok & (t >= t_stim) & (t <= t_stim + t_max))[0]
    if post.size < sustain:
        return None
    above = y[post] > thr
    for i in range(len(post) - sustain + 1):
        if above[i:i + sustain].all():
            j = post[i]
            prev = np.where(ok[:j])[0]
            t_lo = max(t[prev[-1]], t_stim) if prev.size else t_stim
            return (float(t_lo), float(t[j]))
    return None


def delay_interval(receiver: tuple[float, float], target: tuple[float, float]) -> tuple[float, float]:
    """Interval of possible receiver-minus-target onset delays."""
    return (receiver[0] - target[1], receiver[1] - target[0])


def delay_call(intervals: list[tuple[float, float]], *, min_fraction: float = 0.5) -> dict:
    """Aggregate per-event delay intervals without claiming sub-resolution precedence.

    Each interval is LAG (lo > 0), LEAD (hi < 0) or UNRESOLVED (straddles 0). Reports fractions,
    the median interval width (effective temporal resolution) and the median midpoint.
    """
    if not intervals:
        return dict(n=0, frac_lag=np.nan, frac_lead=np.nan, frac_unresolved=np.nan, resolution_s=np.nan, median_mid_s=np.nan)
    iv = np.asarray(intervals, float)
    lag = iv[:, 0] > 0
    lead = iv[:, 1] < 0
    return dict(n=int(len(iv)), frac_lag=float(lag.mean()), frac_lead=float(lead.mean()),
                frac_unresolved=float((~lag & ~lead).mean()), resolution_s=float(np.median(iv[:, 1] - iv[:, 0])),
                median_mid_s=float(np.median(iv.mean(1))))


def naive_volume_onset(vol_idx: np.ndarray, y: np.ndarray, stim_vol: int, *, n_base: int, k_sd: float = 3.0, sustain: int = 2):
    """The UNCORRECTED estimator (volume index only) — kept to demonstrate the scan-order artifact."""
    base = y[:n_base]
    thr = base.mean() + k_sd * max(base.std(ddof=1), 1e-9)
    for i in range(n_base, len(y) - sustain + 1):
        if vol_idx[i] >= stim_vol and (y[i:i + sustain] > thr).all():
            return int(vol_idx[i])
    return None
