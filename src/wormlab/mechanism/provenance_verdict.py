"""Preregistered call rules and near-field provenance classification for the signal-provenance families.

Frozen before any JUDGE access (g3 preregistration); validated on hostile synthetic fixtures S14-S24 first.
Calls use the family estimate's 95% animal-cluster bootstrap CI against a minimum effect:
  POS: ci_lo > +m   NEG: ci_hi < -m   NULL: -m <= ci_lo and ci_hi <= +m   INCONCLUSIVE otherwise.
"""
from __future__ import annotations

import numpy as np

MIN_EFFECT = {
    "T13_optical_copy_subtracted_proximity": 0.03,   # risk difference (same floor as T0)
    "T17_unmixed_proximity": 0.03,                   # risk difference
    "T14_brightness_elasticity": 0.05,               # receiver dF/F per unit target dF/F
    "T16_spontaneous_excess": 0.5,                   # mean standardised covariance excess (z units)
    "T6u_unmixed_target_flat": 0.03,
    "T20_axial_lateral_anisotropy": 0.03,
    "T20u_axial_lateral_anisotropy_unmixed": 0.03,
}
HALO_PRESENT = 0.03          # median halo/target at the 2-3 um knots
T15_OK = (0.6, 1.4)          # optical-profile calibration accepted range for the estimate


def call(family: str, res: dict) -> str:
    m = MIN_EFFECT[family]
    ci = res.get("ci", [np.nan, np.nan])
    if ci is None or not np.all(np.isfinite(ci)):
        return "INCONCLUSIVE"
    lo, hi = ci
    if lo > m:
        return "POS"
    if hi < -m:
        return "NEG"
    if lo >= -m and hi <= m:
        return "NULL"
    return "INCONCLUSIVE"


def classify(fam: dict, halo: dict | None = None) -> dict:
    """Near-field provenance class from family results. Primary: T17 (unmixed). Sub-classification of own-indicator
    responses by T16 (spontaneous coupling present?)."""
    c = {k: call(k, v) for k, v in fam.items() if k in MIN_EFFECT}
    t17 = c.get("T17_unmixed_proximity", "INCONCLUSIVE")
    t16 = c.get("T16_spontaneous_excess", "INCONCLUSIVE")
    halo_present = None
    if halo and halo.get("n"):
        h = halo["median_halo_over_target"]
        halo_present = bool(np.mean(h[:2]) > HALO_PRESENT)
    ef = fam.get("T17_unmixed_proximity", {}).get("explained_fraction_ci", [np.nan, np.nan])
    if t17 == "NULL" or t17 == "NEG":
        label = "CONTAMINATION_SUFFICIENT"
        sub = "halo_or_extended_kernel" if (halo_present or c.get("T13_optical_copy_subtracted_proximity") == "POS") else "static_kernel"
        if t17 == "NEG":
            sub += "|over_subtraction_warning"
    elif t17 == "POS":
        mixture = bool(np.all(np.isfinite(ef)) and ef[0] > 0.2 and ef[1] < 0.9)
        if t16 == "POS":
            label = "OWN_INDICATOR_WITH_SPONTANEOUS_COUPLING"   # H3 or H10 (topology decides)
        elif t16 == "NULL":
            label = "OWN_INDICATOR_STIM_LOCKED"                 # H2 or thresholded H3: non-identifiable by these families
        else:
            label = "OWN_INDICATOR_UNSPECIFIED"
        sub = "mixture_with_contamination" if mixture else "own_dominant"
    else:
        label, sub = "UNDERPOWERED_OR_AMBIGUOUS", ""
    return dict(label=label, sub=sub, calls=c, halo_present=halo_present)
