"""Experiment resolution modelling (deep mechanism campaign, sections 32-35 and 44-46).

Pure simulation and design. Nothing in this module reads Randi neural outcomes.

The question this module answers: given an experiment (acquisition, indicator, readout, genotype, sample size), how
well can a named discriminating CONTRAST separate a named HYPOTHESIS PAIR? It is a forward model plus a compiler:

    stimulus -> target voltage-like drive -> mechanism (H1..H7) -> neighbour drive / optical copy
             -> calcium + indicator kinetics (per cell) -> exposure integration at per-plane sample times
             -> photon shot + read + slow biological noise -> estimator -> call vs floor -> power / MDE

Hypothesis labels follow ``randi/benchmark.py`` families:
    H1 static measurement contamination (optical copy of target photons into the neighbour ROI; zero lag, exact copy)
    H2 direct off-target photostimulation (neighbour driven by the light, own ~0 latency, independent amplitude)
    H3 gap-junction / local electrical coupling (neighbour's OWN response, voltage latency L + coupling low-pass)
    H4 direct chemical synapse (latency 100 ms - 2 s, synaptic low-pass)
    H5 polysynaptic propagation (longer latency, broader low-pass)
    H7 global / shared state (not target-locked: random latency, independent amplitude)

Provenance discipline follows ``measurement.py``: every numeric preset carries a status (ASSUMED unless a repo artifact
supports it), a source and a literature/plausibility range ``(low, high)``. Indicator values are literature RECALL and
must be reconciled with ``docs/mechanism_engine/deep_campaign/BIOPHYSICAL_PARAMETERS.json`` (the separate audit); the
``with_``/``sweep``/``at_bound`` helpers make that reconciliation a one-line sensitivity sweep.

Biology is calibrated ONCE with the reference indicator (nuclear GCaMP6s, the Randi TWISP construct): target calcium
peak so the target reaches ``target_response_dff``; optical kappa (H1) or coupling gain (H2-H7) so the neighbour
reaches ``neighbour_response_dff``. Other indicators then READ the same biology. Rivals are therefore amplitude-matched
in the existing data and can only be separated by what the contrast measures.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import lfilter
from scipy.stats import norm
from scipy.stats import t as student_t

from .measurement import ASSUMED, INFERRED, KNOWN, UNKNOWN
from .spec import SpecError, canonical_sha

RESOLUTION_MODEL_VERSION = "1.0"
DT = 5e-4                       # fine simulation grid (s)
GRID_PRE_S, GRID_POST_S = 3.0, 7.5
AUDIT = "docs/mechanism_engine/deep_campaign/BIOPHYSICAL_PARAMETERS.json"
REPORT = "docs/mechanism_engine/FINAL_REPORT.md"
LIT = "literature recall, NOT audited; reconcile with " + AUDIT
HYPOTHESES = ("H1", "H2", "H3", "H4", "H5", "H7")
READOUTS = ("both_fluorescent", "sparse_neighbour_nonexpressing", "two_colour_target_red")
Z80, Z90, Z975 = norm.ppf(0.80), norm.ppf(0.90), norm.ppf(0.975)


def _meta(status, source, lo=None, hi=None, note=""):
    return {"status": status, "source": source, "range": [lo, hi], "note": note}


class _Sweepable:
    """Helpers shared by the frozen parameter dataclasses."""

    def with_(self, **kw):
        return replace(self, **kw)

    def sweep(self, name: str, values) -> list:
        return [replace(self, **{name: v}) for v in values]

    def at_bound(self, name: str, which: str):
        lo, hi = self.provenance[name]["range"]
        v = lo if which == "low" else hi
        if v is None:
            raise SpecError(f"{name}: no {which} bound recorded")
        return replace(self, **{name: v})

    def content(self) -> dict:
        return asdict(self)


# =====================================================================================================================
# indicator
# =====================================================================================================================
@dataclass(frozen=True)
class IndicatorModel(_Sweepable):
    """Phenomenological indicator: Hill occupancy, asymmetric first-order binding kinetics, optional nuclear low-pass.

    calcium: u = [Ca] (nM); occupancy target = (H(u) - H(rest)) / (1 - H(rest)), H = u^n / (u^n + Kd^n);
             bound fraction b follows the target with tau_rise when rising, tau_decay when falling;
             nuclear signal = first-order low-pass of b (tau_nuclear, 0 = cytosolic); dF/F = dff_max * signal.
    voltage: occupancy target = normalised depolarisation (linear), same asymmetric kinetics, no Hill, no nuclear term.
    ``baseline_photon_rate_hz`` is photons/s collected from one resting ROI at illumination gain 1.
    """
    name: str
    kind: str
    tau_rise_s: float
    tau_decay_s: float
    hill_n: float
    kd_nM: float
    tau_nuclear_s: float
    baseline_photon_rate_hz: float
    dff_max: float
    provenance: dict = field(default_factory=dict, compare=False, hash=False)

    def validate(self) -> None:
        if self.kind not in ("calcium", "voltage"):
            raise SpecError(f"indicator {self.name}: kind must be calcium|voltage")
        for k in ("tau_rise_s", "tau_decay_s", "baseline_photon_rate_hz", "dff_max"):
            if not getattr(self, k) > 0:
                raise SpecError(f"indicator {self.name}: {k} must be > 0")
        if self.tau_nuclear_s < 0:
            raise SpecError(f"indicator {self.name}: tau_nuclear_s must be >= 0")
        if self.kind == "calcium" and not (self.kd_nM > 0 and self.hill_n > 0):
            raise SpecError(f"indicator {self.name}: calcium indicator needs kd_nM > 0 and hill_n > 0")
        for k, m in self.provenance.items():
            if m["status"] not in (KNOWN, INFERRED, ASSUMED, UNKNOWN) or (m["status"] != UNKNOWN and not m["source"]):
                raise SpecError(f"indicator {self.name}: provenance for {k} invalid")


def _indicator(name, kind, source_note, **vals):
    prov = {k: _meta(ASSUMED, LIT, v[1], v[2], source_note) for k, v in vals.items()}
    return IndicatorModel(name=name, kind=kind, **{k: v[0] for k, v in vals.items()}, provenance=prov)


def indicator_preset(name: str) -> IndicatorModel:
    """Named presets. Every value is ASSUMED with a (low, high) literature/plausibility range."""
    P = {
        # (value, low, high). In-vivo effective dff_max, not the in-vitro Fmax/Fmin ratio.
        "GCaMP6s": dict(tau_rise_s=(0.20, 0.10, 0.45), tau_decay_s=(1.2, 0.8, 2.5), hill_n=(2.9, 2.3, 3.2), kd_nM=(144.0, 120.0, 180.0),
                        tau_nuclear_s=(0.0, 0.0, 0.0), baseline_photon_rate_hz=(5e4, 1e4, 2.5e5), dff_max=(1.5, 0.5, 5.0)),
        "GCaMP6f": dict(tau_rise_s=(0.045, 0.02, 0.10), tau_decay_s=(0.35, 0.15, 0.6), hill_n=(2.3, 2.0, 2.6), kd_nM=(375.0, 290.0, 420.0),
                        tau_nuclear_s=(0.0, 0.0, 0.0), baseline_photon_rate_hz=(4e4, 1e4, 2e5), dff_max=(1.2, 0.4, 4.0)),
        "jGCaMP7f": dict(tau_rise_s=(0.025, 0.012, 0.06), tau_decay_s=(0.27, 0.15, 0.45), hill_n=(2.3, 1.9, 2.6), kd_nM=(174.0, 150.0, 210.0),
                         tau_nuclear_s=(0.0, 0.0, 0.0), baseline_photon_rate_hz=(5e4, 1e4, 2.5e5), dff_max=(1.5, 0.5, 5.0)),
        "jGCaMP8f": dict(tau_rise_s=(0.006, 0.003, 0.015), tau_decay_s=(0.07, 0.04, 0.12), hill_n=(1.9, 1.6, 2.2), kd_nM=(334.0, 280.0, 400.0),
                         tau_nuclear_s=(0.0, 0.0, 0.0), baseline_photon_rate_hz=(4e4, 1e4, 2e5), dff_max=(1.5, 0.5, 5.0)),
        "nuclear-GCaMP6s": dict(tau_rise_s=(0.20, 0.10, 0.45), tau_decay_s=(1.2, 0.8, 2.5), hill_n=(2.9, 2.3, 3.2), kd_nM=(144.0, 120.0, 180.0),
                                tau_nuclear_s=(0.5, 0.1, 3.0), baseline_photon_rate_hz=(5e4, 1e4, 2.5e5), dff_max=(1.5, 0.5, 5.0)),
        "VOLTAGE": dict(tau_rise_s=(0.002, 0.0005, 0.005), tau_decay_s=(0.004, 0.001, 0.02), hill_n=(1.0, 1.0, 1.0), kd_nM=(0.0, 0.0, 0.0),
                        tau_nuclear_s=(0.0, 0.0, 0.0), baseline_photon_rate_hz=(1e4, 2e3, 5e4), dff_max=(0.3, 0.05, 0.6)),
        "jRGECO1a": dict(tau_rise_s=(0.08, 0.03, 0.15), tau_decay_s=(0.6, 0.3, 1.0), hill_n=(1.85, 1.6, 2.1), kd_nM=(148.0, 130.0, 200.0),
                         tau_nuclear_s=(0.0, 0.0, 0.0), baseline_photon_rate_hz=(2e4, 5e3, 1e5), dff_max=(1.0, 0.3, 3.0)),
    }
    notes = {
        "nuclear-GCaMP6s": "Randi TWISP construct is rab-3::his-24::GCaMP6s (audit entry randi2023_calcium_indicator_construct, HIGH); "
                           "kinetics of the histone fusion were NOT reported; tau_nuclear is an assumed nuclear-equilibration low-pass",
        "VOLTAGE": "generic GEVI class (ASAP/Voltron-like), full-scale target depolarisation -> dff_max; no worm-neuron validation assumed",
        "jRGECO1a": "red calcium indicator for the two-colour design",
    }
    if name not in P:
        raise SpecError(f"unknown indicator preset {name!r}; known: {sorted(P)}")
    kind = "voltage" if name == "VOLTAGE" else "calcium"
    ind = _indicator(name, kind, notes.get(name, ""), **P[name])
    if name == "nuclear-GCaMP6s":
        prov = dict(ind.provenance)
        prov["baseline_photon_rate_hz"] = _meta(ASSUMED, REPORT + " (raw green ~58 counts above offset per voxel) x assumed 25-px box, "
                                                "~0.5 e-/count, 20 ms frames", 1e4, 2.5e5)
        ind = replace(ind, provenance=prov)
    return ind


INDICATOR_PRESETS = ("GCaMP6s", "GCaMP6f", "jGCaMP7f", "jGCaMP8f", "nuclear-GCaMP6s", "VOLTAGE")


# =====================================================================================================================
# acquisition
# =====================================================================================================================
@dataclass(frozen=True)
class AcquisitionModel(_Sweepable):
    """Scan pattern, per-plane timing, stimulus-clock relation and the photon budget.

    Frame j starts at j*frame_s and images plane(j). ``alternating`` sweeps reverse direction every volume (the Randi
    ETL triangle), so a plane's sampling interval alternates; ``sawtooth`` always sweeps in one direction.
    Sample timestamps mark exposure START; the sample is the mean signal over the exposure.
    ``stim_report_jitter_*``: with probability p the true stimulus lags the reported time by U[0, jitter] (Randi SI 0.2.2).
    ``timing_error_s``: systematic error of neighbour timestamps relative to target ones (e.g. rolling shutter).
    """
    name: str
    n_planes: int
    frame_s: float
    exposure_s: float
    sweep: str = "alternating"
    neighbour_plane_separation: int = 1
    target_plane: int | None = None          # None -> random plane per event (pooled over pairs at all depths)
    stim_phase: str = "uniform"              # "uniform" (asynchronous to scan) | "locked" (to volume start)
    stim_lock_offset_s: float = 0.0
    stim_report_jitter_s: float = 0.0
    stim_report_jitter_prob: float = 0.0
    timing_error_s: float = 0.0
    illumination_gain: float = 1.0
    read_noise_e_per_px: float = 1.6
    roi_pixels: int = 25
    background_photon_rate_hz: float = 1e4
    provenance: dict = field(default_factory=dict, compare=False, hash=False)

    def validate(self) -> None:
        if self.n_planes < 1 or self.frame_s <= 0 or not (0 < self.exposure_s <= self.frame_s + 1e-12):
            raise SpecError(f"acquisition {self.name}: need n_planes >= 1, frame_s > 0, 0 < exposure_s <= frame_s")
        if self.sweep not in ("alternating", "sawtooth"):
            raise SpecError(f"acquisition {self.name}: sweep must be alternating|sawtooth")
        if self.stim_phase not in ("uniform", "locked"):
            raise SpecError(f"acquisition {self.name}: stim_phase must be uniform|locked")
        if not 0 <= self.neighbour_plane_separation < self.n_planes:
            raise SpecError(f"acquisition {self.name}: neighbour_plane_separation must be in [0, n_planes)")
        if self.target_plane is not None and not 0 <= self.target_plane + self.neighbour_plane_separation < self.n_planes:
            raise SpecError(f"acquisition {self.name}: target_plane + separation outside the stack")
        if not 0 <= self.stim_report_jitter_prob <= 1 or self.stim_report_jitter_s < 0:
            raise SpecError(f"acquisition {self.name}: bad stimulus report jitter")
        if self.illumination_gain <= 0 or self.roi_pixels < 1 or self.read_noise_e_per_px < 0 or self.background_photon_rate_hz < 0:
            raise SpecError(f"acquisition {self.name}: bad photon budget")

    @property
    def volume_s(self) -> float:
        return self.n_planes * self.frame_s

    def plane_offsets(self) -> tuple[float, np.ndarray]:
        """(period, offsets[plane, occurrence]) of exposure starts within one scan period."""
        P, f = self.n_planes, self.frame_s
        if P == 1:
            return f, np.zeros((1, 1))
        if self.sweep == "alternating":
            occ = [[] for _ in range(P)]
            for j in range(2 * P):
                pos, vol = j % P, j // P
                occ[pos if vol % 2 == 0 else P - 1 - pos].append(j * f)
            return 2 * P * f, np.asarray(occ)
        return P * f, (np.arange(P) * f)[:, None]

    def sample_times(self, n: int, rng: np.random.Generator, pre_s: float, post_s: float, target_plane: int | None = None) -> dict:
        """Per-event sample times relative to the TRUE stimulus onset, for target and neighbour planes."""
        period, offs = self.plane_offsets()
        P, dp = self.n_planes, self.neighbour_plane_separation
        tp = self.target_plane if target_plane is None else target_plane
        pT = rng.integers(0, P - dp, size=n) if (tp is None and P > 1) else np.full(n, 0 if tp is None else tp)
        pN = pT + dp
        if self.stim_phase == "uniform":
            phi = rng.uniform(0.0, period, n)
        else:
            parity = rng.integers(0, 2, n) if (self.sweep == "alternating" and P > 1) else np.zeros(n)
            phi = (self.stim_lock_offset_s + parity * self.volume_s) % period
        reps = np.arange(-math.ceil((pre_s + period) / period), math.ceil((post_s + period) / period) + 1) * period

        def build(p):
            t = (offs[p][:, :, None] + reps[None, None, :]).reshape(n, -1) - phi[:, None]
            t.sort(axis=1)
            return t, (t >= -pre_s) & (t < post_s)

        tT, mT = build(pT)
        tN, mN = build(pN)
        return dict(tT=tT, mT=mT, tN=tN, mN=mN, target_plane=pT, phase=phi)


def acquisition_preset(name: str) -> AcquisitionModel:
    K = lambda src, lo=None, hi=None, note="": _meta(KNOWN, src, lo, hi, note)          # noqa: E731
    A = lambda src, lo=None, hi=None, note="": _meta(ASSUMED, src, lo, hi, note)        # noqa: E731
    common = dict(read_noise_e_per_px=A("sCMOS class (Hamamatsu ORCA) typical", 1.0, 2.5),
                  roi_pixels=A("5x5 px per plane (Leifer 1x5x5 / 3x5x5 extraction box)", 9, 75),
                  background_photon_rate_hz=A("tissue/out-of-focus background ~20% of a nucleus", 2e3, 5e4),
                  illumination_gain=A("1 = Randi 1.4 mW/mm^2 at 505 nm; Leifer spontaneous recordings used ~10 mW/mm^2 (audit "
                                      "randi2023_imaging_intensity) -> up to ~7x has precedent (GUR-3 light-sensitivity risk)", 1.0, 7.0))
    if name == "randi2023":
        prov = dict(common, n_planes=K("MEASUREMENT_MODEL.md; audit randi2023_planes_per_volume (MEDIUM)", 21, 25),
                    frame_s=K("per-frame NWB timestamps 0.020-0.021 s (audit randi2023_frame_interval)"),
                    exposure_s=A("exposure assumed = frame interval; true exposure not in metadata", 0.005, 0.02),
                    sweep=K("MEASUREMENT_MODEL.md: alternating sweeps"),
                    neighbour_plane_separation=A("near neighbours within ~4 um -> 0-3 planes at 1.22 um/plane", 0, 3),
                    stim_phase=A("30-s ISI random-target protocol; stimulus assumed asynchronous to scan (checkable from NWB metadata)"),
                    stim_report_jitter_s=K("Randi SI 0.2.2 (audit randi2023_stim_timing_jitter): 'occasionally up to one volume'", 0.0, 0.5),
                    stim_report_jitter_prob=A("'occasionally' -> 0.25", 0.0, 1.0),
                    timing_error_s=A("per-frame timestamps; exposure-start vs mid convention unknown -> systematic up to +-frame/2", -0.01, 0.01))
        return AcquisitionModel("randi2023", 25, 0.020, 0.020, "alternating", 1, None, "uniform", 0.0, 0.5, 0.25, 0.0, 1.0, 1.6, 25, 1e4, prov)
    if name == "randi2023_locked":
        return replace(acquisition_preset("randi2023"), name="randi2023_locked", stim_phase="locked")
    if name == "randi2023_nojitter":
        return replace(acquisition_preset("randi2023"), name="randi2023_nojitter", stim_report_jitter_prob=0.0)
    fast = dict(common, frame_s=A("proposed protocol"), exposure_s=A("exposure = frame"),
                stim_report_jitter_s=A("new experiment logs stimulus on the camera clock", 0.0, 0.001),
                timing_error_s=A("rolling shutter / trigger skew, bounded by frame/2", 0.0, 0.0025))
    if name == "two_plane_100hz":
        return AcquisitionModel(name, 2, 0.005, 0.005, "sawtooth", 1, 0, "uniform", provenance=dict(fast, n_planes=A("target + neighbour planes; ETL settling must be < 5 ms")))
    if name == "two_plane_200hz":
        return AcquisitionModel(name, 2, 0.0025, 0.0025, "sawtooth", 1, 0, "uniform", provenance=dict(fast, n_planes=A("ETL settling must be < 2.5 ms")))
    if name == "single_plane_200hz":
        return AcquisitionModel(name, 1, 0.005, 0.005, "sawtooth", 0, 0, "uniform", provenance=dict(fast, n_planes=A("same-plane pairs only; no ETL motion")))
    if name == "single_plane_1khz":
        return AcquisitionModel(name, 1, 0.001, 0.001, "sawtooth", 0, 0, "uniform", provenance=dict(fast, n_planes=A("same-plane pairs; camera ROI crop")))
    raise SpecError(f"unknown acquisition preset {name!r}")


ACQUISITION_PRESETS = ("randi2023", "randi2023_nojitter", "randi2023_locked", "two_plane_100hz", "two_plane_200hz",
                       "single_plane_200hz", "single_plane_1khz")


# =====================================================================================================================
# biology, genotype, analysis
# =====================================================================================================================
@dataclass(frozen=True)
class StimulusModel(_Sweepable):
    kind: str = "step"                       # "step" | "pulse" (pulse = short step)
    duration_s: float = 0.5
    activation_tau_s: float = 0.02           # target drive low-pass (GUR-3/PRDX-2 activation kinetics; common-mode for H1/H3)
    activation_jitter_s: float = 0.01        # per-event target activation latency SD (common to target-following mechanisms)
    provenance: dict = field(default_factory=lambda: {
        "duration_s": _meta(KNOWN, "Randi 2023 WT 0.5 s (audit randi2023_stim_duration_WT)", 0.1, 0.5),
        "activation_tau_s": _meta(ASSUMED, "GUR-3/PRDX-2 activation kinetics not characterised", 0.005, 0.5),
        "activation_jitter_s": _meta(ASSUMED, "not characterised", 0.0, 0.1)}, compare=False, hash=False)


@dataclass(frozen=True)
class BiologyModel(_Sweepable):
    ca_rest_nM: float = 50.0
    ca_tau_s: float = 0.1
    target_response_dff: float = 0.5         # reference-indicator target peak dF/F used to calibrate calcium drive
    neighbour_response_dff: float = 0.15     # reference-indicator responding-neighbour peak dF/F (rivals matched to it)
    brightness_ratio: float = 1.0            # neighbour / target resting brightness
    coupling_latency_s: float = 0.02         # H3 voltage-level latency
    coupling_tau_s: float = 0.01             # H3 coupling low-pass (neighbour membrane)
    coupling_threshold: float = 0.0          # H3 neighbour calcium-entry threshold (fraction of full target depolarisation)
    coupling_latency_jitter_s: float = 0.0
    h2_latency_s: float = 0.0
    chemical_latency_s: float = 0.3
    chemical_tau_s: float = 0.2
    polysynaptic_latency_s: float = 0.8
    polysynaptic_tau_s: float = 0.5
    global_latency_range_s: tuple = (0.0, 3.0)
    amplitude_cv: float = 0.3                # per-event response amplitude CV (lognormal)
    kinetic_cv: float = 0.0                  # per-pair neighbour indicator kinetic heterogeneity (H2-H7 only)
    bio_noise_sd: float = 0.04               # slow biological dF/F fluctuation (OU)
    bio_noise_tau_s: float = 2.0
    provenance: dict = field(default_factory=lambda: {
        "ca_rest_nM": _meta(ASSUMED, LIT, 30, 100), "ca_tau_s": _meta(ASSUMED, LIT, 0.02, 1.0),
        "target_response_dff": _meta(INFERRED, AUDIT + " randi2023_EDFig2ij (target column median dF/F0 0.395, DERIVED)", 0.2, 1.5),
        "neighbour_response_dff": _meta(INFERRED, AUDIT + " randi2023_EDFig2ij (4-10 um: 21.8% > 0.2, median 0.058)", 0.05, 0.4),
        "brightness_ratio": _meta(ASSUMED, "pan-neuronal expression", 0.3, 3.0),
        "coupling_latency_s": _meta(ASSUMED, "assumed range: electrical coupling 1-100 ms", 0.0, 0.1),
        "coupling_tau_s": _meta(ASSUMED, "neuron membrane time constant class", 0.002, 0.05),
        "chemical_latency_s": _meta(ASSUMED, "assumed range: chemical transmission 100 ms - 2 s", 0.1, 2.0),
        "amplitude_cv": _meta(ASSUMED, "event-to-event response variability", 0.1, 0.8),
        "bio_noise_sd": _meta(ASSUMED, "pre-stimulus green dF/F fluctuation not yet measured in-repo", 0.01, 0.1),
        "bio_noise_tau_s": _meta(ASSUMED, "slow calcium fluctuations", 0.5, 10.0)}, compare=False, hash=False)


@dataclass(frozen=True)
class SpatialModel(_Sweepable):
    kappa: float | None = None               # H1 optical kernel (None -> amplitude-matched)
    kappa_pair_cv: float = 0.5               # pair-to-pair geometric variability of kappa
    kappa_background: float = 0.0            # optical crosstalk present under NON-H1 worlds (fraction of the H1-required kappa)
    distance_um: float = 4.0
    footprint_fwhm_um: float = 3.1
    spot_offsets_um: tuple = (0.0, 1.0, 2.0, 3.0, 4.0)
    provenance: dict = field(default_factory=lambda: {
        "footprint_fwhm_um": _meta(KNOWN, "Randi ED Fig. 2 legend (audit: 2.4-3.7 um derived per axis)", 2.4, 3.7),
        "kappa_pair_cv": _meta(ASSUMED, "geometry variability", 0.2, 1.0),
        "kappa_background": _meta(ASSUMED, "true optical crosstalk in a non-H1 world", 0.0, 0.5)}, compare=False, hash=False)


@dataclass(frozen=True)
class GenotypeContrastModel(_Sweepable):
    """Animal-level proximity-effect model for the innexin (unc-7;unc-9) contrast."""
    wt_effect: float = 0.083                 # near-far response-probability risk difference (mean of CONF 0.091, RES 0.076)
    animal_sd: float = 0.105                 # SE x sqrt(n) from the held-out bootstrap CIs
    h3_residual_fraction: float = 0.2        # coupling surviving in unc-7;unc-9 (other innexins)
    h3_share: float = 1.0                    # fraction of the WT effect that is H3 in an H3 world
    h5_reduction_fraction: float = 0.3       # polysynaptic effect removed by the double mutant
    h1_mutant_ratio: float = 1.0             # H1 predicts unchanged; <1 models reduced target activation/brightness in mutants
    mutant_sd_inflation: float = 1.25
    floor: float | None = None               # None -> midpoint of predicted H1 and H3 effects
    provenance: dict = field(default_factory=lambda: {
        "wt_effect": _meta(INFERRED, REPORT + " T0 CONFIRMATION 0.091, RESERVE 0.076"),
        "animal_sd": _meta(INFERRED, REPORT + " T0 CIs: CONF half-width 0.0325 (n=40) -> SD 0.103; RES 0.030 (n=46) -> SD 0.104", 0.08, 0.14),
        "h3_residual_fraction": _meta(ASSUMED, "25 innexin genes; unc-7/unc-9 dominant in motor circuit, not all neurons", 0.0, 0.6),
        "mutant_sd_inflation": _meta(ASSUMED, "altered baseline activity in Unc animals", 1.0, 1.6),
        "h1_mutant_ratio": _meta(ASSUMED, "H1 unchanged if target activation and brightness are matched", 0.7, 1.0)}, compare=False, hash=False)


@dataclass(frozen=True)
class AnalysisPlan(_Sweepable):
    window_pre_s: float | None = None        # None -> chosen from acquisition/indicator
    window_post_s: float | None = None
    alpha: float = 0.05
    mc_reps: int = 200
    seed: int = 20260917
    lag_search_s: tuple = (-0.5, 2.0)
    ratio_floor: float = 0.5                 # sparse / two-colour: floor on observed / H1-required ratio
    provenance: dict = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class ExperimentResolutionSpec(_Sweepable):
    spec_id: str
    acquisition: AcquisitionModel
    indicator: IndicatorModel
    reference_indicator: IndicatorModel = field(default_factory=lambda: indicator_preset("nuclear-GCaMP6s"))
    n_animals: int = 10
    n_pairs_per_animal: int = 10
    n_events_per_pair: int = 3
    n_events_pooled: int | None = None       # pooled target-up events per distance shell (None -> animals x pairs x events)
    stimulus: StimulusModel = field(default_factory=StimulusModel)
    genotype: str = "WT"
    manipulation: str = "none"
    spatial: SpatialModel = field(default_factory=SpatialModel)
    readout: str = "both_fluorescent"
    biology: BiologyModel = field(default_factory=BiologyModel)
    genotype_model: GenotypeContrastModel = field(default_factory=GenotypeContrastModel)
    analysis: AnalysisPlan = field(default_factory=AnalysisPlan)
    red_indicator: IndicatorModel = field(default_factory=lambda: indicator_preset("jRGECO1a"))
    spectral_crosstalk: float = 0.02         # red -> green bleed-through (two-colour readout)
    target_green_autofluor: float = 0.05     # target resting green (non-responsive) when it lacks GCaMP
    extra: dict = field(default_factory=dict)
    version: str = RESOLUTION_MODEL_VERSION

    def validate(self) -> None:
        if not self.spec_id.strip():
            raise SpecError("spec_id required")
        self.acquisition.validate()
        self.indicator.validate()
        self.reference_indicator.validate()
        if self.reference_indicator.kind != "calcium":
            raise SpecError(f"{self.spec_id}: reference indicator (biology calibration) must be a calcium indicator")
        if self.readout not in READOUTS:
            raise SpecError(f"{self.spec_id}: readout must be one of {READOUTS}")
        if min(self.n_animals, self.n_pairs_per_animal, self.n_events_per_pair) < 1:
            raise SpecError(f"{self.spec_id}: sample sizes must be >= 1")
        if self.n_events_pooled is not None and self.n_events_pooled < 1:
            raise SpecError(f"{self.spec_id}: n_events_pooled must be >= 1")
        b = self.biology
        if not 0 < b.neighbour_response_dff < b.target_response_dff:
            raise SpecError(f"{self.spec_id}: need 0 < neighbour_response_dff < target_response_dff")
        if b.coupling_latency_s < 0 or b.h2_latency_s < 0 or b.chemical_latency_s < 0 or not 0 <= b.coupling_threshold < 1:
            raise SpecError(f"{self.spec_id}: latencies must be >= 0 and 0 <= coupling_threshold < 1")
        if self.stimulus.duration_s <= 0 or self.stimulus.kind not in ("step", "pulse"):
            raise SpecError(f"{self.spec_id}: bad stimulus")
        if self.spatial.kappa is not None and self.spatial.kappa <= 0:
            raise SpecError(f"{self.spec_id}: kappa must be > 0 when given")
        if not 0 < self.analysis.alpha < 0.5 or self.analysis.mc_reps < 10:
            raise SpecError(f"{self.spec_id}: alpha in (0, 0.5) and mc_reps >= 10 required")
        if self.genotype_model.animal_sd <= 0:
            raise SpecError(f"{self.spec_id}: animal_sd must be > 0")

    @property
    def n_events(self) -> int:
        return self.n_events_pooled or self.n_animals * self.n_pairs_per_animal * self.n_events_per_pair

    def content(self) -> dict:
        d = asdict(self)
        return d

    @property
    def spec_hash(self) -> str:
        return canonical_sha(self.content())

    def windows(self) -> tuple[float, float]:
        ind = self.indicator
        slow = 3.0 * (ind.tau_rise_s + ind.tau_nuclear_s) + self.stimulus.duration_s
        if self.acquisition.frame_s >= 0.01:
            pre, post = 2.0, 4.0
        else:
            pre, post = 0.3, min(4.0, max(0.6, slow + 0.1))
        return (self.analysis.window_pre_s or pre, self.analysis.window_post_s or post)

    def assumptions(self) -> list[str]:
        out = []
        for part in ("acquisition", "indicator", "reference_indicator", "stimulus", "biology", "spatial", "genotype_model"):
            for k, m in getattr(self, part).provenance.items():
                if m["status"] in (ASSUMED, UNKNOWN):
                    out.append(f"{part}.{k}")
        return sorted(out)


# =====================================================================================================================
# forward model
# =====================================================================================================================
def _lowpass(x, tau, dt=DT):
    if tau <= 0:
        return np.array(x, float)
    a = 1.0 - math.exp(-dt / tau)
    return lfilter([a], [1.0, -(1.0 - a)], x, axis=-1)


def _asym(x, tau_r, tau_d, dt=DT):
    """First-order tracking with tau_r while rising and tau_d while falling (scalar loop per row: fast for few rows)."""
    ar, ad = 1.0 - math.exp(-dt / tau_r), 1.0 - math.exp(-dt / tau_d)
    x = np.atleast_2d(np.asarray(x, float))
    y = np.empty_like(x)
    for r in range(x.shape[0]):
        b, out = 0.0, []
        app = out.append
        for v in x[r].tolist():
            d = v - b
            b += (ar if d > 0 else ad) * d
            app(b)
        y[r] = out
    return y


def _shift(x, lag_s, dt=DT):
    s = int(round(lag_s / dt))
    if s <= 0:
        return np.array(x, float)
    y = np.zeros_like(x)
    y[..., s:] = x[..., :-s]
    return y


def time_grid() -> np.ndarray:
    return np.arange(-GRID_PRE_S, GRID_POST_S, DT)


def indicator_dff(drive: np.ndarray, ind: IndicatorModel, bio: BiologyModel, ca_peak_nM: float) -> np.ndarray:
    """Noise-free dF/F of one indicator for a (K, T) normalised drive (1 = full target depolarisation)."""
    drive = np.clip(np.atleast_2d(drive), 0.0, None)
    if ind.kind == "voltage":
        return ind.dff_max * _asym(drive, ind.tau_rise_s, ind.tau_decay_s)
    u = bio.ca_rest_nM + (ca_peak_nM - bio.ca_rest_nM) * _lowpass(drive, bio.ca_tau_s)
    hill = lambda c: c ** ind.hill_n / (c ** ind.hill_n + ind.kd_nM ** ind.hill_n)   # noqa: E731
    h0 = hill(bio.ca_rest_nM)
    occ = (hill(u) - h0) / (1.0 - h0)
    x = _lowpass(_asym(occ, ind.tau_rise_s, ind.tau_decay_s), ind.tau_nuclear_s)
    return ind.dff_max * x


def _stim(t, spec):
    return ((t >= 0) & (t < spec.stimulus.duration_s)).astype(float)


def _target_drive(t, spec, levels):
    return levels[:, None] * _lowpass(_stim(t, spec), spec.stimulus.activation_tau_s)[None, :]


def _neighbour_drive(t, spec, mech, vT, gain, levels):
    b = spec.biology
    if mech == "H1":
        return None
    if mech == "H3":
        v = gain * _lowpass(_shift(vT, b.coupling_latency_s), b.coupling_tau_s)
        return np.clip(v - b.coupling_threshold, 0, None) if b.coupling_threshold > 0 else v
    if mech == "H4":
        return gain * _lowpass(_shift(vT, b.chemical_latency_s), b.chemical_tau_s)
    if mech == "H5":
        return gain * _lowpass(_shift(vT, b.polysynaptic_latency_s), b.polysynaptic_tau_s)
    if mech in ("H2", "H7"):   # driven by the light (H2) or a non-locked state change (H7; latency added per event)
        base = _lowpass(_shift(_stim(t, spec), b.h2_latency_s if mech == "H2" else 0.0), spec.stimulus.activation_tau_s)
        return gain * levels[:, None] * base[None, :]
    raise SpecError(f"unknown mechanism {mech!r}; known {HYPOTHESES}")


def _peak(x, t):
    return float(x[..., t >= 0].max())


_CACHE: dict = {}


def calibrate(spec: ExperimentResolutionSpec, mech: str) -> dict:
    """Solve calcium peak (target) and kappa / coupling gain (neighbour) with the REFERENCE indicator."""
    key = ("cal", mech, canonical_sha({k: spec.content()[k] for k in ("reference_indicator", "stimulus", "biology", "spatial")}))
    if key in _CACHE:
        return _CACHE[key]
    t = np.arange(-0.05, 6.0, DT)          # calibration grid (peaks occur well inside)
    ref, b = spec.reference_indicator, spec.biology
    one = np.ones(1)
    vT = _target_drive(t, spec, one)

    def bisect(f, target, lo, hi, n=30):
        flo = f(lo)
        if flo > target:
            return lo
        for _ in range(n):
            mid = math.sqrt(lo * hi)
            if f(mid) < target:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    ca_peak = bisect(lambda c: _peak(indicator_dff(vT, ref, b, c), t), b.target_response_dff, b.ca_rest_nM * 1.0001, 1e5)
    A_T = _peak(indicator_dff(vT, ref, b, ca_peak), t)
    r = b.brightness_ratio
    kappa_req = b.neighbour_response_dff * r / (A_T - b.neighbour_response_dff)
    out = dict(ca_peak_nM=ca_peak, target_peak_ref=A_T, kappa_required=kappa_req, gain=0.0,
               kappa=(spec.spatial.kappa if spec.spatial.kappa is not None else kappa_req) if mech == "H1" else spec.spatial.kappa_background * kappa_req)
    if mech != "H1":
        f = lambda g: _peak(indicator_dff(_neighbour_drive(t, spec, mech, vT, g, one), ref, b, ca_peak), t)   # noqa: E731
        out["gain"] = bisect(f, b.neighbour_response_dff, 1e-4, 50.0)
    out["neighbour_peak_ref"] = b.neighbour_response_dff
    _CACHE[key] = out
    return out


def amplitude_levels(cv: float, k: int = 5) -> np.ndarray:
    if cv <= 0:
        return np.ones(k)
    s = math.sqrt(math.log(1 + cv * cv))
    return np.exp(-s * s / 2 + s * norm.ppf((np.arange(k) + 0.5) / k))


def _exposure_average(x, exposure_s, dt=DT):
    ne = max(1, int(round(exposure_s / dt)))
    if ne == 1:
        return x
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, axis=-1)], axis=-1)
    y = (c[..., ne:] - c[..., :-ne]) / ne
    return np.concatenate([y, np.repeat(y[..., -1:], ne - 1, axis=-1)], axis=-1)


@dataclass
class Templates:
    t0: float
    dt: float
    levels: np.ndarray
    target: np.ndarray               # (K, T) exposure-averaged target dF/F (spec.indicator)
    target_unit: np.ndarray          # (T,) level 1
    neighbour_own: np.ndarray | None  # (B, K, T) exposure-averaged neighbour own dF/F; B = kinetic bank
    neighbour_unit: np.ndarray | None
    target_red: np.ndarray | None
    cal: dict

    def at(self, arr, k, t):
        idx = np.clip(np.rint((t - self.t0) / self.dt).astype(np.int64), 0, arr.shape[-1] - 1)
        return arr[k, idx]


def build_templates(spec: ExperimentResolutionSpec, mech: str, *, kinetic_bank: int = 12) -> Templates:
    key = ("tm", mech, spec.spec_hash)
    if key in _CACHE:
        return _CACHE[key]
    cal = calibrate(spec, mech)
    t = time_grid()
    b, ind, e = spec.biology, spec.indicator, spec.acquisition.exposure_s
    lv = amplitude_levels(b.amplitude_cv)
    lv1 = np.concatenate([lv, [1.0]])
    vT = _target_drive(t, spec, lv1)
    dT = _exposure_average(indicator_dff(vT, ind, b, cal["ca_peak_nM"]), e)
    own = own_unit = red = None
    if mech != "H1":
        vN = _neighbour_drive(t, spec, mech, vT, cal["gain"], lv1)
        inds = [ind]
        if b.kinetic_cv > 0:
            rng = np.random.default_rng(spec.analysis.seed + 7)
            s = math.sqrt(math.log(1 + b.kinetic_cv ** 2))
            for _ in range(kinetic_bank - 1):     # neighbour-only perturbation; target keeps nominal kinetics
                f = np.exp(rng.normal(-s * s / 2, s, 3))
                inds.append(replace(ind, tau_rise_s=ind.tau_rise_s * f[0], tau_decay_s=ind.tau_decay_s * f[1],
                                    tau_nuclear_s=ind.tau_nuclear_s * f[2]))
        bank = np.stack([_exposure_average(indicator_dff(vN, i, b, cal["ca_peak_nM"]), e) for i in inds])
        own, own_unit = bank[:, :-1, :], bank[0, -1, :]
    if spec.readout == "two_colour_target_red":
        red = _exposure_average(indicator_dff(vT, spec.red_indicator, b, cal["ca_peak_nM"]), e)[:-1]
    tm = Templates(float(t[0]), DT, lv, dT[:-1], dT[-1], own, own_unit, red, cal)
    _CACHE[key] = tm
    return tm


# ---------------------------------------------------------------------------------------------------------------------
# event simulation
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class EventBatch:
    tT: np.ndarray      # reported times relative to reported stimulus
    mT: np.ndarray
    cT: np.ndarray      # background-subtracted counts (target ROI; red channel for two-colour)
    yT: np.ndarray      # dF/F
    tN: np.ndarray
    mN: np.ndarray
    cN: np.ndarray      # background-subtracted counts (neighbour ROI; green channel)
    yN: np.ndarray
    meta: dict


def _ou_at(times: list, n, sd, tau, lo, hi, rng, h):
    G = int(math.ceil((hi - lo) / h)) + 2
    a = math.exp(-h / tau)
    x0 = rng.normal(0, sd, n)
    w = rng.normal(0, sd * math.sqrt(1 - a * a), (n, G))
    x = lfilter([1.0], [1.0, -a], w, axis=1, zi=(a * x0)[:, None])[0]
    out = []
    for tt in times:
        u = np.clip((tt - lo) / h, 0, G - 1.000001)
        i0 = np.floor(u).astype(np.int64)
        fr = u - i0
        rows = np.arange(n)[:, None]
        out.append(x[rows, i0] * (1 - fr) + x[rows, i0 + 1] * fr)
    return out


def _dff(c, t, m):
    pre = m & (t < 0)
    base = np.where(pre, c, 0).sum(1) / np.maximum(pre.sum(1), 1)
    return c / np.where(np.abs(base) > 1e-9, base, np.nan)[:, None] - 1.0


def simulate_events(spec: ExperimentResolutionSpec, mech: str, n: int, rng: np.random.Generator, *,
                    kappa_mult: np.ndarray | None = None, target_plane: int | None = None, pre_post: tuple | None = None) -> EventBatch:
    tm = build_templates(spec, mech)
    acq, b, st = spec.acquisition, spec.biology, spec.stimulus
    pre, post = pre_post or spec.windows()
    js = np.where(rng.random(n) < acq.stim_report_jitter_prob, rng.uniform(0, acq.stim_report_jitter_s, n), 0.0)
    S = acq.sample_times(n, rng, pre + acq.stim_report_jitter_s, post, target_plane)
    tT_true, tN_true = S["tT"], S["tN"]
    tT_rep, tN_rep = tT_true + js[:, None], tN_true + js[:, None] - acq.timing_error_s
    mT = (tT_rep >= -pre) & (tT_rep < post)
    mN = (tN_rep >= -pre) & (tN_rep < post)
    K = len(tm.levels)
    kT = rng.integers(0, K, n)
    ja = rng.normal(0, st.activation_jitter_s, n)
    dT = tm.at(tm.target, kT[:, None], tT_true - ja[:, None])
    dTc = tm.at(tm.target, kT[:, None], tN_true - ja[:, None])
    if mech == "H1":
        dN = np.zeros_like(tN_true)
    else:
        if mech in ("H2", "H7"):
            kN, jN = rng.integers(0, K, n), rng.normal(0, st.activation_jitter_s, n)
            if mech == "H7":
                jN = jN + rng.uniform(*b.global_latency_range_s, n)
        else:
            kN, jN = kT, ja + rng.normal(0, b.coupling_latency_jitter_s, n)
        bank = rng.integers(0, tm.neighbour_own.shape[0], n)
        idx = np.clip(np.rint((tN_true - jN[:, None] - tm.t0) / tm.dt).astype(np.int64), 0, tm.neighbour_own.shape[-1] - 1)
        dN = tm.neighbour_own[bank[:, None], kN[:, None], idx]
    kap = tm.cal["kappa"] * (np.ones(n) if kappa_mult is None else kappa_mult)
    e, g = acq.exposure_s, acq.illumination_gain
    F0T = spec.indicator.baseline_photon_rate_hz * e * g
    F0N = 0.0 if spec.readout == "sparse_neighbour_nonexpressing" else F0T * b.brightness_ratio
    bgc = acq.background_photon_rate_hz * e
    rv = acq.read_noise_e_per_px ** 2 * acq.roi_pixels
    h = 0.005 if acq.frame_s >= 0.01 else max(DT, min(0.002, acq.frame_s / 2))
    bT, bTc, bN = (*_ou_at([tT_true, tN_true], n, b.bio_noise_sd, b.bio_noise_tau_s, -pre - acq.stim_report_jitter_s - 1, post + 1, rng, h),
                   *_ou_at([tN_true], n, b.bio_noise_sd, b.bio_noise_tau_s, -pre - acq.stim_report_jitter_s - 1, post + 1, rng, h))
    if spec.readout == "two_colour_target_red":
        F0R = spec.red_indicator.baseline_photon_rate_hz * e * g
        dR = tm.at(tm.target_red, kT[:, None], tT_true - ja[:, None])
        dRc = tm.at(tm.target_red, kT[:, None], tN_true - ja[:, None])
        lamT = F0R * (1 + dR) * (1 + bT)
        lamN = F0N * (1 + dN) * (1 + bN) + kap[:, None] * (spec.target_green_autofluor * F0T * (1 + bTc)
                                                            + spec.spectral_crosstalk * F0R * (1 + dRc) * (1 + bTc))
    else:
        lamT = F0T * (1 + dT) * (1 + bT)
        lamN = F0N * (1 + dN) * (1 + bN) + kap[:, None] * F0T * (1 + dTc) * (1 + bTc)
    cT = lamT + rng.normal(0, 1, lamT.shape) * np.sqrt(np.clip(lamT, 0, None) + bgc + rv)
    cN = lamN + rng.normal(0, 1, lamN.shape) * np.sqrt(np.clip(lamN, 0, None) + bgc + rv)
    yT = _dff(cT, tT_rep, mT)
    yN = _dff(cN, tN_rep, mN) if F0N > 0 else np.full_like(cN, np.nan)
    return EventBatch(tT_rep, mT, cT, yT, tN_rep, mN, cN, yN,
                      dict(mech=mech, kappa=tm.cal["kappa"], kappa_required=tm.cal["kappa_required"], F0T=F0T, F0N=F0N,
                           target_plane=S["target_plane"], report_jitter=js))


# ---------------------------------------------------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------------------------------------------------
def reference_shape(spec: ExperimentResolutionSpec) -> tuple[float, float, np.ndarray]:
    """Analyst's template: target dF/F (level 1) smeared by the known stimulus-report jitter distribution, peak 1."""
    tm = build_templates(spec, "H1")
    x = tm.target_unit
    acq = spec.acquisition
    if acq.stim_report_jitter_prob > 0 and acq.stim_report_jitter_s > 0:
        js = np.linspace(0, acq.stim_report_jitter_s, 21)
        x = (1 - acq.stim_report_jitter_prob) * x + acq.stim_report_jitter_prob * np.mean([_shift(x, j) for j in js], axis=0)
    return tm.t0, tm.dt, x / max(x.max(), 1e-12)


def _binned(t, y, w, lo, hi, bin_s=0.001):
    sel = w & np.isfinite(y) & (t >= lo) & (t < hi)
    bi = np.floor((t[sel] - lo) / bin_s).astype(np.int64)
    B = int(math.ceil((hi - lo) / bin_s)) + 1
    cnt = np.bincount(bi, minlength=B).astype(float)
    S = np.bincount(bi, y[sel], minlength=B)
    tsum = np.bincount(bi, t[sel], minlength=B)
    nz = cnt > 0
    return tsum[nz] / cnt[nz], cnt[nz], S[nz], float((y[sel] ** 2).sum())


def _sse(ref, tb, cnt, S, Syy, shifts):
    t0, dt, x = ref
    idx = np.clip(np.rint((tb[None, :] - shifts[:, None] - t0) / dt).astype(np.int64), 0, len(x) - 1)
    R = x[idx]
    A11, A12, A22 = (R * R * cnt).sum(1), (R * cnt).sum(1), cnt.sum()
    r1, r2 = (R * S).sum(1), S.sum()
    det = np.where(np.abs(A11 * A22 - A12 ** 2) > 1e-12, A11 * A22 - A12 ** 2, np.inf)
    a = (r1 * A22 - r2 * A12) / det
    c = (A11 * r2 - A12 * r1) / det
    return Syy - 2 * (a * r1 + c * r2) + a * a * A11 + 2 * a * c * A12 + c * c * A22, a


def fit_shift(ref, t, y, w, lo, hi, search=(-0.5, 2.0)) -> tuple[float, float, float]:
    """Pooled equivalent-time template fit: y = a * ref(t - s) + c. Returns (shift, amplitude, sse)."""
    tb, cnt, S, Syy = _binned(t, y, w, lo, hi)
    if len(tb) < 3:
        return float("nan"), float("nan"), float("nan")
    coarse = np.arange(search[0], search[1] + 1e-9, 0.01)
    e, _ = _sse(ref, tb, cnt, S, Syy, coarse)
    i0 = int(np.nanargmin(e))
    if i0 in (0, len(coarse) - 1):          # optimum on the search boundary: shift not identifiable in this window
        return float("nan"), float("nan"), float("nan")
    s0 = coarse[i0]
    fine = np.arange(s0 - 0.012, s0 + 0.0125, 0.001)
    e, a = _sse(ref, tb, cnt, S, Syy, fine)
    i = int(np.nanargmin(e))
    s = fine[i]
    if 0 < i < len(fine) - 1:
        den = e[i - 1] - 2 * e[i] + e[i + 1]
        if den > 0:
            s += 0.001 * 0.5 * (e[i - 1] - e[i + 1]) / den
    return float(s), float(a[i]), float(e[i])


def lag_estimate(spec, batch: EventBatch, ref=None) -> dict:
    ref = ref or reference_shape(spec)
    pre, post = spec.windows()
    lo = -min(pre, 1.0)
    sT, aT, _ = fit_shift(ref, batch.tT, batch.yT, batch.mT, lo, post, spec.analysis.lag_search_s)
    sN, aN, _ = fit_shift(ref, batch.tN, batch.yN, batch.mN, lo, post, spec.analysis.lag_search_s)
    return dict(t0_target=sT, t0_neighbour=sN, delta=sN - sT, amp_target=aT, amp_neighbour=aN)


def expected_lag(spec: ExperimentResolutionSpec, mech: str) -> float:
    """Noise-free apparent lag of the mechanism under the pooled template estimator (dense, exact sampling)."""
    tm = build_templates(spec, mech)
    t = np.arange(-1.0, spec.windows()[1], 0.001)
    ref = reference_shape(spec)
    x = ref[2] * tm.target_unit.max()
    idx = np.clip(np.rint((t - tm.t0) / tm.dt).astype(np.int64), 0, len(x) - 1)
    if mech == "H1":
        return 0.0
    own = tm.neighbour_unit
    if mech == "H7":
        lo, hi = spec.biology.global_latency_range_s
        own = np.mean([_shift(own, j) for j in np.linspace(lo, hi, 31)], axis=0)
    acq = spec.acquisition
    if acq.stim_report_jitter_prob > 0 and acq.stim_report_jitter_s > 0:
        js = np.linspace(0, acq.stim_report_jitter_s, 21)
        own = (1 - acq.stim_report_jitter_prob) * own + acq.stim_report_jitter_prob * np.mean([_shift(own, j) for j in js], axis=0)
    w = np.ones((1, len(t)), bool)
    sT = fit_shift(ref, t[None, :], x[idx][None, :], w, -1.0, t[-1] + 1e-3, spec.analysis.lag_search_s)[0]
    sN = fit_shift(ref, t[None, :], own[idx][None, :], w, -1.0, t[-1] + 1e-3, spec.analysis.lag_search_s)[0]
    return sN - sT


# ---------------------------------------------------------------------------------------------------------------------
# Cramer-Rao bound for the neighbour-minus-target lag (joint Gaussian, H1 null shape = pure lag)
# ---------------------------------------------------------------------------------------------------------------------
def lag_crb(spec: ExperimentResolutionSpec, *, n_draws: int | None = None, target_plane: int | None = None, seed: int = 0) -> dict:
    """Per-event CRB on delta = t0_N - t0_T for a PURE LAG (neighbour shape = scaled target shape).

    Joint model per event over stacked [target samples; neighbour samples]:
      mean  a_T m(t - t0_T) + b_T,  a_N s m(t - t0_N) + b_N,  with m = jitter-averaged target template
      cov   shot+read (diagonal) + OU biological (neighbour carries kappa-weighted copy of target OU under H1)
            + amplitude variability and stimulus-report jitter as exact second moments over a jitter quadrature.
    Returns se_per_event_s; SE for N pooled events = se_per_event_s / sqrt(N) (FIM additivity over iid events).
    """
    acq, b, ind = spec.acquisition, spec.biology, spec.indicator
    tm = build_templates(spec, "H1")
    pre, post = spec.windows()
    kap, r = tm.cal["kappa"], b.brightness_ratio
    s = kap / (r + kap)
    x = tm.target_unit
    dx = np.gradient(x, tm.dt)
    if acq.stim_report_jitter_prob > 0 and acq.stim_report_jitter_s > 0:
        js = np.concatenate([[0.0], np.linspace(0, acq.stim_report_jitter_s, 11)])
        wj = np.concatenate([[1 - acq.stim_report_jitter_prob], np.full(11, acq.stim_report_jitter_prob / 11)])
    else:
        js, wj = np.zeros(1), np.ones(1)
    e, g = acq.exposure_s, acq.illumination_gain
    lamT0 = ind.baseline_photon_rate_hz * e * g
    lamN0 = lamT0 * (r + kap)
    bgc, rv = acq.background_photon_rate_hz * e, acq.read_noise_e_per_px ** 2 * acq.roi_pixels
    rng = np.random.default_rng(seed)
    n_draws = n_draws or (64 if acq.frame_s >= 0.01 else 4)
    J = np.zeros((6, 6))
    cv2 = b.amplitude_cv ** 2
    for _ in range(n_draws):
        S = acq.sample_times(1, rng, pre, post, target_plane)
        tT, tN = S["tT"][0][S["mT"][0]], S["tN"][0][S["mN"][0]]
        tt = np.concatenate([tT, tN])
        nT = len(tT)
        sc = np.concatenate([np.ones(nT), np.full(len(tN), s)])
        idx = lambda tv: np.clip(np.rint((tv - tm.t0) / tm.dt).astype(np.int64), 0, len(x) - 1)   # noqa: E731
        Rj = np.stack([x[idx(tt - j)] for j in js]) * sc          # (Jq, M)
        dRj = np.stack([dx[idx(tt - j)] for j in js]) * sc
        m = wj @ Rj
        dm = wj @ dRj
        M2 = (Rj * wj[:, None]).T @ Rj
        cov = (1 + cv2) * M2 - np.outer(m, m)
        lam0 = np.concatenate([np.full(nT, lamT0), np.full(len(tN), lamN0)])
        white = (lam0 * (1 + m) + bgc + rv) / lam0 ** 2
        D = np.abs(tt[:, None] - tt[None, :])
        K = b.bio_noise_sd ** 2 * np.exp(-D / b.bio_noise_tau_s)
        wbio = np.ones((len(tt), len(tt)))
        wbio[:nT, nT:] = s
        wbio[nT:, :nT] = s
        wbio[nT:, nT:] = (1 - s) ** 2 + s ** 2
        Sig = np.diag(white) + K * wbio + cov
        Dm = np.zeros((len(tt), 6))
        Dm[:nT, 0], Dm[:nT, 1], Dm[:nT, 2] = -dm[:nT], m[:nT], 1.0
        Dm[nT:, 3], Dm[nT:, 4], Dm[nT:, 5] = -dm[nT:], m[nT:], 1.0
        cf = cho_factor(Sig + 1e-12 * np.eye(len(tt)))
        J += Dm.T @ cho_solve(cf, Dm)
    J /= n_draws
    c = np.array([-1, 0, 0, 1, 0, 0.0])
    try:
        var = float(c @ np.linalg.solve(J, c))
    except np.linalg.LinAlgError:
        var = float("inf")
    return dict(se_per_event_s=math.sqrt(max(var, 0.0)), neighbour_dff_amp=s * float(x.max()), target_dff_amp=float(x.max()),
                kappa=kap, n_draws=n_draws, window=[pre, post])


def lag_power_from_se(se: float, delta: float, alpha: float = 0.05) -> dict:
    """Normal-approximation detection and engine-style separation power for a lag difference ``delta``."""
    z = norm.ppf(1 - alpha / 2)
    if se <= 0 or not np.isfinite(se):
        return dict(detection=float("nan"), separation=float("nan"))
    d = abs(delta) / se
    det = float(norm.cdf(d - z) + norm.cdf(-d - z))
    f = d / 2
    p_null = max(0.0, 2 * norm.cdf(f - z) - 1)
    p_pos = float(norm.cdf(f - z))
    return dict(detection=det, separation=float(min(p_null, p_pos)), p_null_correct=p_null, p_pos_correct=p_pos)


def mde_lag(se: float, alpha: float = 0.05, power: float = 0.8) -> dict:
    z, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    # separation: floor at the midpoint; need both P(NULL|0) and P(POS|L) >= power
    k_null = 2 * (z + norm.ppf((1 + power) / 2))
    k_pos = 2 * (z + zb)
    return dict(detection_s=(z + zb) * se, separation_s=max(k_null, k_pos) * se)


# =====================================================================================================================
# contrasts + compiler
# =====================================================================================================================
CONTRASTS = {
    "lag": dict(description="pooled equivalent-time template fit of neighbour-minus-target onset shift",
                predictions={"H1": "delta = 0 (exact copy)", "H2": "delta ~ own activation latency (~0), independent amplitude",
                             "H3": "delta = coupling latency + coupling low-pass (5-100 ms)", "H4": "delta 100 ms - 2 s",
                             "H5": "delta >= 500 ms", "H7": "no target-locked delay (large smeared shift)"}),
    "shape": dict(description="likelihood-ratio of free (shift x extra low-pass) model vs exact scaled-copy model",
                  predictions={"H1": "copy model adequate", "H2": "shape departs", "H3": "shape departs", "H4": "shape departs",
                               "H5": "shape departs", "H7": "shape departs"}),
    "sparse_expression": dict(description="non-expressing neighbour ROI response in raw counts / H1-required kappa x target counts",
                              predictions={"H1": "ratio ~ 1 (POS)", "H2": "ratio ~ kappa_background (NULL)", "H3": "ratio ~ kappa_background (NULL)",
                                           "H4": "ratio ~ kappa_background (NULL)", "H5": "ratio ~ kappa_background (NULL)",
                                           "H7": "ratio ~ kappa_background (NULL)"}),
    "two_colour": dict(description="target expresses only a red indicator; neighbour GREEN response / expected biological response",
                       predictions={"H1": "green ratio ~ 0 (NULL)", "H2": "green ratio ~ 1 (POS)", "H3": "green ratio ~ 1 (POS)",
                                    "H4": "green ratio ~ 1 (POS)", "H5": "green ratio ~ 1 (POS)", "H7": "green ratio ~ 1 (POS)"}),
    "genotype": dict(description="WT minus unc-7;unc-9 near-far response-probability risk difference (animal units)",
                     predictions={"H1": "no reduction (NULL)", "H2": "no reduction (NULL)", "H3": "reduction (POS)",
                                  "H4": "no reduction (NULL)", "H5": "partial reduction", "H7": "no reduction (NULL)"}),
    "spot_offset": dict(description="slope of neighbour response on spot-neighbour proximity at fixed target activation",
                        predictions={"H1": "zero slope (NULL)", "H2": "positive slope (POS)", "H3": "zero slope (NULL)",
                                     "H4": "zero slope (NULL)", "H5": "zero slope (NULL)", "H7": "zero slope (NULL)"}),
    "kappa_calibration": dict(description="log(observed neighbour count response / (static-fluorophore kappa x target count response))",
                              predictions={"H1": "log ratio ~ 0 (NULL)", "H2": "log ratio >> 0 (POS)", "H3": "log ratio >> 0 (POS)",
                                           "H4": "log ratio >> 0 (POS)", "H5": "log ratio >> 0 (POS)", "H7": "log ratio >> 0 (POS)"}),
    "coupling_coefficient": dict(description="paired-patch steady-state coupling coefficient of near pairs",
                                 predictions={"H1": "CC ~ background (NULL)", "H2": "CC ~ background (NULL)", "H3": "CC >= required (POS)",
                                              "H4": "CC ~ background (NULL)", "H5": "CC ~ background (NULL)", "H7": "CC ~ background (NULL)"}),
}


def _seed(spec, *names) -> np.random.SeedSequence:
    h = canonical_sha([spec.analysis.seed, *names])
    return np.random.SeedSequence(int(h[:16], 16))


def _calls_power(est_a, se_a, est_b, se_b, center_a, center_b, floor, alpha):
    """Engine-style calls: A is correct if CI within [center_a +- floor]; B correct if CI beyond center_a +- floor toward B."""
    z = norm.ppf(1 - alpha / 2)
    sign = 1.0 if center_b >= center_a else -1.0
    a_ok = (est_a - z * se_a >= center_a - floor) & (est_a + z * se_a <= center_a + floor)
    if sign > 0:
        b_ok = est_b - z * se_b > center_a + floor
    else:
        b_ok = est_b + z * se_b < center_a - floor
    return float(np.mean(a_ok)), float(np.mean(b_ok))


class ResolutionCompiler:
    """compile(spec, (A, B), contrast) -> power / expected information / minimum detectable effect."""

    def compile(self, spec: ExperimentResolutionSpec, hypothesis_pair: tuple, contrast: str, *, reps: int | None = None) -> dict:
        spec.validate()
        if contrast not in CONTRASTS:
            raise SpecError(f"unknown contrast {contrast!r}; known {sorted(CONTRASTS)}")
        A, B = hypothesis_pair
        preds = CONTRASTS[contrast]["predictions"]
        for h in (A, B):
            if h not in preds:
                raise SpecError(f"contrast {contrast} has no prediction for {h}")
        if preds[A] == preds[B]:
            raise SpecError(f"contrast {contrast} is non-discriminating for {A} vs {B}: both predict {preds[A]!r}")
        reps = reps or spec.analysis.mc_reps
        fn = getattr(self, "_c_" + contrast)
        out = fn(spec, A, B, reps)
        p_sep = out.get("power_separation")
        out.update(contrast=contrast, hypothesis_pair=[A, B], predictions={A: preds[A], B: preds[B]}, spec_id=spec.spec_id,
                   spec_hash=spec.spec_hash, model_version=RESOLUTION_MODEL_VERSION, seed=spec.analysis.seed, reps=reps,
                   expected_information_bits=_eig_bits(p_sep), assumptions=spec.assumptions())
        return out

    # ---- (a) lag ---------------------------------------------------------------------------------------------------
    def _c_lag(self, spec, A, B, reps):
        n = spec.n_events
        ref = reference_shape(spec)
        dist = {}
        for h in (A, B):
            rng = np.random.default_rng(_seed(spec, "lag", h, n))
            d = []
            for _ in range(reps):
                d.append(lag_estimate(spec, simulate_events(spec, h, n, rng), ref)["delta"])
            dist[h] = np.asarray(d)
        da, db = dist[A], dist[B]
        da, db = da[np.isfinite(da)], db[np.isfinite(db)]
        se_a, se_b = float(np.std(da, ddof=1)), float(np.std(db, ddof=1))
        qlo, qhi = np.quantile(da, [spec.analysis.alpha / 2, 1 - spec.analysis.alpha / 2])
        power_det = float(np.mean((db < qlo) | (db > qhi)))
        ea, eb = expected_lag(spec, A), expected_lag(spec, B)
        floor = abs(eb - ea) / 2
        pa, pb = _calls_power(da, se_a, db, se_b, ea, eb, floor, spec.analysis.alpha)
        crb = lag_crb(spec)
        se_crb = crb["se_per_event_s"] / math.sqrt(n)
        return dict(n_events=n, expected_delta_s={A: ea, B: eb}, mc_mean_delta_s={A: float(np.mean(da)), B: float(np.mean(db))},
                    mc_se_s={A: se_a, B: se_b}, crb_se_s=se_crb, crb_per_event_se_s=crb["se_per_event_s"],
                    power_detection=power_det, power_separation=min(pa, pb), p_correct={A: pa, B: pb}, floor_s=floor,
                    mde_s=dict(mc=mde_lag(se_a, spec.analysis.alpha), crb=mde_lag(se_crb, spec.analysis.alpha)))

    # ---- (b) shape -------------------------------------------------------------------------------------------------
    def _c_shape(self, spec, A, B, reps):
        n = spec.n_events
        t0, dt, x = reference_shape(spec)
        taus = (0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
        bank = [(t0, dt, (lambda y: y / max(y.max(), 1e-12))(_lowpass(x, tau))) for tau in taus]
        pre, post = spec.windows()
        lo = -min(pre, 1.0)

        def stat(batch):
            sT = fit_shift(bank[0], batch.tT, batch.yT, batch.mT, lo, post, spec.analysis.lag_search_s)[0]
            tb, cnt, S, Syy = _binned(batch.tN, batch.yN, batch.mN, lo, post)
            sse_copy = _sse(bank[0], tb, cnt, S, Syy, np.array([sT]))[0][0]
            shifts = sT + np.arange(-0.05, 0.3001, 0.005)
            best = min(np.nanmin(_sse(bk, tb, cnt, S, Syy, shifts)[0]) for bk in bank)
            dof = max(cnt.sum() - 4, 1)
            return (sse_copy - best) / (best / dof)

        dist = {}
        for h in (A, B):
            rng = np.random.default_rng(_seed(spec, "shape", h, n))
            dist[h] = np.asarray([stat(simulate_events(spec, h, n, rng)) for _ in range(reps)])
        null_h = A if A == "H1" else B if B == "H1" else A
        alt_h = B if null_h == A else A
        crit = float(np.quantile(dist[null_h], 1 - spec.analysis.alpha))
        power = float(np.mean(dist[alt_h] > crit))
        return dict(n_events=n, null_hypothesis=null_h, critical_value=crit, power_detection=power, power_separation=power,
                    median_stat={h: float(np.median(v)) for h, v in dist.items()})

    # ---- (c) sparse expression -------------------------------------------------------------------------------------
    def _ratio_contrast(self, spec, A, B, reps, readout, pos_hyps):
        spec = replace(spec, readout=readout)
        nA, nP, nE = spec.n_animals, spec.n_pairs_per_animal, spec.n_events_per_pair
        pre, post = spec.windows()
        win = (0.25 if spec.acquisition.frame_s >= 0.01 else 0.05, min(post, 3.0))
        floor = spec.analysis.ratio_floor
        res = {}
        for h in (A, B):
            rng = np.random.default_rng(_seed(spec, readout, h, nA, nP, nE))
            tmh = build_templates(spec, h)
            est = []
            for _ in range(reps):
                n = nA * nP * nE
                pair_k = np.repeat(np.exp(rng.normal(-0.5 * math.log(1 + spec.spatial.kappa_pair_cv ** 2),
                                                     math.sqrt(math.log(1 + spec.spatial.kappa_pair_cv ** 2)), nA * nP)), nE)
                bt = simulate_events(spec, h, n, rng, kappa_mult=pair_k)

                def resp(c, t, m):
                    post_m = m & (t >= win[0]) & (t < win[1])
                    pre_m = m & (t < 0)
                    return (np.where(post_m, c, 0).sum(1) / np.maximum(post_m.sum(1), 1)
                            - np.where(pre_m, c, 0).sum(1) / np.maximum(pre_m.sum(1), 1))
                if readout == "sparse_neighbour_nonexpressing":
                    num = resp(bt.cN, bt.tN, bt.mN)
                    den = tmh.cal["kappa_required"] * resp(bt.cT, bt.tT, bt.mT)
                else:   # two-colour: neighbour green dF/F response relative to the expected biological response
                    yN = bt.cN / (spec.indicator.baseline_photon_rate_hz * spec.acquisition.exposure_s * spec.acquisition.illumination_gain
                                  * spec.biology.brightness_ratio) - 1.0
                    num = resp(yN, bt.tN, bt.mN)
                    den = np.full(n, _expected_window_response(spec, win))
                per_animal = num.reshape(nA, -1).sum(1) / den.reshape(nA, -1).sum(1)
                est.append((per_animal.mean(), per_animal.std(ddof=1) / math.sqrt(nA) if nA > 1 else np.inf))
            res[h] = np.asarray(est)
        tcrit = student_t.ppf(1 - spec.analysis.alpha / 2, max(nA - 1, 1))

        def p_correct(h):
            e, se = res[h][:, 0], res[h][:, 1]
            if h in pos_hyps:
                return float(np.mean(e - tcrit * se > floor))
            return float(np.mean((e - tcrit * se >= -floor) & (e + tcrit * se <= floor)))
        pa, pb = p_correct(A), p_correct(B)
        return dict(n_animals=nA, n_pairs_per_animal=nP, n_events_per_pair=nE, floor=floor,
                    mean_ratio={h: float(res[h][:, 0].mean()) for h in res}, mean_se={h: float(res[h][:, 1].mean()) for h in res},
                    p_correct={A: pa, B: pb}, power_separation=min(pa, pb), power_detection=max(pa, pb))

    def _c_sparse_expression(self, spec, A, B, reps):
        return self._ratio_contrast(spec, A, B, reps, "sparse_neighbour_nonexpressing", {"H1"})

    def _c_two_colour(self, spec, A, B, reps):
        return self._ratio_contrast(spec, A, B, reps, "two_colour_target_red", {"H2", "H3", "H4", "H5", "H7"})

    # ---- (e) genotype ----------------------------------------------------------------------------------------------
    def _c_genotype(self, spec, A, B, reps):
        gm = spec.genotype_model
        mu = {h: genotype_predicted_reduction(gm, h) for h in (A, B)}
        floor = gm.floor if gm.floor is not None else abs(mu[A] - mu[B]) / 2
        n = spec.n_animals
        out = dict(n_per_genotype=n, predicted_reduction=mu, floor=floor)
        out.update(genotype_power(gm, A, B, n, reps, spec.analysis.alpha, floor, _seed(spec, "genotype", A, B, n)))
        out["n_required"] = {f"{int(p * 100)}": genotype_n_required(gm, A, B, p, spec.analysis.alpha, floor) for p in (0.8, 0.9)}
        return out

    # ---- F spot offset ---------------------------------------------------------------------------------------------
    def _c_spot_offset(self, spec, A, B, reps):
        x = spec.extra
        beta = x.get("h2_effect_at_spot", 0.10)
        gamma = x.get("target_mediated_effect", 0.083)
        p0 = x.get("baseline_response_prob", 0.08)
        animal_sd = x.get("slope_animal_sd", 0.05)
        sig = spec.spatial.footprint_fwhm_um / 2.3548
        D = spec.spatial.distance_um
        d = np.asarray(spec.spatial.spot_offsets_um)
        gT = np.exp(-d ** 2 / (2 * sig ** 2))
        gN = np.exp(-(D - d) ** 2 / (2 * sig ** 2))
        X = np.column_stack([np.ones_like(d), gT, gN])
        ne = spec.n_pairs_per_animal * spec.n_events_per_pair
        rng = np.random.default_rng(_seed(spec, "spot", A, B))
        floor = x.get("floor", beta / 2)
        res = {}
        for h in (A, B):
            est = []
            for _ in range(reps):
                slopes = []
                for _a in range(spec.n_animals):
                    p = p0 + gamma * gT + (beta + rng.normal(0, animal_sd) if h == "H2" else rng.normal(0, animal_sd)) * gN
                    y = rng.binomial(ne, np.clip(p, 0, 1)) / ne
                    slopes.append(np.linalg.lstsq(X, y, rcond=None)[0][2])
                slopes = np.asarray(slopes)
                est.append((slopes.mean(), slopes.std(ddof=1) / math.sqrt(len(slopes))))
            res[h] = np.asarray(est)
        tcrit = student_t.ppf(1 - spec.analysis.alpha / 2, spec.n_animals - 1)
        pc = {}
        for h in (A, B):
            e, se = res[h][:, 0], res[h][:, 1]
            pc[h] = float(np.mean(e - tcrit * se > floor)) if h == "H2" else float(np.mean((e - tcrit * se >= -floor) & (e + tcrit * se <= floor)))
        return dict(n_animals=spec.n_animals, events_per_offset=ne, offsets_um=list(d), floor=floor, p_correct=pc,
                    power_separation=min(pc.values()), power_detection=max(pc.values()))

    # ---- G kappa calibration ---------------------------------------------------------------------------------------
    def _c_kappa_calibration(self, spec, A, B, reps):
        x = spec.extra
        rho = x.get("kappa_true_over_required_non_h1", 0.2)
        psf_bias = x.get("psf_bias", 1.0)
        mismatch = x.get("static_functional_mismatch_cv", 0.3)
        resp_cv = x.get("per_event_ratio_cv", 0.8)
        floor = x.get("floor", math.log(2.0))
        sd_pair = math.sqrt(math.log(1 + mismatch ** 2) + math.log(1 + resp_cv ** 2) / spec.n_events_per_pair)
        rng = np.random.default_rng(_seed(spec, "kappa", A, B))
        tcrit = student_t.ppf(1 - spec.analysis.alpha / 2, spec.n_animals - 1)
        pc, means = {}, {}
        for h in (A, B):
            mu = math.log(psf_bias) + (0.0 if h == "H1" else -math.log(rho))
            pair = rng.normal(mu, sd_pair, (reps, spec.n_animals, spec.n_pairs_per_animal))
            an = pair.mean(2) + rng.normal(0, 0.1, (reps, spec.n_animals))
            e, se = an.mean(1), an.std(1, ddof=1) / math.sqrt(spec.n_animals)
            means[h] = float(e.mean())
            pc[h] = float(np.mean((e - tcrit * se >= -floor) & (e + tcrit * se <= floor))) if h == "H1" else float(np.mean(e - tcrit * se > floor))
        return dict(n_animals=spec.n_animals, n_pairs_per_animal=spec.n_pairs_per_animal, floor_log=floor, mean_log_ratio=means,
                    p_correct=pc, power_separation=min(pc.values()), power_detection=max(pc.values()))

    # ---- H paired patch --------------------------------------------------------------------------------------------
    def _c_coupling_coefficient(self, spec, A, B, reps):
        x = spec.extra
        cal = calibrate(spec, "H3")
        cc_req = x.get("cc_required", min(cal["gain"], 1.0))
        cc_bg = x.get("cc_background", 0.02)
        se_cc = x.get("cc_measurement_se", 0.03)
        het = x.get("cc_pair_cv", 0.4)
        n = x.get("n_pairs", spec.n_animals)
        floor = x.get("floor", (cc_req + cc_bg) / 2 - cc_bg)
        rng = np.random.default_rng(_seed(spec, "patch", A, B, n))
        tcrit = student_t.ppf(1 - spec.analysis.alpha / 2, n - 1)
        pc = {}
        for h in (A, B):
            mu = cc_req if h == "H3" else cc_bg
            vals = mu * np.exp(rng.normal(-0.5 * math.log(1 + het ** 2), math.sqrt(math.log(1 + het ** 2)), (reps, n))) + rng.normal(0, se_cc, (reps, n))
            e, se = vals.mean(1) - cc_bg, vals.std(1, ddof=1) / math.sqrt(n)
            pc[h] = float(np.mean(e - tcrit * se > floor)) if h == "H3" else float(np.mean((e - tcrit * se >= -floor) & (e + tcrit * se <= floor)))
        return dict(n_pairs=n, cc_required=cc_req, cc_background=cc_bg, floor=floor, p_correct=pc,
                    power_separation=min(pc.values()), power_detection=max(pc.values()))


def _eig_bits(p_sep):
    """Expected information (bits) about a 50:50 binary A-vs-B question from a test that makes the correct decided call
    with probability p and is otherwise INCONCLUSIVE (never wrong): EIG = p * 1 bit."""
    if p_sep is None or not np.isfinite(p_sep):
        return None
    return float(p_sep)


def _expected_window_response(spec, win):
    """Noise-free expected mean dF/F response in the window for a biologically responding neighbour (level-averaged)."""
    tm = build_templates(spec, "H3")
    t = np.arange(win[0], win[1], 0.005)
    idx = np.clip(np.rint((t - tm.t0) / tm.dt).astype(np.int64), 0, tm.neighbour_own.shape[-1] - 1)
    return float(tm.neighbour_own[0][:, idx].mean())


# ---------------------------------------------------------------------------------------------------------------------
# genotype helpers (animal-level)
# ---------------------------------------------------------------------------------------------------------------------
def genotype_predicted_reduction(gm: GenotypeContrastModel, h: str) -> float:
    """Predicted WT-minus-mutant reduction of the near-far risk difference."""
    if h == "H3":
        return gm.wt_effect * gm.h3_share * (1 - gm.h3_residual_fraction)
    if h == "H5":
        return gm.wt_effect * gm.h5_reduction_fraction
    return gm.wt_effect * (1 - gm.h1_mutant_ratio)


def genotype_power(gm, A, B, n, reps, alpha, floor, seedseq) -> dict:
    """Animal-level MC. The hypothesis predicting the SMALLER reduction is correct when the Welch CI lies within
    [its prediction +- floor]; the other is correct when the CI lower bound exceeds that prediction + floor."""
    rng = np.random.default_rng(seedseq)
    sw, sm = gm.animal_sd, gm.animal_sd * gm.mutant_sd_inflation
    reds = {h: genotype_predicted_reduction(gm, h) for h in (A, B)}
    lo_h = min(reds, key=reds.get)
    center = reds[lo_h]
    pc, det = {}, {}
    for h in (A, B):
        red = reds[h]
        wt = rng.normal(gm.wt_effect, sw, (reps, n))
        mu = rng.normal(gm.wt_effect - red, sm, (reps, n))
        d = wt.mean(1) - mu.mean(1)
        vw, vm = wt.var(1, ddof=1) / n, mu.var(1, ddof=1) / n
        se = np.sqrt(vw + vm)
        df = (vw + vm) ** 2 / (vw ** 2 / (n - 1) + vm ** 2 / (n - 1))
        tc = student_t.ppf(1 - alpha / 2, df)
        if h == lo_h:
            pc[h] = float(np.mean((d - tc * se >= center - floor) & (d + tc * se <= center + floor)))
        else:
            pc[h] = float(np.mean(d - tc * se > center + floor))
        det[h] = float(np.mean(d - tc * se > 0))
    return dict(p_correct=pc, power_separation=min(pc.values()),
                power_detection=max(det.values()), p_reduction_detected=det)


def genotype_n_required(gm, A, B, power, alpha, floor) -> dict:
    """Normal-approximation animals per genotype for (i) detecting the H3 reduction vs zero, (ii) engine-style separation."""
    z = norm.ppf(1 - alpha / 2)
    var1 = gm.animal_sd ** 2 * (1 + gm.mutant_sd_inflation ** 2)
    ra, rb = genotype_predicted_reduction(gm, A), genotype_predicted_reduction(gm, B)
    diff = abs(ra - rb)
    n_det = math.ceil((z + norm.ppf(power)) ** 2 * var1 / diff ** 2) if diff > 0 else None
    # separation: POS side needs (diff - floor) >= (z + z_p) se; NULL side needs floor >= (z + z_{(1+p)/2}) se
    k = max((z + norm.ppf(power)) / max(diff - floor, 1e-12), (z + norm.ppf((1 + power) / 2)) / max(floor, 1e-12))
    n_sep = math.ceil(k ** 2 * var1) if diff > 0 else None
    return dict(detect_reduction=n_det, separation=n_sep)


def default_spec(spec_id="randi2023_nuclear_gcamp6s", acquisition="randi2023", indicator="nuclear-GCaMP6s", **kw) -> ExperimentResolutionSpec:
    acq = acquisition_preset(acquisition) if isinstance(acquisition, str) else acquisition
    ind = indicator_preset(indicator) if isinstance(indicator, str) else indicator
    return ExperimentResolutionSpec(spec_id=spec_id, acquisition=acq, indicator=ind, **kw)
