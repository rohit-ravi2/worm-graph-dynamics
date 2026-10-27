"""Faithful re-implementation of the Randi et al. 2023 (funatlas) significance pipeline.

Vendored/ported from Leifer-lab code (read-only clones, MIT/GPL per their LICENSE files):
  pumpprobe      commit 1dbc5e0  (pumpprobe/Funatlas.py, pumpprobe/Fconn.py)
  wormdatamodel  commit 2ab9561  (wormdatamodel/signal/signal.py)
  multipy 0.16   (multipy/fdr.py: qvalue)  [python2 xrange -> range]
Each function cites the source lines it ports. Behavioural quirks are kept on purpose
(e.g. global shift_vol for the first stim, P not updated on convergence break).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_coeffs
from scipy.interpolate import UnivariateSpline
from scipy.stats import kstest, combine_pvalues

DT = 0.5
SHIFT_VOL = 60          # Fconn.shift_vol = int(delta_t_pre/Dt), delta_t_pre = 30 s (responses_detect.py:16, Fconn.py:170)
NAN_TH_Q = 0.05         # Funatlas.nan_th (Funatlas.py:11); used by get_kolmogorov_smirnov_p (Funatlas.py:2587)


# ---------------------------------------------------------------- preprocessing (signal.py)
def interpolate_nans(data, nan_mask):
    """signal.py:507-522"""
    out = np.copy(data)
    for i in range(data.shape[1]):
        nans = nan_mask[:, i]
        try:
            out[nans, i] = np.interp(nans.nonzero()[0], (~nans).nonzero()[0], data[~nans, i])
        except Exception:
            pass
    return out


def _double_exp(X, P):
    """signal.py:748-751"""
    return P[0] * np.exp(-X * np.abs(P[1])) + P[2] * np.exp(-X * np.abs(P[3])) + np.abs(P[-1])


def _error(P, f, X, Y):
    """signal.py:754-757"""
    return np.sum(np.power(f(X, P) - Y, 2))


def calc_photobl_one(y):
    """signal.py:759-793 for one trace. Returns (maxY, P)."""
    X = np.arange(len(y))
    maxY = 0.0
    P_out = np.zeros(5)
    try:
        maxY = np.max(y)
        Y = y / maxY
        mask = np.ones_like(Y, dtype=bool)
        P = np.array([1., 0.006, 1., 0.001, 0.2])
        it = 0
        while it < 100:
            R = minimize(_error, P, args=(_double_exp, X[mask], Y[mask]))
            if np.sum(np.absolute((P - R.x) / P)) < 1e-2:
                break
            P = R.x
            std = np.std(_double_exp(X[mask], P) - Y[mask])
            mask[:] = np.absolute(_double_exp(X, P) - Y) < 2. * std
            it += 1
        P_out = P
    except Exception:
        pass
    return maxY, P_out


def appl_photobl_one(y, maxY, P):
    """signal.py:795-821"""
    X = np.arange(len(y))
    fit = _double_exp(X, P) * maxY
    if np.any(np.abs(fit) < 1e-2):
        return y, False
    return y / fit * fit[0], True


def remove_spikes(data):
    """signal.py:594-613 (in place on a copy)"""
    data = np.copy(data)
    for j in range(data.shape[1]):
        if np.all(~np.isnan(data[:, j])):
            tot_std = np.nanstd(data[:, j])
            spikes = np.where(data[:, j] - np.average(data[:, j]) > tot_std * 5)[0]
            for s in spikes:
                data[s, j] = data[s - 1, j]
    return data


def smooth_sg_causal(data, n=13, poly=1):
    """signal.py:542-592, mode sg_causal"""
    sm = savgol_coeffs(n, poly, pos=n - 1)
    shift = (n - 1) // 2
    out = np.copy(data)
    for i in range(data.shape[1]):
        out[shift:, i] = np.convolve(data[:, i], sm, mode="same")[:-shift]
    return out


def rolling_window(a, window):
    """signal.py:1056-1063"""
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window - 1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad, mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def loc_std(y, window=8):
    """signal.py:991-1035, 1-D case"""
    if len(y) == 0:
        return 0.0
    return np.sqrt(np.median(np.var(rolling_window(np.asarray(y, float), window), axis=-1)))


def max_nan_run(mask):
    """Longest contiguous True run; Fconn.nan_ok(a, n) == (max_nan_run(a) <= n)  (Fconn.py:1741-1758)."""
    if not np.any(mask):
        return 0
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return int(np.max(np.where(d == -1)[0] - np.where(d == 1)[0]))


# ---------------------------------------------------------------- per-event features (Funatlas.py:2361-2421)
SDERKER = savgol_coeffs(39, 2, deriv=2, delta=DT)
FDERKER = savgol_coeffs(39, 1, pos=38, deriv=1, delta=DT)
SMOOTHKER = savgol_coeffs(39, 1, deriv=0, delta=DT)


def neuron_convs(trace):
    return (np.convolve(SDERKER, trace, mode="same"), np.convolve(FDERKER, trace, mode="same"),
            np.convolve(SMOOTHKER, trace, mode="same"))


def significance_features(trace, convs, i0, i1, shift_vol=SHIFT_VOL):
    """Port of Funatlas.get_significance_features WITHOUT the nan check (returned separately so the
    threshold can be applied later). Returns dict with dff (dF/sigma, decay-corrected), sd (mean 2nd
    derivative over 15 s post), dFF (intensity-map dF/F = mean(dytrace_unnorm)/mean(y[:shift_vol])),
    pre_zero flag.  Funatlas.py:2364-2421 and funatlas_plot_intensity_map.py:184-200."""
    sd, fd, smoothy = convs
    y = trace[i0:i1]
    baseline = np.average(y[:shift_vol])
    pre = loc_std(y, 8)
    out = dict(pre_zero=(pre == 0.0), dff=np.nan, sd=np.nan, dFF=np.nan, corrected=False)
    if pre == 0.0:
        return out
    with np.errstate(all='ignore'):
        dytrace_unnorm = y[shift_vol:shift_vol + 60] - baseline
        sd__ = np.average(sd[i0 + shift_vol:i0 + shift_vol + 30])
        fd_0 = fd[i0 + shift_vol]
        if fd_0 < 0:
            if i0 + shift_vol + 60 < smoothy.shape[0]:
                zero = smoothy[i0 + shift_vol + 60]
                length_remaining = 60
            else:
                zero = smoothy[-1]
                length_remaining = smoothy.shape[0] - (i0 + shift_vol)
            time = np.arange(length_remaining) * DT
            A = smoothy[i0 + shift_vol] - zero
            corry = A * np.exp(fd_0 * time) + zero
            dytrace_unnorm = y[shift_vol:shift_vol + length_remaining] - corry
            out['corrected'] = True
        dy = np.average(dytrace_unnorm / pre)
        out['dff'] = dy
        out['sd'] = sd__
        # intensity-map dF/F (funatlas_plot_intensity_map.py:192-200); 'pre' there is the mean baseline
        out['dFF'] = np.average(dytrace_unnorm) / baseline if baseline != 0 else np.nan
    return out


def stim_windows(ee, T):
    """Fconn.py:242-247: i0, per-stim shift_vol, i1."""
    i0 = max(0, ee - SHIFT_VOL)
    sv = SHIFT_VOL if i0 > 0 else ee
    i1 = min(int(i0 + 60 / DT + sv), T)
    return i0, sv, i1


# ---------------------------------------------------------------- statistics
def storey_qvalue(pvals):
    """multipy 0.16 fdr.py:156-214 (Storey & Tibshirani 2003 with cubic-spline pi0)."""
    m, pvals = len(pvals), np.asarray(pvals)
    ind = np.argsort(pvals)
    rev_ind = np.argsort(ind)
    pvals = pvals[ind]
    kappa = np.arange(0, 0.96, 0.01)
    pik = [sum(pvals > k) / (m * (1 - k)) for k in kappa]
    cs = UnivariateSpline(kappa, pik, k=3, s=None, ext=0)
    pi0 = float(cs(1.))
    if pi0 < 0 or pi0 > 1:
        pi0 = 1
    qvals = np.zeros(np.shape(pvals))
    qvals[-1] = pi0 * pvals[-1]
    for i in range(m - 2, -1, -1):
        qvals[i] = min(pi0 * m * pvals[i] / float(i + 1), qvals[i + 1])
    return qvals[rev_ind], pi0


def pair_p(ctrl_dff, ctrl_sd, dff_, sd_):
    """Funatlas.py:2647-2653"""
    _, pdff = kstest(ctrl_dff, dff_, alternative="two-sided")
    _, psd = kstest(ctrl_sd, sd_, alternative="two-sided")
    if pdff == 0: pdff = 1e-10
    if psd == 0: psd = 1e-10
    _, p = combine_pvalues([pdff, psd], method="fisher")
    return p, pdff, psd


def pair_tost_p(ctrl_dff, ctrl_sd, dff_, sd_, tost_th=1.2):
    """Funatlas.py:2590-2597, 2655-2661"""
    from statsmodels.stats.weightstats import ttost_ind
    l1 = np.std(ctrl_dff) * tost_th
    l2 = np.std(ctrl_sd) * tost_th
    t1, _, _ = ttost_ind(ctrl_dff, dff_, -l1, l1)
    t2, _, _ = ttost_ind(ctrl_sd, sd_, -l2, l2)
    if t1 == 0: t1 = 1e-10
    if t2 == 0: t2 = 1e-10
    _, p = combine_pvalues([t1, t2], method="fisher")
    return p


def q_matrix(pmat):
    """Funatlas.py:2684-2689"""
    pmat = np.copy(pmat)
    pmat[pmat == 0] = 1e-10
    fin = np.isfinite(pmat)
    q = np.full_like(pmat, np.nan)
    qv, pi0 = storey_qvalue(pmat[fin])
    q[fin] = qv
    return q, pi0
