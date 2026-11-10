"""BS.1770 K-weighted loudness, true peak, and the look-ahead limiter."""
import math

import numpy as np
from scipy.signal import butter, lfilter, resample_poly, sosfilt
from scipy.ndimage import minimum_filter1d

__all__ = ['k_weighting_coeffs', 'k_weight', 'integrated_loudness', 'loudness_curve',
           'loudness_range', 'true_peak_db', 'true_peak_limit', 'lufs', 'band_limit',
           'frame_ms', 'to_db', 'smooth_db']


# ------------------------------------------------------------------ K-weighting
def k_weighting_coeffs(fs):
    """ITU-R BS.1770 stage-1 shelf + stage-2 RLB high-pass, for any sample rate."""
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = np.tan(np.pi * f0 / fs)
    Vh = 10.0 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b1 = [(Vh + Vb * K / Q + K * K) / a0, 2.0 * (K * K - Vh) / a0,
          (Vh - Vb * K / Q + K * K) / a0]
    a1 = [1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0]
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = np.tan(np.pi * f0 / fs)
    d = 1.0 + K / Q + K * K
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, 2.0 * (K * K - 1.0) / d, (1.0 - K / Q + K * K) / d]
    return (np.array(b1), np.array(a1)), (np.array(b2), np.array(a2))


def k_weight(x, fs):
    (b1, a1), (b2, a2) = k_weighting_coeffs(fs)
    return lfilter(b2, a2, lfilter(b1, a1, x))


def _blocks(z_ms, block, hop):
    n = len(z_ms)
    if n < block:
        return np.array([z_ms.mean()]) if n else np.array([])
    nblk = 1 + (n - block) // hop
    c = np.concatenate(([0.0], np.cumsum(z_ms)))
    idx = np.arange(nblk) * hop
    return (c[idx + block] - c[idx]) / block


def _lufs_of_ms(ms):
    return -0.691 + 10.0 * np.log10(np.maximum(ms, 1e-30))


def integrated_loudness(z_ms, fs):
    """Gated integrated loudness (LKFS) per BS.1770."""
    ms = _blocks(z_ms, int(round(0.400 * fs)), int(round(0.100 * fs)))
    if ms.size == 0:
        return -np.inf
    l = _lufs_of_ms(ms)
    keep = l > -70.0
    if not keep.any():
        return -np.inf
    rel = _lufs_of_ms(ms[keep].mean()) - 10.0
    keep2 = keep & (l > rel)
    if not keep2.any():
        return -np.inf
    return float(_lufs_of_ms(ms[keep2].mean()))


def loudness_curve(z_ms, fs, win=0.400, hop=0.100, absolute_gate=True):
    ms = _blocks(z_ms, int(round(win * fs)), int(round(hop * fs)))
    if ms.size == 0:
        return np.array([])
    l = _lufs_of_ms(ms)
    return np.where(l > -70.0, l, np.nan) if absolute_gate else l


def loudness_range(z_ms, fs):
    """EBU R128 LRA: 3 s window, 100 ms hop, -70 absolute and -20 relative gate."""
    st = loudness_curve(z_ms, fs, win=3.0, hop=0.1, absolute_gate=True)
    st = st[~np.isnan(st)]
    if st.size < 2:
        return 0.0
    mean_p = np.mean(10.0 ** ((st + 0.691) / 10.0))
    st = st[st > -0.691 + 10.0 * np.log10(mean_p) - 20.0]
    if st.size < 2:
        return 0.0
    return float(np.percentile(st, 95) - np.percentile(st, 10))


def lufs(x, fs):
    """Convenience: integrated loudness of a waveform (mono or multichannel->mono)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    z = k_weight(x, fs)
    return integrated_loudness(z * z, fs)


# ------------------------------------------------------------------- true peak
def true_peak_db(x, fs, oversample=4, block=1 << 18):
    """dBTP by oversampled peak detection (BS.1770 Annex 2 spirit)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.max(axis=1) if x.shape[1] > 1 else x[:, 0]
    n = len(x)
    if n == 0:
        return -np.inf
    ov, peak = 4096, 0.0
    for s0 in range(0, n, block):
        s1 = min(n, s0 + block)
        a, b = max(0, s0 - ov), min(n, s1 + ov)
        y = resample_poly(x[a:b], oversample, 1)
        h = ov * oversample
        seg = y[h:len(y) - h] if len(y) > 2 * h else y
        if seg.size:
            peak = max(peak, float(np.max(np.abs(seg))))
    return 20.0 * np.log10(max(peak, 1e-12))


def true_peak_limit(x, fs, ceiling_db=-2.0, lookahead=0.012, release=0.200,
                    oversample=4):
    """Look-ahead true-peak limiter with conventional 12 ms / 200 ms defaults.

    Compliance is guaranteed by a sliding minimum of the required gain over the
    look-ahead window; the one-pole release only shapes the recovery curve.
    Returns (limited_signal, applied_gain_db_at_worst_point).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x, 0.0
    ceil = 10 ** (ceiling_db / 20.0)
    y = resample_poly(x, oversample, 1)
    g_req = np.minimum(1.0, ceil / np.maximum(np.abs(y), 1e-12))
    if g_req.min() >= 1.0:
        return x, 0.0
    la = max(1, int(round(lookahead * fs * oversample)))
    g_min = minimum_filter1d(g_req, size=2 * la + 1, mode='nearest')
    a = math.exp(-1.0 / max(1.0, release * fs * oversample))
    g_rel = lfilter([1 - a], [1, -a], g_min, zi=[a * g_min[0]])[0]
    g = np.minimum(g_min, g_rel)
    out = resample_poly(y * g, 1, oversample)[:len(x)]
    return out, 20.0 * math.log10(max(float(np.min(g)), 1e-9))


# ------------------------------------------------------------------- utilities
def band_limit(x, fs, lo=80.0, hi=4000.0):
    hi = min(hi, fs / 2 * 0.95)
    sos = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype='band', output='sos')
    return sosfilt(sos, x)


def frame_ms(x, fs, frame=0.025, hop=0.010, chunk=4000):
    """Mean square per frame (linear energy)."""
    nf, nh = int(round(frame * fs)), int(round(hop * fs))
    if len(x) < nf:
        return np.array([])
    n = 1 + (len(x) - nf) // nh
    out = np.empty(n)
    for s0 in range(0, n, chunk):
        s1 = min(n, s0 + chunk)
        idx = (nh * np.arange(s0, s1))[:, None] + np.arange(nf)[None, :]
        out[s0:s1] = np.mean(x[idx].astype(np.float64) ** 2, axis=1)
    return out


def to_db(ms):
    return 10 * np.log10(np.maximum(ms, 1e-20))


def smooth_db(g, win_s, hop=0.010):
    k = max(1, int(round(win_s / hop)))
    if k < 3 or len(g) < k:
        return g
    w = np.hanning(k)
    w /= w.sum()
    pad = k // 2
    return np.convolve(np.pad(g, (pad, k - 1 - pad), mode='edge'), w, mode='valid')
