"""YIN-style F0 tracker (vectorised, chunked, no external DSP dependencies)."""
import numpy as np
from scipy.signal import resample_poly

__all__ = ['f0_yin', 'to_midi']


def to_midi(f0):
    """Hz -> fractional MIDI note numbers (NaN stays NaN)."""
    f0 = np.asarray(f0, dtype=np.float64)
    out = np.full(f0.shape, np.nan)
    v = f0 > 0
    out[v] = 69.0 + 12.0 * np.log2(f0[v] / 440.0)
    return out


def f0_yin(x, fs, fmin=60.0, fmax=1100.0, frame=0.040, hop=0.010, thr=0.15,
           chunk=3000, tgt_sr=16000):
    """Cumulative-mean-normalised difference function, parabolic interpolation,
    median-filtered over voiced runs.  Returns (f0 Hz, aperiodicity)."""
    if fs != tgt_sr:
        g = np.gcd(int(fs), tgt_sr)
        x = resample_poly(x, tgt_sr // g, fs // g)
    fs = tgt_sr
    x = np.asarray(x, dtype=np.float32)
    nf, nh = max(1, int(frame * fs)), max(1, int(hop * fs))
    if len(x) < nf + 2:
        return np.zeros(0), np.zeros(0)
    nfr = 1 + (len(x) - nf) // nh
    lo = max(1, int(fs / fmax))
    hi = min(nf - 1, int(fs / fmin))
    nfft = 1 << int(np.ceil(np.log2(2 * nf)))
    f0 = np.zeros(nfr, dtype=np.float32)
    ap = np.ones(nfr, dtype=np.float32)
    for s0 in range(0, nfr, chunk):
        s1 = min(nfr, s0 + chunk)
        idx = (nh * np.arange(s0, s1))[:, None] + np.arange(nf)[None, :]
        fr = x[idx].astype(np.float64)
        fr -= fr.mean(axis=1, keepdims=True)
        spec = np.fft.rfft(fr, nfft, axis=1)
        ac = np.fft.irfft(spec * np.conj(spec), nfft, axis=1)[:, :nf]
        ac0 = np.maximum(ac[:, :1], 1e-12)
        diff = np.maximum(ac0 + ac[:, :1] - 2 * ac, 0.0)
        cum = np.cumsum(diff[:, 1:], axis=1)
        cmnd = np.ones_like(diff)
        with np.errstate(divide='ignore', invalid='ignore'):
            cmnd[:, 1:] = diff[:, 1:] / np.maximum(cum / np.arange(1, nf), 1e-12)
        seg = cmnd[:, lo:hi]
        ok = seg.min(axis=1) < thr
        first = np.argmax(seg < thr, axis=1)
        for j in np.nonzero(ok)[0]:
            t = first[j]
            while t + 1 < seg.shape[1] and seg[j, t + 1] < seg[j, t]:
                t += 1
            if 0 < t < seg.shape[1] - 1:
                a, b, c = seg[j, t - 1], seg[j, t], seg[j, t + 1]
                den = a - 2 * b + c
                d = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
            else:
                d = 0.0
            lag = lo + t + d
            f0[s0 + j] = fs / lag if lag > 0 else 0.0
            ap[s0 + j] = seg[j, t]
    v = f0 > 0
    if v.sum() > 4:
        med = f0.copy()
        for i in range(1, nfr - 1):
            if v[i - 1] and v[i] and v[i + 1]:
                med[i] = np.median(f0[i - 1:i + 2])
        f0 = med
    return f0.astype(np.float64), ap.astype(np.float64)
