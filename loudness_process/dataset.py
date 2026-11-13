"""Dataset discovery, per-file features, source grouping and the audit report."""
import glob
import os
import re

import numpy as np
import soundfile as sf

from .loudness import (band_limit, frame_ms, to_db, true_peak_db, lufs)
from .pitch import f0_yin, to_midi

HOP = 0.010
MIN_SPAN_FRAME = 7.0     # semitones; a frame-level slope fit needs at least this
MIN_SPAN_SEG = 5.0       # semitones; for a segment to enter the per-source fit

__all__ = ['discover', 'group_of', 'order_key', 'features', 'pbar_of',
           'frame_slope', 'analyze_file', 'load_features', 'audit', 'energy_lufs']


# ---------------------------------------------------------------- file naming
def group_of(name):
    """Name prefix -> source key, so that clips of the same recording are pooled.

    '123_045'      -> 'song123'   (numeric source id + clip index)
    'lead_02'      -> 'lead'      (anything before a trailing index)
    'Vox_1_00_EN'  -> 'Vox_1'     (a trailing 2-3 letter language tag is ignored)
    'vocal.wav'    -> 'vocal'     (no trailing index: the file is its own source)
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    m = re.match(r'^(.*)_[A-Za-z]{2,3}$', stem)      # drop a language tag
    if m:
        stem = m.group(1)
    m = re.match(r'^(\d+)_\d+$', stem)
    if m:
        return 'song' + m.group(1)
    m = re.match(r'^(.*)_\d+$', stem)
    return m.group(1) if m else stem


def group_key(path, root, by='name'):
    """Source key for a file.  by='dir' also pins the containing folder, so that
    the same song in two folders (different language, different style) is not
    merged into one source."""
    if by == 'dir':
        rel = os.path.relpath(os.path.dirname(path), root).replace('\\', '/')
        return rel + '/' + group_of(path)
    return group_of(path)


def order_key(name):
    """Sort key inside a source: trailing number, so clip order is preserved."""
    nums = re.findall(r'\d+', os.path.splitext(os.path.basename(name))[0])
    return int(nums[-1]) if nums else 0


def discover(root, pattern='*.wav', by='name'):
    """-> (paths, groups).  `pattern` may contain '**' for nested layouts."""
    paths = sorted(glob.glob(os.path.join(root, pattern), recursive='**' in pattern))
    groups = {}
    for p in paths:
        groups.setdefault(group_key(p, root, by), []).append(p)
    for g in groups:
        groups[g].sort(key=lambda p: order_key(p))
    return paths, groups


# ------------------------------------------------------------------- features
def features(path):
    """Per-frame level (band-limited), pitch in MIDI, activity mask, mono mix."""
    x, fs = sf.read(path, dtype='float64', always_2d=True)
    mono = x.mean(axis=1)
    ms = frame_ms(mono, fs)
    msb = frame_ms(band_limit(mono, fs), fs)
    f0, _ = f0_yin(mono, fs, frame=0.040, hop=HOP)
    n = min(len(ms), len(msb), len(f0))
    ms, msb, f0 = ms[:n], msb[:n], f0[:n]
    L = to_db(msb)
    midi = to_midi(f0).astype(np.float32)
    return dict(mono=mono, fs=fs, ms=ms, L=L, midi=midi, act=active_mask(L, f0))


def active_mask(L, f0):
    """Voiced frames that are clearly signal rather than noise.

    The noise-floor guard matters for material that was normalised up: a plain
    'within N dB of the loud end' rule then swallows the noise floor and wrecks
    the slope estimate.
    """
    v = (f0 > 0) & np.isfinite(L)
    if v.sum() < 50:
        return v
    floor = float(np.percentile(L, 10))
    hi = float(np.percentile(L[v], 95))
    m = v & (L > max(hi - 30.0, floor + 15.0))
    if m.sum() < max(50, int(0.02 * len(L))):
        m = v & (L > np.percentile(L[v], 40))
    return m


def pbar_of(L, P, A):
    """Energy-weighted mean pitch of the active frames, in MIDI."""
    a = A & np.isfinite(P)
    if a.sum() < 20:
        return None
    w = 10 ** (L[a].astype(np.float64) / 10.0)
    return float((P[a].astype(np.float64) * w).sum() / w.sum())


def frame_slope(L, P, A, min_span=MIN_SPAN_FRAME, min_frames=200):
    """dB per semitone of level against pitch, robust to 2.5-sigma outliers.

    `min_frames` must stay reachable for short clips: a 5-15 s clip typically has
    only 300-400 voiced frames, so a higher bar silently disables the estimate for
    most of a sliced dataset.
    """
    a = A & np.isfinite(P)
    if a.sum() < min_frames or np.ptp(P[a]) < min_span:
        return None
    b = float(np.polyfit(P[a].astype(np.float64), L[a].astype(np.float64), 1)[0])
    r = L[a] - b * P[a]
    keep = np.abs(r) < 2.5 * np.std(r) + 1e-9
    if keep.sum() >= max(80, int(0.3 * a.sum())):
        b = float(np.polyfit(P[a][keep].astype(np.float64),
                             L[a][keep].astype(np.float64), 1)[0])
    return b


def analyze_file(path):
    """Everything the audit and the normaliser need, for one file."""
    f = features(path)
    mono, fs = f['mono'], f['fs']
    a = f['act'] & np.isfinite(f['midi'])
    info = sf.info(path)
    return dict(path=path, name=os.path.basename(path),
                dur=len(mono) / fs, fs=fs, channels=info.channels,
                subtype=info.subtype,
                lufs=lufs(mono, fs), tp=true_peak_db(mono, fs),
                pbar=pbar_of(f['L'], f['midi'], f['act']),
                span=float(np.ptp(f['midi'][a])) if a.sum() else 0.0,
                bseg=frame_slope(f['L'], f['midi'], f['act'], min_span=MIN_SPAN_SEG),
                f=f)


def _safe_analyze(args):
    p, root, by = args
    try:
        d = analyze_file(p)
        d.pop('f', None)          # only the numbers are needed downstream
        d['group'] = group_key(p, root, by)
        return d
    except Exception:
        return None


def load_features(paths, jobs=10, progress=None, root=None, by='name'):
    """Parallel analyze_file(); failed files are skipped silently."""
    from concurrent.futures import ProcessPoolExecutor
    if root is None:
        root = os.path.dirname(paths[0]) if paths else '.'
    args = [(p, root, by) for p in paths]
    out = []
    if jobs <= 1:
        for i, r in enumerate(map(_safe_analyze, args)):
            if r:
                out.append(r)
            if progress and (i + 1) % 200 == 0:
                progress(i + 1, len(paths))
        return out
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, r in enumerate(ex.map(_safe_analyze, args, chunksize=4)):
            if r:
                out.append(r)
            if progress and (i + 1) % 200 == 0:
                progress(i + 1, len(paths))
    return out


# ---------------------------------------------------------------------- audit
def energy_lufs(L, dur):
    """Duration-weighted corpus loudness from per-file loudness values."""
    L = np.asarray(L, dtype=float)
    dur = np.asarray(dur, dtype=float)
    w = 10 ** ((L + 0.691) / 10.0) * dur
    return float(10 * np.log10(max(w.sum() / dur.sum(), 1e-12)) - 0.691)


def audit(rows, target=-24.0, tol=2.0, tp_max=-2.0, b_target_oct=6.0):
    """Summarise a dataset and score it against a delivery spec."""
    if not rows:
        return {}
    L = np.array([r['lufs'] for r in rows])
    T = np.array([r['dur'] for r in rows])
    TP = np.array([r['tp'] for r in rows])
    PB = np.array([r['pbar'] if r['pbar'] is not None else np.nan for r in rows])
    Bt = b_target_oct / 12.0
    ref = float(np.nanmedian(PB))
    A = float(np.nanmedian(L - Bt * (PB - ref)))
    dev = L - (A + Bt * (PB - ref))

    per_group = {}
    for r in rows:
        per_group.setdefault(r['group'], []).append(r)
    gslope, gres = [], []
    for g, items in per_group.items():
        use = [x for x in items if x['pbar'] is not None and x['span'] >= MIN_SPAN_SEG]
        if len(use) >= 3 and np.ptp([x['pbar'] for x in use]) >= 2.0:
            P = np.array([x['pbar'] for x in use])
            Y = np.array([x['lufs'] for x in use])
            W = np.array([x['dur'] for x in use])
            W = W / W.sum()
            fit = np.polyfit(P, Y, 1, w=np.sqrt(W))
            gslope.append(fit[0] * 12)
            gres.append(float(np.std(Y - np.polyval(fit, P))))

    fmt = {}
    for r in rows:
        k = (r['channels'], r['fs'], r['subtype'])
        fmt[k] = fmt.get(k, 0) + 1

    inside = np.abs(L - target) <= tol
    return dict(
        n=len(rows), minutes=float(T.sum() / 60), groups=len(per_group),
        corpus=energy_lufs(L, T), mean=float(L.mean()), sd=float(L.std()),
        lo=float(L.min()), hi=float(L.max()), tp_max_obs=float(TP.max()),
        tp_over=int(np.sum(TP > tp_max)),
        seg_slope_oct=np.array([r['bseg'] * 12 for r in rows if r['bseg'] is not None]),
        song_slope_oct=np.array(gslope), song_res=np.array(gres),
        pitch_sd=float(np.nanstd(PB)),
        dev=dev, inside=float(100 * inside.mean()),
        names=[os.path.basename(r['name']) for r in rows],
        formats=[(k[0], k[1], k[2], v) for k, v in sorted(fmt.items(), key=lambda kv: -kv[1])],
        target=target, tol=tol, tp_max=tp_max, b_target_oct=b_target_oct,
    )
