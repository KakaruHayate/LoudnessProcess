"""Pitch-conditioned level normalisation plus the loudness-compliance chain.

For one source (a whole recording, or a group of clips cut from the same recording):

    gain(t) = DC + (B_tilt - B_source) * (p(t) - pbar)

* B_tilt   : the slope the model should render INSIDE a clip (default +6 dB/oct).
             A constant gain cannot change it, so this term is a shape change only.
* DC       : where the clip sits on the loudness axis,
               target + design_offset(pbar - pbar_ref) + delta
             with design_offset linear (+B_seg dB per semitone) and, optionally,
             soft-saturated at +-tol so a hard tolerance window is respected.
* delta    : one number for the whole dataset, so the corpus hits the target.
* limiter  : true-peak ceiling, engaged only where a file would exceed it.

No compression, no EQ, no other processing is applied.
"""
import math
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .dataset import active_mask, features, frame_slope, order_key, pbar_of
from .loudness import lufs, smooth_db, true_peak_db, true_peak_limit

HOP = 0.010
SMOOTH_S = 0.070

__all__ = ['Options', 'design_offset', 'plan_dataset', 'process_group']


def design_offset(dpitch, b_semi, tol_window=0.0):
    """Level offset versus reference pitch: linear, optionally soft-saturated."""
    if tol_window and tol_window > 0:
        return tol_window * np.tanh(b_semi * dpitch / tol_window)
    return b_semi * dpitch


SRC_SUBTYPE = {16: 'PCM_16', 24: 'PCM_24', 32: 'FLOAT'}


class Options:
    """All processing parameters in one place (see the CLI for the mapping)."""

    def __init__(self, target=-24.0, tol=2.0, tp_max=-2.0, b_tilt_oct=6.0,
                 b_seg_oct=None, lam=0.6, dc_scope='pitch', level_scope='corpus',
                 limiter=True, lookahead=0.012, release=0.200, smooth=SMOOTH_S,
                 keep_tree=True, out_sr=0, out_bits=0, out_channels=0, dither=True):
        self.target = target
        self.tol = tol
        self.tp_max = tp_max
        self.b_tilt = b_tilt_oct / 12.0
        self.b_seg = (b_seg_oct if b_seg_oct is not None else b_tilt_oct) / 12.0
        self.lam = lam
        self.dc_scope = dc_scope
        self.level_scope = level_scope
        self.limiter = limiter
        self.lookahead = lookahead
        self.release = release
        self.smooth = smooth
        self.keep_tree = keep_tree
        self.out_sr = int(out_sr or 0)
        self.out_bits = int(out_bits or 0)
        self.out_channels = int(out_channels or 0)
        self.dither = dither
        self.out_subtype = SRC_SUBTYPE.get(self.out_bits)


def convert(x, fs, opts):
    """Bring a signal onto the requested export format.

    Channel reduction averages rather than sums: a dual-mono file read as stereo
    is 3.01 LU louder than the same file as mono, which is the classic way to
    build an inconsistent corpus.
    """
    if opts.out_sr and int(opts.out_sr) != int(fs):
        g = math.gcd(int(opts.out_sr), int(fs))
        x = resample_poly(x, int(opts.out_sr) // g, int(fs) // g, axis=0)
        fs = int(opts.out_sr)
    if opts.out_channels and x.shape[1] != opts.out_channels:
        if opts.out_channels == 1:
            x = x.mean(axis=1, keepdims=True)
        else:
            x = np.repeat(x.mean(axis=1, keepdims=True), opts.out_channels, axis=1)
    return x, fs


def _quantise(y, opts):
    """TPDF dither at +-1 LSB when reducing to 16 bit or below."""
    if not opts.dither or not opts.out_bits or opts.out_bits > 16:
        return y
    lsb = 1.0 / (2 ** (opts.out_bits - 1))
    return y + (np.random.random(y.shape) - np.random.random(y.shape)) * lsb


# --------------------------------------------------------------------- planning
def plan_dataset(rows, opts):
    """Estimate per-source slopes and the single global offset before touching audio.

    `rows` are analyze_file() outputs; no per-frame data is needed here.
    """
    groups = {}
    for r in rows:
        groups.setdefault(r['group'], []).append(r)

    ref = (float(np.nanmedian([r['pbar'] for r in rows if r['pbar'] is not None]))
           if any(r['pbar'] is not None for r in rows) else 60.0)

    b_song = {}
    for g, items in groups.items():
        use = [x for x in items if x['pbar'] is not None and x['span'] >= 5.0]
        if len(use) >= 3 and np.ptp([x['pbar'] for x in use]) >= 2.0:
            P = np.array([x['pbar'] for x in use])
            Y = np.array([x['lufs'] for x in use])
            W = np.array([x['dur'] for x in use])
            W = W / W.sum()
            b_song[g] = float(np.polyfit(P, Y, 1, w=np.sqrt(W))[0])
        else:
            b_song[g] = None          # fall back to the frame-level fit per file

    delta = 0.0
    if opts.dc_scope == 'pitch' and opts.level_scope == 'corpus':
        wl, du = [], []
        for r in rows:
            if r['pbar'] is None:
                continue
            wl.append(opts.target + design_offset(r['pbar'] - ref, opts.b_seg, opts.tol))
            du.append(r['dur'])
        if wl:
            wl, du = np.array(wl), np.array(du)
            w = 10 ** ((wl + 0.691) / 10.0) * du
            corpus = 10 * np.log10(w.sum() / du.sum()) - 0.691
            delta = opts.target - corpus
    return dict(groups=groups, ref=ref, b_song=b_song, delta=delta)


# ------------------------------------------------------------------ processing
def process_group(group, paths, out_dir, in_root, opts, b_song, ref, delta):
    """Normalise every clip of one source, write it out, return report rows."""
    paths = sorted(paths, key=lambda p: order_key(p))
    recs = []
    for p in paths:
        try:
            x, fs = sf.read(p, dtype='float64', always_2d=True)
            x, fs = convert(x, fs, opts)
            f = features(p, x, fs)
        except Exception:
            continue
        recs.append(dict(path=p, x=x, lufs=lufs(f['mono'], fs), **f))
    if not recs:
        return []
    fs = recs[0]['fs']

    for r in recs:
        r['pbar'] = pbar_of(r['L'], r['midi'], r['act'])
        a = r['act'] & np.isfinite(r['midi'])
        r['span'] = float(np.ptp(r['midi'][a])) if a.sum() else 0.0
        r['bseg'] = frame_slope(r['L'], r['midi'], r['act'], min_span=5.0)

    unit_slope = b_song.get(group) if isinstance(b_song, dict) else None
    if unit_slope is None:
        Lc = np.concatenate([r['L'] for r in recs])
        unit_slope = frame_slope(Lc, np.concatenate([r['midi'] for r in recs]),
                                 np.concatenate([r['act'] for r in recs]))
    b_new = (None if unit_slope is None
             else opts.lam * opts.b_tilt + (1 - opts.lam) * unit_slope)

    offs = np.cumsum([0] + [len(r['L']) for r in recs])
    g = np.zeros(int(offs[-1]))
    for i, r in enumerate(recs):
        a, b = offs[i], offs[i + 1]
        pb = r['pbar'] if r['pbar'] is not None else ref
        if opts.dc_scope == 'constant':
            want = opts.target
        else:
            want = opts.target + delta + design_offset(pb - ref, opts.b_seg, opts.tol)
            if opts.tol:
                # inset the clamp by the level-loop tolerance (0.02 dB) so that a clip
                # sitting exactly on the boundary cannot end up outside the window
                edge = max(opts.tol - 0.03, 0.0)
                want = float(np.clip(want, opts.target - edge, opts.target + edge))
        cgain = want - r['lufs']
        tilt = np.zeros(b - a)
        bs = r['bseg'] if r['bseg'] is not None else unit_slope
        if bs is not None and b_new is not None:
            m = r['act'] & np.isfinite(r['midi'])
            if m.sum() > 20:
                tilt[m] = (b_new - bs) * (r['midi'][m] - pb)
                tilt = np.interp(np.arange(b - a), np.where(m)[0], tilt[m])
                tilt = smooth_db(tilt, opts.smooth)
        g[a:b] = cgain + tilt
        r['want'] = want

    rows = []
    for i, r in enumerate(recs):
        nf = len(r['L'])
        gseg = g[offs[i]:offs[i] + nf]
        gain = 10 ** (np.interp(np.arange(len(r['mono'])) / fs,
                                np.arange(nf) * HOP, gseg) / 20.0)
        y = r['x'] * gain[:, None]
        note = []
        if opts.level_scope != 'none':
            # constant gain only -> the intra-clip dynamics are untouched
            for _ in range(4):
                cur = lufs(y.mean(axis=1), fs)
                if abs(r['want'] - cur) < 0.02:
                    break
                y = y * 10 ** ((r['want'] - cur) / 20.0)
        if opts.limiter:
            for _ in range(4):
                tp = true_peak_db(y, fs)
                if tp <= opts.tp_max:
                    break
                y, gr = true_peak_limit(y, fs, opts.tp_max - 0.08,
                                        opts.lookahead, opts.release)
                note.append('lim%.1fdB' % min(gr, 0.0))
            tp = true_peak_db(y, fs)
            if tp > opts.tp_max:      # last resort: the spec is a hard guarantee
                y = y * 10 ** ((opts.tp_max - 0.02 - tp) / 20.0)
                note.append('trim%.1fdB' % (opts.tp_max - 0.02 - tp))
        l_after = lufs(y.mean(axis=1), fs)
        tp_after = true_peak_db(y, fs)
        ok = (abs(l_after - opts.target) <= (opts.tol or 1.0) + 0.01
              and tp_after <= opts.tp_max + 0.01)
        if not ok:
            note.append('OUT-OF-SPEC')
        rel = (os.path.relpath(r['path'], in_root) if opts.keep_tree
               else os.path.basename(r['path']))
        outp = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(outp) or out_dir, exist_ok=True)
        info = sf.info(r['path'])
        sub = opts.out_subtype or (info.subtype
                                   if info.subtype in ('PCM_16', 'PCM_24', 'FLOAT')
                                   else 'PCM_24')
        sf.write(outp, _quantise(y, opts), fs, subtype=sub)
        rows.append(dict(file=os.path.basename(r['path']), group=group,
                         dur=round(len(r['mono']) / fs, 2),
                         pbar=round(r['pbar'], 1) if r['pbar'] else '',
                         lufs_before=round(r['lufs'], 2),
                         want=round(r['want'], 2),
                         lufs_after=round(l_after, 2),
                         tp_after=round(tp_after, 2),
                         ok=int(ok),
                         gain_db=round(l_after - r['lufs'], 2),
                         slope_seg_oct=round(r['bseg'] * 12, 2) if r['bseg'] else '',
                         source_slope_oct=round(unit_slope * 12, 2) if unit_slope else '',
                         note=' '.join(note)))
    return rows
