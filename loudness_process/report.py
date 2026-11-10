"""Report writers: CSV for tooling, Markdown for humans."""
import csv
import os

import numpy as np

__all__ = ['write_csv', 'audit_markdown', 'result_markdown', 'save_markdown']


def save_markdown(path, text):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def write_csv(path, rows, fields=None):
    if not rows:
        return
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fields = fields or list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def audit_markdown(a, title='Loudness / dynamics audit'):
    if not a:
        return '# %s\n\nNo audio found.\n' % title
    out = ['# %s' % title, '', '## Overview', '',
           '| item | value |', '|---|---|',
           '| files | %d |' % a['n'],
           '| sources | %d |' % a['groups'],
           '| duration | %.0f min |' % a['minutes'],
           '| **corpus loudness (energy weighted)** | **%.2f LKFS** |' % a['corpus'],
           '| per-file LUFS | %.2f +/- %.2f (%.2f .. %.2f) |'
           % (a['mean'], a['sd'], a['lo'], a['hi']),
           '| max true peak | %.2f dBTP |' % a['tp_max_obs'],
           '| mean-pitch spread | %.2f semitones |' % a['pitch_sd'],
           '', '## Formats', '',
           '| channels | sample rate | subtype | files |', '|---|---|---|---|']
    for ch, sr, sub, n in a['formats']:
        out.append('| %d | %d | %s | %d |' % (ch, sr, sub, n))

    out += ['', '## Dynamic consistency', '',
            '| metric | value | reading |', '|---|---|---|']
    ss = a['seg_slope_oct']
    if len(ss):
        out.append('| intra-clip pitch slope (median / sd) | %+.2f / %.2f dB/oct | %s |'
                   % (np.median(ss), np.std(ss),
                      'consistent' if np.std(ss) <= 1.5
                      else 'inconsistent: the model would learn contradictory dynamics'))
    so = a['song_slope_oct']
    if len(so):
        out.append('| inter-clip pitch slope (per source) | median %+.2f, sd %.2f dB/oct '
                   '| target %+.1f |' % (np.median(so), np.std(so), a['b_target_oct']))
    if len(a['song_res']):
        out.append('| inter-clip residual (unexplained by pitch) | %.2f dB | %s |'
                   % (np.median(a['song_res']),
                      'good' if np.median(a['song_res']) < 1.0
                      else 'high: renders will jump in loudness'))

    out += ['', '## Delivery spec (target %.1f LKFS +/-%.1f LU, TP <= %.1f dBTP)'
            % (a['target'], a['tol'], a['tp_max']), '',
            '| check | result |', '|---|---|',
            '| files inside +/-%.1f LU | **%.1f%%** |' % (a['tol'], a['inside']),
            '| files over %.1f dBTP | %d |' % (a['tp_max'], a['tp_over']),
            '| corpus loudness | %.2f LKFS (target %.1f) |' % (a['corpus'], a['target'])]
    verdict = ('**pass**' if (a['inside'] >= 99.9 and a['tp_over'] == 0
                              and abs(a['corpus'] - a['target']) <= 0.5)
               else '**fail - normalise first**')
    out += ['', 'Conclusion: %s' % verdict]

    dev, names = a['dev'], a.get('names') or []
    worst = [i for i in np.argsort(-np.abs(dev))[:10] if np.isfinite(dev[i])]
    out += ['', '## Clips furthest from the design pitch line', '',
            '| # | file | deviation (dB) |', '|---|---|---|']
    for k, i in enumerate(worst):
        out.append('| %d | %s | %+.2f |'
                   % (k + 1, names[i] if i < len(names) else i, dev[i]))
    out += ['', '> Large deviations usually mean a wrong take, a mis-cut clip, or an '
            'inconsistent recording gain.', '']
    return '\n'.join(out)


def result_markdown(rows, opts, plan, title='Processed dataset'):
    if not rows:
        return '# %s\n\nNothing was written.\n' % title
    L = np.array([r['lufs_after'] for r in rows])
    T = np.array([r['dur'] for r in rows])
    TP = np.array([r['tp_after'] for r in rows])
    w = 10 ** ((L + 0.691) / 10.0) * T
    corpus = 10 * np.log10(w.sum() / T.sum()) - 0.691
    ok = sum(int(r['ok']) for r in rows)
    lim = sum(1 for r in rows if r['note'].startswith('lim'))
    bs = [v for v in plan['b_song'].values() if v is not None]

    out = ['# %s' % title, '', '## Parameters', '',
           '| parameter | value |', '|---|---|',
           '| loudness target | %.1f LKFS |' % opts.target,
           '| tolerance window | +/-%.1f LU |' % opts.tol,
           '| true-peak ceiling | %.1f dBTP (look-ahead %.0f ms / release %.0f ms) |'
           % (opts.tp_max, opts.lookahead * 1000, opts.release * 1000),
           '| intra-clip design slope | +%.1f dB/oct (partial, lambda=%.2f) |'
           % (opts.b_tilt * 12, opts.lam),
           '| inter-clip design slope | +%.1f dB/oct |' % (opts.b_seg * 12),
           '| corpus offset delta | %+.2f dB |' % plan['delta'],
           '| reference pitch | %.1f (MIDI) |' % plan['ref'],
           '', '## Result', '',
           '| item | value |', '|---|---|',
           '| files | %d |' % len(rows),
           '| **corpus loudness** | **%.2f LKFS** |' % corpus,
           '| per-file LUFS | %.2f +/- %.2f (%.2f .. %.2f) |'
           % (L.mean(), L.std(), L.min(), L.max()),
           '| max true peak | %.2f dBTP |' % TP.max(),
           '| within spec | **%d/%d** |' % (ok, len(rows)),
           '| limited | %d files |' % lim]
    if bs:
        out.append('| source slope (in -> out, median) | %+.1f -> %+.1f dB/oct |'
                   % (np.median(bs) * 12,
                      np.median([opts.lam * opts.b_tilt + (1 - opts.lam) * b
                                 for b in bs]) * 12))
    out += ['', '> The `want` column of the report CSV stores the intended level of every '
            'clip (the design pitch line). If your acoustic model accepts an explicit '
            'energy/dynamics channel, that column can be fed to it directly to get '
            'controllable dynamics back without leaving the loudness spec.', '']
    return '\n'.join(out)
