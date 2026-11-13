"""Command line interface.

    python -m loudness_process audit   <in>              # report only, no audio written
    python -m loudness_process process <in> <out>        # normalise and write
    python -m loudness_process run     <in> <out>        # audit + process
    python -m loudness_process verify  <in> <out>        # before/after metrics
    python -m loudness_process ab      <in> <out>        # listening comparison
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np

from .dataset import audit, discover, group_of, load_features
from .normalize import Options, plan_dataset, process_group
from .report import audit_markdown, result_markdown, save_markdown, write_csv

PRESETS = {
    # clips cut from longer recordings, sharing a per-recording name prefix
    'sliced': dict(glob='wavs/*.wav'),
    # one file per source recording
    'stems': dict(glob='*.wav'),
}
AUTO_GLOBS = ['wavs/*.wav', '**/wavs/*.wav', '*.wav', '**/*.wav']

BANNER = ('loudness_process - loudness/dynamics conditioning for '
          'singing-voice datasets')


def _say(msg=''):
    print(msg, flush=True)


def _progress(done, total):
    _say('   ... %d/%d' % (done, total))


def _options(a):
    return Options(target=a.target, tol=a.tol, tp_max=a.tp_max,
                   b_tilt_oct=a.b_tilt_oct, b_seg_oct=a.b_seg_oct, lam=a.lam,
                   dc_scope=a.dc_scope, level_scope=a.level_scope,
                   limiter=not a.no_limiter, lookahead=a.lookahead,
                   release=a.release, keep_tree=bool(a.keep_tree))


def _discover(a):
    root = a.indir
    pat = a.glob
    if pat is None:
        if a.preset in PRESETS:
            pat = PRESETS[a.preset]['glob']
        else:
            for cand in AUTO_GLOBS:
                if glob.glob(os.path.join(root, cand), recursive='**' in cand):
                    pat = cand
                    break
            pat = pat or '*.wav'
    paths, groups = discover(root, pat)
    if not paths:
        _say('!! no audio matched %r under %s' % (pat, root))
        sys.exit(2)
    if a.limit:
        keep = set(paths[:a.limit])
        paths = [p for p in paths if p in keep]
        groups = {}
        for p in paths:
            groups.setdefault(group_of(p), []).append(p)
    _say('%d files, %d sources  (pattern %r)' % (len(paths), len(groups), pat))
    return paths, groups, pat


def _worker(args):
    return process_group(*args)


def cmd_audit(a):
    paths, groups, pat = _discover(a)
    _say('analysing ...')
    rows = load_features(paths, jobs=a.jobs, progress=_progress)
    rep = audit(rows, target=a.target, tol=a.tol, tp_max=a.tp_max,
                b_target_oct=a.b_tilt_oct)
    out = a.outdir or os.path.join(a.indir, '_loudness_audit')
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, 'audit_files.csv'), rows)
    save_markdown(os.path.join(out, 'audit_report.md'),
                  audit_markdown(rep, title='Audit - %s'
                                 % os.path.basename(os.path.normpath(a.indir))))
    _say('')
    _say(audit_markdown(rep))
    _say('-> %s' % os.path.join(out, 'audit_report.md'))
    return rep


def cmd_process(a, plan_rows=None):
    paths, groups, pat = _discover(a)
    opts = _options(a)
    if plan_rows is None:
        _say('analysing ...')
        plan_rows = load_features(paths, jobs=a.jobs, progress=_progress)
    plan = plan_dataset(plan_rows, opts)
    bs = [v for v in plan['b_song'].values() if v is not None]
    _say('source slope: median %s dB/oct over %d sources'
         % (('%.2f' % (np.median(bs) * 12)) if bs else 'n/a', len(bs)))
    _say('corpus offset delta = %+.2f dB  (reference pitch %.1f)'
         % (plan['delta'], plan['ref']))

    tasks = sorted(plan['groups'].items(), key=lambda kv: -len(kv[1]))
    out_rows = []
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        payload = [(g, [r['path'] for r in items], a.outdir, a.indir, opts,
                    plan['b_song'].get(g), plan['ref'], plan['delta'])
                   for g, items in tasks]
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for i, rows in enumerate(ex.map(_worker, payload, chunksize=1)):
                out_rows.extend(rows)
                if (i + 1) % 25 == 0:
                    _say('   ... source %d/%d' % (i + 1, len(tasks)))
    else:
        for g, items in tasks:
            out_rows.extend(process_group(g, [r['path'] for r in items], a.outdir,
                                          a.indir, opts, plan['b_song'].get(g),
                                          plan['ref'], plan['delta']))

    os.makedirs(a.outdir, exist_ok=True)
    write_csv(os.path.join(a.outdir, '_report.csv'), out_rows)
    write_csv(os.path.join(a.outdir, '_groups.csv'),
              [dict(group=g, n_files=len(plan['groups'][g]),
                    slope_in_oct='' if v is None else round(v * 12, 2),
                    slope_out_oct='' if v is None else round(
                        (opts.lam * opts.b_tilt + (1 - opts.lam) * v) * 12, 2))
               for g, v in sorted(plan['b_song'].items())])
    if a.copy_sidecar:
        n = 0
        for p in glob.glob(os.path.join(a.indir, '**', a.sidecar_glob), recursive=True):
            base = os.path.basename(p)
            if base.startswith('_') or base.startswith('audit_'):
                continue
            dst = os.path.join(a.outdir, os.path.relpath(p, a.indir))
            os.makedirs(os.path.dirname(dst) or a.outdir, exist_ok=True)
            shutil.copy2(p, dst)
            n += 1
        _say('copied %d sidecar files matching %r' % (n, a.sidecar_glob))

    md = result_markdown(out_rows, opts, plan, title='Processed - %s'
                         % os.path.basename(os.path.normpath(a.outdir)))
    save_markdown(os.path.join(a.outdir, '_report.md'), md)
    _say('')
    _say(md)
    return out_rows


def cmd_run(a):
    paths, groups, pat = _discover(a)
    _say('analysing ...')
    rows = load_features(paths, jobs=a.jobs, progress=_progress)
    rep = audit(rows, target=a.target, tol=a.tol, tp_max=a.tp_max,
                b_target_oct=a.b_tilt_oct)
    os.makedirs(a.outdir, exist_ok=True)
    write_csv(os.path.join(a.outdir, 'audit_files_before.csv'), rows)
    save_markdown(os.path.join(a.outdir, 'audit_report.md'),
                  audit_markdown(rep, title='Audit before processing - %s'
                                 % os.path.basename(os.path.normpath(a.indir))))
    _say('')
    _say(audit_markdown(rep))
    _say('')
    _say('processing ...')
    out_rows = cmd_process(a, plan_rows=rows)
    _say('done: %d files -> %s' % (len(out_rows), a.outdir))


# --------------------------------------------------------------------------- ab
def cmd_ab(a):
    import csv
    import soundfile as sf
    from .loudness import lufs

    rep = os.path.join(a.outdir, '_report.csv')
    if not os.path.exists(rep):
        _say('!! run "process" first (%s missing)' % rep)
        sys.exit(2)
    rows = list(csv.DictReader(open(rep, encoding='utf-8')))
    out_map, in_map = {}, {}
    for q in glob.glob(os.path.join(a.outdir, '**', '*.wav'), recursive=True):
        out_map.setdefault(os.path.basename(q), q)
    for q in glob.glob(os.path.join(a.indir, '**', '*.wav'), recursive=True):
        in_map.setdefault(os.path.basename(q), q)

    # "worst" clips = how far the ORIGINAL sat from where its register says it should
    cand = []
    for r in rows:
        if r['file'] not in in_map or r['file'] not in out_map:
            continue
        try:
            dev = float(r['lufs_before']) - float(r['want'])
        except (TypeError, ValueError):
            continue
        if float(r['dur']) < a.min_dur:
            continue
        cand.append((abs(dev), r, dev))
    if not cand:
        _say('!! no comparable files')
        sys.exit(2)
    cand.sort(key=lambda t: -t[0])
    picked, seen = [], {}
    for _, r, dev in cand:
        if seen.get(r['group'], 0) >= 2:
            continue
        seen[r['group']] = seen.get(r['group'], 0) + 1
        r['_dev'] = dev
        picked.append(r)
        if len(picked) >= a.n:
            break
    _say('picked %d clips furthest from the design pitch line:' % len(picked))
    for r in picked:
        _say('  %-30s pitch %5.1f   was %7.2f -> target %7.2f   dev %+6.2f'
             % (r['file'][:28], float(r['pbar'] or 0), float(r['lufs_before']),
                float(r['want']), r['_dev']))

    tracks = {'before': [], 'after': []}
    fs_out = None
    for r in picked:
        for tag, p in (('before', in_map[r['file']]), ('after', out_map[r['file']])):
            x, fs = sf.read(p, dtype='float64', always_2d=True)
            fs_out = fs
            s = x.mean(axis=1)
            f = int(0.03 * fs)
            w = np.ones(len(s))
            w[:f] = np.linspace(0, 1, f)
            w[-f:] = np.linspace(1, 0, f)
            tracks[tag].append(s * w)

    os.makedirs(a.outdir, exist_ok=True)
    gap = np.zeros(int(0.35 * fs_out))
    per = {}
    for tag, segs in tracks.items():
        out = []
        for s in segs:
            out.extend([s, gap])
        y = np.concatenate(out)
        y *= 10 ** ((-20.0 - lufs(y, fs_out)) / 20.0)
        pk = np.max(np.abs(y))
        if pk > 0.98:
            y = y / pk * 0.98
        fn = os.path.join(a.outdir, 'AB_%s.wav' % tag)
        sf.write(fn, y, fs_out, subtype='PCM_24')
        per[tag] = np.array([lufs(s, fs_out) for s in segs])
        _say('  %-44s clip-to-clip sd %.2f LU  span %.2f LU'
             % (fn, per[tag].std(), per[tag].max() - per[tag].min()))

    # interleaved: before / after of the same clip back to back
    out = []
    for b, af in zip(tracks['before'], tracks['after']):
        out.extend([b, gap, af, gap])
    y = np.concatenate(out)
    y *= 10 ** ((-20.0 - lufs(y, fs_out)) / 20.0)
    fn = os.path.join(a.outdir, 'AB_interleaved.wav')
    sf.write(fn, y, fs_out, subtype='PCM_24')
    _say('  %-44s odd = before, even = after' % fn)


# ----------------------------------------------------------------------- verify
def _auto_glob(root):
    for cand in AUTO_GLOBS:
        if glob.glob(os.path.join(root, cand), recursive='**' in cand):
            return cand
    return '*.wav'


def cmd_verify(a):
    pat = a.glob or (PRESETS[a.preset]['glob'] if a.preset in PRESETS else _auto_glob(a.indir))
    rows_in, _ = discover(a.indir, pat)
    out_pat = a.glob or (_auto_glob(a.outdir) if not pat.startswith('**') else pat)
    rows_out, _ = discover(a.outdir, out_pat)
    skip = lambda p: os.path.basename(p).startswith(('AB_', '_', 'audit_'))
    rows_in = [p for p in rows_in if not skip(p)]
    rows_out = [p for p in rows_out if not skip(p)]
    if not rows_out:
        _say('!! no wav files in %s' % a.outdir)
        sys.exit(2)
    _say('analysing %d + %d files ...' % (len(rows_in), len(rows_out)))
    ri = load_features(rows_in, jobs=a.jobs, progress=_progress)
    ro = load_features(rows_out, jobs=a.jobs, progress=_progress)

    def stat(rows):
        by = {}
        for r in rows:
            by.setdefault(r['group'], []).append(r)
        sl, res, bseg = [], [], []
        for g, items in by.items():
            P = [x['pbar'] for x in items if x['pbar'] is not None]
            Y = [x['lufs'] for x in items if x['pbar'] is not None]
            D = [x['dur'] for x in items if x['pbar'] is not None]
            if len(P) >= 3 and np.ptp(P) >= 2.0:
                W = np.array(D)
                W = W / W.sum()
                fit = np.polyfit(np.array(P), np.array(Y), 1, w=np.sqrt(W))
                sl.append(fit[0] * 12)
                res.append(float(np.std(np.array(Y) - np.polyval(fit, np.array(P)))))
            bseg += [x['bseg'] * 12 for x in items if x['bseg'] is not None]
        L = np.array([r['lufs'] for r in rows])
        return dict(slope=np.array(sl), res=np.array(res), bseg=np.array(bseg),
                    lufs_sd=float(np.std(L)), n=len(rows), groups=len(by))

    i, o = stat(ri), stat(ro)
    _say('')
    _say('%-10s %6s %12s %8s %8s %12s %9s'
         % ('', 'n', 'inter-slope', 'sd', 'resid', 'intra-slope', 'LUFS sd'))
    for nm, s in (('before', i), ('after', o)):
        _say('%-10s %6d %+12.2f %8.2f %8.2f %12.2f %9.2f'
             % (nm, s['n'],
                np.median(s['slope']) if len(s['slope']) else float('nan'),
                np.std(s['slope']) if len(s['slope']) else float('nan'),
                np.median(s['res']) if len(s['res']) else float('nan'),
                np.median(s['bseg']) if len(s['bseg']) else float('nan'),
                s['lufs_sd']))
    if not len(i['slope']):
        _say('')
        _say('note: the inter-clip columns need sources with >=3 clips that span '
             '>=2 semitones; this dataset has no such source, so they are undefined.')


# -------------------------------------------------------------------------- cli
def build_parser():
    p = argparse.ArgumentParser(prog='loudness_process', description=BANNER)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('indir', help='input dataset directory')
        sp.add_argument('outdir', nargs='?', default=None,
                        help='output directory (optional for audit)')
        sp.add_argument('--glob', default=None,
                        help="audio pattern, e.g. 'wavs/*.wav' or '**/wavs/*.wav'")
        sp.add_argument('--preset', default='auto', choices=['auto'] + list(PRESETS))
        sp.add_argument('--target', type=float, default=-24.0,
                        help='loudness target in LKFS')
        sp.add_argument('--tol', type=float, default=2.0,
                        help='loudness tolerance window in LU (also the saturation point)')
        sp.add_argument('--tp-max', type=float, default=-2.0,
                        help='true-peak ceiling in dBTP')
        sp.add_argument('--b-tilt-oct', type=float, default=6.0,
                        help='intra-clip design slope, dB/oct')
        sp.add_argument('--b-seg-oct', type=float, default=None,
                        help='inter-clip design slope (default: same as --b-tilt-oct)')
        sp.add_argument('--lam', type=float, default=0.6,
                        help='partial-correction factor for the slope')
        sp.add_argument('--dc-scope', default='pitch', choices=['pitch', 'constant'])
        sp.add_argument('--level-scope', default='corpus', choices=['corpus', 'none'])
        sp.add_argument('--no-limiter', action='store_true')
        sp.add_argument('--lookahead', type=float, default=0.012)
        sp.add_argument('--release', type=float, default=0.200)
        sp.add_argument('--keep-tree', type=int, default=1,
                        help='keep the input directory layout')
        sp.add_argument('--copy-sidecar', action='store_true',
                        help='copy annotation files next to the output audio')
        sp.add_argument('--sidecar-glob', default='*.csv',
                        help='pattern for --copy-sidecar')
        sp.add_argument('--jobs', type=int, default=10)
        sp.add_argument('--limit', type=int, default=0,
                        help='process only the first N files (debug)')

    common(sub.add_parser('audit', help='report only, no audio written'))
    common(sub.add_parser('process', help='normalise and write the output dataset'))
    common(sub.add_parser('run', help='audit + process'))
    common(sub.add_parser('verify', help='before/after dynamic metrics'))

    ab = sub.add_parser('ab', help='build a listening comparison')
    ab.add_argument('indir')
    ab.add_argument('outdir')
    ab.add_argument('--n', type=int, default=10)
    ab.add_argument('--min-dur', type=float, default=2.5)
    return p


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    a = build_parser().parse_args(argv)
    _say(BANNER)
    if a.cmd in ('process', 'run') and not a.outdir:
        _say('!! %s needs an output directory' % a.cmd)
        sys.exit(2)
    if a.cmd == 'audit':
        cmd_audit(a)
    elif a.cmd == 'process':
        cmd_process(a)
    elif a.cmd == 'run':
        cmd_run(a)
    elif a.cmd == 'verify':
        cmd_verify(a)
    elif a.cmd == 'ab':
        cmd_ab(a)


if __name__ == '__main__':
    main()
