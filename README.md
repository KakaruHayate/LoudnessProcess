# LoudnessProcess

**English** | [简体中文](README.zh-CN.md)

**Loudness and dynamics conditioning for singing-voice / speech datasets.**

Takes a dataset of clips that were recorded or processed inconsistently, and rewrites it so that

- every clip's overall loudness sits on the same measured target (ITU-R BS.1770), and
- the loudness *inside* each clip still follows the music (and follows the pitch, which is the
  only thing a pitch-conditioned acoustic model can actually learn).

No compressor, no EQ. The only level-shaping operations are a constant gain per clip, a smooth
gain that follows pitch, and — where required — a true-peak limiter.

```
audit  ->  process  ->  verify  ->  ab
```

---

## What problem it solves

Clip-level loudness in a dataset is almost never consistent. Clips come from different takes,
sessions, songs and processing chains, so the same vowel at the same pitch can sit several dB
apart for no musical reason. A spectrogram-based model that predicts an absolute log-mel target
has no way to know which one was intended, so its output loudness varies from render to render.

The obvious fixes each break something:

| approach | what it fixes | what it breaks |
|---|---|---|
| loudness matching (constant gain per clip) | the overall level | nothing — but it cannot touch the *slope* of level against pitch, which is where most of the inconsistency actually lives |
| AGC / loudness levelers | level *and* the slope | the slope gets flattened, so the model renders dynamics that feel dull |
| manual editing | both | slow, and the result depends on who did it |

The reason the second column is so limited: a clip's level is not one number. It decomposes as

```
L(t) = a  +  B·p(t)  +  slow(t)  +  fast(t)
       ↑     ↑          ↑           ↑
     offset  pitch    phrase      syllable
```

- `a` — one constant per clip. **The model cannot see it.** This is what makes renders jump.
- `B·p(t)` — level follows pitch. Pitch *is* an input, so this part is **learnable and controllable**.
- `slow(t)`, `fast(t)` — the actual singing. Must be preserved.

A constant gain can only change `a`. It cannot change `B`. And `B` is wildly inconsistent in
real data (measured slopes anywhere from −3 to +17 dB/oct within a single collection), so a model
trained on it learns an average of contradictory behaviours. That is what "unstable dynamics" means
in practice.

This tool changes `a` **and** `B`, and leaves `slow`/`fast` untouched by construction.

---

## How it works

### The gain applied to every sample

```
gain(t) = DC + (B_tilt − B_source) · (p(t) − p̄)
```

| symbol | meaning |
|---|---|
| `B_tilt` | the slope the model should render *inside* a clip (default **+6 dB/oct**) |
| `B_source` | the slope actually measured for that source, pooled over all of its clips |
| `p(t)`, `p̄` | per-frame pitch and the clip's energy-weighted mean pitch |
| `DC` | where the clip sits on the loudness axis (see below) |

`B_tilt` calibration: raw singing typically measures **+6.8 … +8.5 dB/oct**; a good commercial
render sits near **+6**; an aggressively levelled recording drops to **+2**. Below +3 the result
starts to sound flat, so `+6` is a safe default.

### Where a clip sits (`DC`)

```
DC = target + δ + design_offset(p̄ − p̄_ref)
```

`design_offset` is linear (`+B_seg` dB per semitone) and, when a tolerance window `tol` is given,
**soft-saturates** at `±tol`:

```
design_offset(Δp) = tol · tanh( B_seg · Δp / tol )
```

- inside the window it is exactly the linear design slope;
- at extreme registers it flattens smoothly onto `target ± tol`, so the whole dataset can be made
  to respect a hard tolerance;
- it stays a **monotone function of pitch**, so the model can still learn it — it never becomes
  random.

`δ` is a single number for the whole dataset, chosen so that the duration-weighted corpus loudness
lands exactly on the target. Set `--tol 0` to disable the window and keep the full inter-clip
pitch dynamics (at the cost of some clips falling outside the tolerance).

### Limits of the method

A tolerance window physically cannot hold a large inter-clip pitch relation: `±2 LU` of room
against `1.5 octaves × 6 dB/oct` (≈ `±5 LU`) is a contradiction. Something has to give, and the
tool makes the trade explicit and measurable rather than silently picking one.

---

## Install

```bash
pip install -r requirements.txt
```

Python 3.9+. Requires `numpy`, `scipy`, `soundfile`.

---

## Usage

### 1 — audit (no audio is written)

```bash
python -m loudness_process audit path/to/dataset
```

Writes `audit_report.md` + `audit_files.csv` into `dataset/_loudness_audit/`. This is the step that
tells you *whether* the data needs work and *which clips* are the problem.

### 2 — process (writes the output dataset)

```bash
python -m loudness_process process path/to/dataset path/to/output \
    --copy-sidecar --jobs 10
```

The output mirrors the input layout (`--keep-tree`, on by default), so it is a drop-in replacement.
`--copy-sidecar` copies annotation files (`*.csv` by default, see `--sidecar-glob`) alongside.

Nested layouts are auto-detected; force a pattern if needed:

```bash
python -m loudness_process process in out --glob "**/wavs/*.wav"
```

One file per source (whole takes, no clips):

```bash
python -m loudness_process process in out --preset stems --glob "*.wav"
```

### 3 — verify (before / after metrics)

```bash
python -m loudness_process verify path/to/dataset path/to/output
```

### 4 — ab (listening comparison)

```bash
python -m loudness_process ab path/to/dataset path/to/output --n 10
```

Picks the clips that deviate most from the design pitch line and writes three level-matched files:
`AB_before.wav`, `AB_after.wav`, and `AB_interleaved.wav` (odd clips = before, even = after, so the
same phrase is heard twice in a row).

---

## Parameters

| option | default | meaning |
|---|---|---|
| `--target` | `-24.0` | loudness target, LKFS (`-24` = ATSC A/85; use `-23` for EBU R128) |
| `--tol` | `2.0` | tolerance window ±LU, also the saturation point (`0` = off) |
| `--tp-max` | `-2.0` | true-peak ceiling, dBTP |
| `--b-tilt-oct` | `6.0` | intra-clip design slope, dB/oct |
| `--b-seg-oct` | same as above | inter-clip design slope, dB/oct |
| `--lam` | `0.6` | partial-correction factor (1.0 = pull every source fully onto the target) |
| `--dc-scope` | `pitch` | `pitch` = level follows the design line; `constant` = one level for all clips |
| `--level-scope` | `corpus` | `corpus` = align the dataset; `none` = shape only, no level change |
| `--no-limiter` | off | disable the true-peak limiter |
| `--lookahead`, `--release` | `0.012`, `0.200` | limiter time constants, seconds |
| `--preset` | `auto` | `sliced` (clips share a name prefix) / `stems` (one file per source) / `auto` |
| `--group-by` | `name` | `name` pools clips with a shared name prefix across folders; `dir` also pins the folder, when the same name means a different recording in each folder |
| `--glob` | preset | audio pattern; `**` enables recursive matching |
| `--keep-tree` | `1` | keep the input directory layout |
| `--copy-sidecar`, `--sidecar-glob` | off, `*.csv` | copy annotation files to the output |
| `--out-sr` | `0` (keep) | resample the output |
| `--out-bits` | `0` (keep) | `16` / `24` / `32`; a TPDF dither is applied when writing 16 bit (`--no-dither` disables) |
| `--out-channels` | `0` (keep) | `1` downmixes to mono by *averaging*, so a dual-mono file does not gain 3.01 LU |
| `--jobs` | `10` | worker processes |
| `--limit` | `0` | only the first N files (debugging) |

### Picking parameters

- **`--target`** — match whatever the rest of your corpus already uses, so mixing datasets does not
  shift the level.
- **`--b-tilt-oct`** — `+6` is the safe default. Lower flattens, higher exaggerates.
- **`--lam`** — `0.6` keeps some of each source's own character; `0.8` is more aggressive.
  Pick it from what the audit shows: a source already near `+6 dB/oct` needs nothing (`0.6`),
  a levelled or heavily compressed source (`+1 … +3`) needs `0.8 … 1.0`.
- **`--tol`** — set it to your delivery tolerance if you have one; `0` if you would rather keep the
  full inter-clip dynamics and accept the spread.

---

## Output

```
<out>/
  ...audio, same layout and same filenames as the input...
  _report.csv            per clip: lufs_before, want, lufs_after, tp_after, ok, gain_db,
                         slope_seg_oct, source_slope_oct, note
  _groups.csv            per source: slope in / out
  _report.md             human-readable summary
  audit_report.md        (run/audit) the state before processing
  audit_files_before.csv
  AB_*.wav               (ab) listening comparison
```

The **`want`** column is the intended level of every clip — the design pitch line. If your acoustic
model accepts an explicit energy/dynamics channel, feeding it this column restores controllable
dynamics without leaving the loudness spec.

### Making every dataset byte-identical in format

By default the source sample rate, bit depth and channel count are preserved, so a corpus assembled
from mixed material stays mixed. Export a uniform one in a single pass:

```bash
python -m loudness_process process in out --out-sr 44100 --out-bits 16 --out-channels 1
```

Conversion happens *before* analysis, so the level and peak targets are measured on the signal that
is actually written — which matters because a stereo downmix changes loudness by up to 3 dB.

---

## Reading the audit report

Four numbers matter:

| metric | good | bad | meaning |
|---|---|---|---|
| **intra-clip slope** (median / sd) | median ≈ +6, sd ≤ 1.5 | sd > 3 | the dynamic shape a single render can show; a large sd means clips disagree with each other |
| **inter-clip slope** (per source) | near the target, sd ≤ 2 | range −5 … +17 dB/oct | how much louder higher notes get; negative values are a limiter fingerprint |
| **inter-clip residual** | < 1.0 dB | > 2 dB | level not explained by pitch — **this is what makes renders jump** |
| **share within ±tol** | 100 % | < 90 % | whether the dataset meets the delivery spec |

The report also lists the clips furthest from the design pitch line. Large deviations
(> 4 dB) are usually a wrong take, a mis-cut clip, or an inconsistent recording gain, and are worth
a manual look.

---

## Measurement conventions

- **Integrated loudness**: ITU-R BS.1770 — K-weighting (coefficients recomputed for the actual
  sample rate), 400 ms blocks with 75 % overlap, −70 LKFS absolute gate and −10 LU relative gate.
  Verified against `ffmpeg`'s `ebur128` to within ±0.05 LU.
- BS.1770-3 and -4 give identical integrated loudness; they differ only in true-peak detail.
- **Everything is analysed as mono.** A dual-mono file measured as stereo reads **3.01 LU louder**
  than the same file measured as mono — the most common way to get an inconsistent corpus.
- **True peak**: 4× oversampled peak detection.
- **Short clips**: trimming 10 % off each end moves the integrated value by about 0.12 LU (median)
  or 0.29 LU (p90), so tolerances tighter than ±0.5 LU are not meaningful for clips of a few seconds.

---

## Known limitations

1. **The tolerance window compresses inter-clip pitch dynamics.** `±2 LU` cannot hold
   `1.5 octaves × 6 dB/oct`. Use `--tol 0` to keep them and accept the spread.
2. **Heavily limited source material cannot be restored.** The tool can rebuild the macro
   pitch-to-loudness relation, but dynamics already destroyed by a limiter are gone. Such data
   shows up in the audit as a large intra-clip slope sd and an inter-clip slope near or below zero.
3. **Slopes are estimated per source, not per clip.** A 5–15 s clip has too little pitch range for a
   reliable fit (observed scatter reaches 7 dB/oct), so clips must be pooled by source. Naming them
   with a shared prefix (`--preset sliced`) is what makes this work.
4. **`--dc-scope pitch` needs usable pitch.** Whispered or fully unvoiced clips fall back to the
   dataset reference pitch.

---

## Layout

```
loudness_process/
  __init__.py
  __main__.py      python -m loudness_process
  cli.py           subcommands and options
  loudness.py      BS.1770 K-weighting / LUFS / LRA / true peak / limiter
  pitch.py         YIN pitch tracker
  dataset.py       discovery, grouping, per-frame features, slopes, audit
  normalize.py     the normalisation chain
  report.py        CSV / Markdown writers
```

## License

MIT — see [LICENSE](LICENSE).
