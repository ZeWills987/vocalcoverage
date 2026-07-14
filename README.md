# vocalcoverage

A lightweight, open-source (MIT) Python library that measures the presence
of vocals in a track, given an already-separated vocal stem and the
original full mix.

Use cases: vocal/instrumental detection, karaoke-section detection, or as
an input feature for downstream classifiers. `vocalcoverage` is agnostic
of any application context — it only does audio analysis.

## Installation

```bash
pip install vocalcoverage
```

`vocalcoverage` also requires **FFmpeg** as a system dependency, used to
decode audio files:

- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows: `winget install ffmpeg` (or download from https://ffmpeg.org/download.html)

If FFmpeg is not found on your `PATH`, `vocalcoverage` raises a clear error
explaining how to install it.

## Quick usage

`vocalcoverage` does not perform source separation itself — you must
provide a vocal stem already separated from the mix, using a tool such as
[Demucs](https://github.com/facebookresearch/demucs),
[Spleeter](https://github.com/deezer/spleeter), or
[audio-separator](https://github.com/nomadkaraoke/python-audio-separator).

```python
from vocalcoverage import analyze

result = analyze("mix.wav", "vocals.wav")

print(result["vocal_coverage"])  # e.g. 0.72
for segment in result["segments"]:
    print(segment["start"], segment["end"], segment["ratio_mean"])
```

## CLI

```bash
vocalcoverage analyze mix.wav vocals.wav --json
vocalcoverage analyze mix.wav vocals.wav --frame-duration 0.5
```

## How the pipeline works

1. **Load & align** — decode the mix and vocal stem, verify they share the
   same duration.
2. **Frame the signal** — split into non-overlapping frames (default 1s).
3. **Detect silence** — frames below an RMS threshold are excluded from
   coverage.
4. **RMS ratio** — `RMS(vocals) / RMS(mix)` per frame.
5. **Harmonic confirmation** — `librosa.pyin` estimates f0 and voicing
   confidence within the vocal range per frame.
6. **Per-frame decision** — a frame is "vocal" only if both the RMS ratio
   and the f0 confidence clear their thresholds (defaults:
   `ratio_threshold=0.1`, `f0_confidence_threshold=0.2`).
7. **Temporal smoothing** — a centered majority-vote window removes
   frame-to-frame flicker.
8. **Segmentation** — consecutive vocal frames are grouped into segments
   with their mean ratio and f0 confidence.

`vocal_coverage` = number of vocal frames (after smoothing) / number of
non-silent frames.

All decision thresholds (`ratio_threshold`, `f0_confidence_threshold`,
`silence_threshold_db`, etc.) are parameters with documented defaults —
`vocalcoverage` never hardcodes an application-level "coverage < X% means
instrumental" judgment.

### A note on the default `f0_confidence_threshold` (0.2)

pyin's voicing confidence is structurally lower on real singing than on
synthetic test tones. A pure, stable sine yields confidence close to 1.0,
but real voice — with vibrato, micro-variations in pitch, breathiness and
consonants — spreads pyin's probability mass across neighboring pitch
candidates, and per-frame confidence commonly lands well below 0.5 even
when a fundamental is clearly detected in the vocal range. An earlier
default of 0.5 caused severe under-detection on real a cappella material
(~18% measured coverage where ~95–100% was expected, despite RMS ratios
near 1.0 and valid f0 detections).

The recalibrated default of 0.2 restores recall on real voice while
keeping the lead-instrument guard intact: broadband/inharmonic leakage
(noise, unpitched synth textures) measures around 0.01 confidence in our
tests — an order of magnitude below the threshold. If your material is
mostly clean synthetic vocals you can raise the threshold; if you see
missed detections on heavily ornamented singing, lower it further.

### Confidence aggregation: percentile, not mean

`frame_f0` summarizes pyin's per-hop voicing probability across a frame
using the 75th percentile, not the mean. A 1-second singing frame contains
many pyin sub-hops (attacks, consonants, breaths, brief pauses between
notes) that are legitimately unvoiced even while the singer is audibly
present. Averaging those in dilutes the frame's confidence far below what
a listener would call "clearly sung" — on real a cappella material this
alone caused adjacent, equally loud, equally pitched frames to swing from
comfortably passing (mean confidence 0.39) to rejected (mean confidence
0.16-0.20), purely because one had a slightly longer unvoiced attack.
The 75th percentile reflects "is there a clearly voiced portion in this
frame" instead of "is the whole frame voiced," while leaving noise/lead
instrument leakage untouched — those have no high-confidence sub-hops to
raise, so percentile and mean agree (~0.01) on non-vocal content.

## How it works: in-memory decoding

`vocalcoverage` never writes converted audio files — FFmpeg decoding
happens in-memory only, for analysis purposes. Audio is decoded to raw PCM
via an FFmpeg subprocess and read directly from its stdout; no intermediate
`.wav` or other file ever touches disk.

## Limitations

- **Garbage in, garbage out**: results depend entirely on the quality of
  the vocal stem separation provided as input. `vocalcoverage` does not
  perform or validate source separation itself.
- **Lead instruments with vocal-like timbre** (saxophone, lead guitar, lead
  synth) can produce a high RMS ratio if they leak into the vocal stem. The
  f0 harmonic confirmation reduces but does not eliminate this risk — this
  is the main known limitation.
- **Heavily processed voices** (vocoder effects, extreme pitch shifting)
  may fall outside the default `f0_min`/`f0_max` range and go undetected.

## License

MIT

---

Built to power vocal detection at Mkzik.
