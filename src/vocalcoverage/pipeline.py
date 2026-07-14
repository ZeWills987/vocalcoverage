"""Vocal coverage analysis pipeline.

Given a full mix and its already-separated vocal stem, estimates how much
of the track contains audible vocals, frame by frame and as segments.

Pipeline stages:
    1. load_aligned      - decode mix + vocals, verify matching duration
    2. frame_signal       - split into non-overlapping frames
    3. is_silent_frame    - exclude frames below an RMS silence threshold
    4. frame_ratio        - RMS(vocals) / RMS(mix) per frame
    5. frame_f0           - pyin-based harmonic confirmation per frame
    6. is_vocal_frame     - ratio + f0 confidence decision
    7. smooth_frames      - majority-vote temporal smoothing
    8. frames_to_segments - group consecutive vocal frames into segments
"""

from __future__ import annotations

import numpy as np

from .decode import decode_audio

# Frames shorter than this fraction of frame_duration are dropped instead of padded.
_MIN_PARTIAL_FRAME_FRACTION = 0.5

# pyin needs enough samples for its internal analysis window; frames shorter
# than this are padded (not resized) before being passed to pyin so short
# frame_duration values still produce a usable estimate.
_PYIN_MIN_FRAME_SECONDS = 0.25


def load_aligned(
    mix_path: str, vocals_path: str, sr: int = 22050, tolerance_seconds: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a mix and its vocal stem, verifying they cover the same duration."""
    mix = decode_audio(mix_path, sr=sr)
    vocals = decode_audio(vocals_path, sr=sr)

    mix_duration = len(mix) / sr
    vocals_duration = len(vocals) / sr
    if abs(mix_duration - vocals_duration) > tolerance_seconds:
        raise ValueError(
            f"Mix and vocal stem durations differ by more than {tolerance_seconds}s "
            f"(mix={mix_duration:.3f}s, vocals={vocals_duration:.3f}s). "
            "This usually means the wrong stem was provided or the files are not aligned."
        )

    min_len = min(len(mix), len(vocals))
    return mix[:min_len], vocals[:min_len]


def frame_signal(signal: np.ndarray, sr: int, frame_duration: float = 1.0) -> np.ndarray:
    """Split a signal into non-overlapping frames of frame_duration seconds.

    A trailing partial frame shorter than half a frame's duration is
    dropped; otherwise it is zero-padded to a full frame.
    """
    frame_len = int(round(frame_duration * sr))
    if frame_len <= 0:
        raise ValueError("frame_duration must be positive")

    n_frames = len(signal) // frame_len
    remainder = len(signal) - n_frames * frame_len

    frames = []
    if n_frames > 0:
        frames.append(signal[: n_frames * frame_len].reshape(n_frames, frame_len))

    if remainder > 0 and remainder >= frame_len * _MIN_PARTIAL_FRAME_FRACTION:
        tail = signal[n_frames * frame_len :]
        padded_tail = np.pad(tail, (0, frame_len - remainder))
        frames.append(padded_tail.reshape(1, frame_len))

    if not frames:
        return np.empty((0, frame_len), dtype=signal.dtype)

    return np.concatenate(frames, axis=0)


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0


def is_silent_frame(mix_frame: np.ndarray, threshold_db: float = -40.0) -> bool:
    """Return True if a frame's RMS level (in dB) is below threshold_db."""
    rms = _rms(mix_frame)
    if rms <= 0.0:
        return True
    rms_db = 20.0 * np.log10(rms)
    return bool(rms_db < threshold_db)


def frame_ratio(vocals_frame: np.ndarray, mix_frame: np.ndarray) -> float:
    """RMS(vocals_frame) / RMS(mix_frame), clamped to [0.0, 1.0]."""
    mix_rms = _rms(mix_frame)
    if mix_rms <= 0.0:
        return 0.0
    ratio = _rms(vocals_frame) / mix_rms
    return float(np.clip(ratio, 0.0, 1.0))


def frame_f0(
    vocals_frame: np.ndarray, sr: int, f0_min: float = 80.0, f0_max: float = 1000.0
) -> dict:
    """Estimate f0 presence and voicing confidence for a frame via librosa.pyin."""
    import librosa

    frame = vocals_frame
    min_samples = int(_PYIN_MIN_FRAME_SECONDS * sr)
    if len(frame) < min_samples:
        frame = np.pad(frame, (0, min_samples - len(frame)))

    frame_length = min(2048, len(frame))
    hop_length = max(frame_length // 4, 1)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        frame,
        fmin=f0_min,
        fmax=f0_max,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # The 75th percentile (rather than the mean) of pyin's per-hop voicing
    # probability avoids diluting confidence with the unvoiced sub-hops that
    # normally occur within a singing frame (note attacks, consonants,
    # breaths): a mean averages those in, a lead-in silence in an otherwise
    # clearly sung frame can pull it under threshold, while noise/inharmonic
    # leakage has no such high-confidence sub-hops to raise the percentile.
    confidence = float(np.nanpercentile(voiced_prob, 75)) if voiced_prob.size else 0.0
    if np.isnan(confidence):
        confidence = 0.0

    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

    if voiced_f0.size > 0:
        return {
            "f0_detected": True,
            "f0_hz": float(np.mean(voiced_f0)),
            "confidence": confidence,
        }
    return {"f0_detected": False, "f0_hz": None, "confidence": confidence}


def is_vocal_frame(
    ratio: float,
    f0_result: dict,
    ratio_threshold: float = 0.1,
    f0_confidence_threshold: float = 0.2,
) -> bool:
    """Decide vocal presence from RMS ratio and harmonic confidence.

    Requiring both conditions guards against a "lead instrument" false
    positive: a synth/sax/guitar leaking into the vocal stem often produces
    a high RMS ratio but lacks the stable harmonic structure pyin expects
    in the vocal f0 range.

    Default thresholds are calibrated on real audio: pyin's voicing
    confidence sits well below 1.0 on real singing (vibrato and
    micro-variations spread probability across pitch candidates), while
    broadband/inharmonic leakage measures near 0.01 — so 0.2 keeps high
    recall on voice with a wide margin against lead-instrument leakage.
    """
    return ratio >= ratio_threshold and f0_result["confidence"] >= f0_confidence_threshold


def smooth_frames(frame_flags: list[bool], window: int = 3) -> list[bool]:
    """Majority-vote smoothing over a centered sliding window."""
    n = len(frame_flags)
    if n == 0 or window <= 1:
        return list(frame_flags)

    half = window // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_slice = frame_flags[lo:hi]
        votes = sum(1 for flag in window_slice if flag)
        smoothed.append(votes * 2 > len(window_slice))
    return smoothed


def frames_to_segments(
    smoothed_flags: list[bool],
    frame_duration: float,
    ratios: list[float] | None = None,
    f0_confidences: list[float] | None = None,
) -> list[dict]:
    """Group consecutive vocal frames into segments with mean ratio/confidence."""
    segments = []
    n = len(smoothed_flags)
    i = 0
    while i < n:
        if not smoothed_flags[i]:
            i += 1
            continue
        start = i
        while i < n and smoothed_flags[i]:
            i += 1
        end = i

        seg_ratios = ratios[start:end] if ratios is not None else []
        seg_f0 = f0_confidences[start:end] if f0_confidences is not None else []

        segments.append(
            {
                "start": start * frame_duration,
                "end": end * frame_duration,
                "ratio_mean": float(np.mean(seg_ratios)) if seg_ratios else None,
                "f0_confidence": float(np.mean(seg_f0)) if seg_f0 else None,
            }
        )
    return segments


def analyze(
    mix_path: str,
    vocals_path: str,
    frame_duration: float = 1.0,
    f0_min: float = 80.0,
    f0_max: float = 1000.0,
    silence_threshold_db: float = -40.0,
    ratio_threshold: float = 0.1,
    f0_confidence_threshold: float = 0.2,
    smoothing_window: int = 3,
    sr: int = 22050,
) -> dict:
    """Run the full vocal coverage pipeline on a mix and its vocal stem."""
    mix, vocals = load_aligned(mix_path, vocals_path, sr=sr)

    mix_frames = frame_signal(mix, sr, frame_duration)
    vocals_frames = frame_signal(vocals, sr, frame_duration)
    n_frames = min(len(mix_frames), len(vocals_frames))

    silent_flags = []
    ratios = []
    f0_results = []
    raw_vocal_flags = []

    for i in range(n_frames):
        mix_frame = mix_frames[i]
        vocals_frame = vocals_frames[i]

        silent = is_silent_frame(mix_frame, threshold_db=silence_threshold_db)
        silent_flags.append(silent)

        ratio = frame_ratio(vocals_frame, mix_frame)
        f0_result = frame_f0(vocals_frame, sr, f0_min=f0_min, f0_max=f0_max)

        ratios.append(ratio)
        f0_results.append(f0_result)
        raw_vocal_flags.append(
            not silent
            and is_vocal_frame(ratio, f0_result, ratio_threshold, f0_confidence_threshold)
        )

    smoothed_flags = smooth_frames(raw_vocal_flags, window=smoothing_window)
    # Silence always wins over smoothing: a silent frame cannot become "vocal".
    smoothed_flags = [
        flag and not silent for flag, silent in zip(smoothed_flags, silent_flags)
    ]

    segments = frames_to_segments(
        smoothed_flags,
        frame_duration,
        ratios=ratios,
        f0_confidences=[r["confidence"] for r in f0_results],
    )

    frame_scores = [
        {
            "time": i * frame_duration,
            "ratio": ratios[i],
            "f0_detected": f0_results[i]["f0_detected"],
            "f0_hz": f0_results[i]["f0_hz"],
            "is_vocal_frame": smoothed_flags[i],
        }
        for i in range(n_frames)
    ]

    silent_frame_count = sum(silent_flags)
    non_silent_count = n_frames - silent_frame_count
    vocal_frame_count = sum(1 for flag in smoothed_flags if flag)
    vocal_coverage = (vocal_frame_count / non_silent_count) if non_silent_count > 0 else 0.0

    return {
        "track": vocals_path,
        "vocal_coverage": vocal_coverage,
        "frame_count": n_frames,
        "silent_frame_count": silent_frame_count,
        "segments": segments,
        "frame_scores": frame_scores,
    }
