from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from vocalcoverage.pipeline import (
    analyze,
    frame_f0,
    frame_ratio,
    frame_signal,
    frames_to_segments,
    is_silent_frame,
    is_vocal_frame,
    load_aligned,
    smooth_frames,
)
from conftest import SR, sine_wave, silence, vibrato_tone, white_noise


def test_frame_signal_drops_short_tail():
    signal = np.zeros(SR * 2 + int(0.2 * SR), dtype=np.float32)  # 20% tail
    frames = frame_signal(signal, SR, frame_duration=1.0)
    assert len(frames) == 2


def test_frame_signal_pads_long_tail():
    signal = np.zeros(SR * 2 + int(0.6 * SR), dtype=np.float32)  # 60% tail
    frames = frame_signal(signal, SR, frame_duration=1.0)
    assert len(frames) == 3
    assert len(frames[-1]) == SR


def test_is_silent_frame_true_for_silence():
    frame = silence(1.0)
    assert is_silent_frame(frame, threshold_db=-40.0) is True


def test_is_silent_frame_false_for_loud_signal():
    frame = sine_wave(220.0, 1.0, amplitude=0.8)
    assert is_silent_frame(frame, threshold_db=-40.0) is False


def test_frame_ratio_clamped_and_zero_division_safe():
    vocals = sine_wave(220.0, 1.0, amplitude=0.8)
    mix = silence(1.0)
    assert frame_ratio(vocals, mix) == 0.0

    mix_loud = sine_wave(220.0, 1.0, amplitude=0.1)
    vocals_loud = sine_wave(220.0, 1.0, amplitude=5.0)
    assert frame_ratio(vocals_loud, mix_loud) == 1.0


def test_frame_f0_detects_sine_in_vocal_range():
    frame = sine_wave(220.0, 1.0, amplitude=0.8)
    result = frame_f0(frame, SR, f0_min=80.0, f0_max=1000.0)
    assert result["f0_detected"] is True
    assert result["f0_hz"] == pytest.approx(220.0, rel=0.1)
    assert result["confidence"] > 0.0


def test_is_vocal_frame_true_for_sine():
    frame = sine_wave(220.0, 1.0, amplitude=0.8)
    ratio = 0.8
    f0_result = frame_f0(frame, SR)
    assert is_vocal_frame(ratio, f0_result) is True


def test_frame_f0_confidence_uses_percentile_not_mean():
    """Regression test for a real-world under-detection: a singing frame with
    a normal unvoiced attack/consonant portion was rejected because the mean
    of pyin's per-hop voicing probability was diluted below threshold, even
    though a clear majority of hops were confidently voiced.

    This fixes the exact voiced_prob pattern captured from a real a cappella
    track (t=7.0s in track_avec_voix, UVR-MDX-NET-Inst_HQ_3 stem): mean was
    0.157 (would fail the default 0.2 threshold) while p75 was 0.261 (passes,
    matching the audibly sung content in the source frame).
    """
    real_voiced_prob = np.array([
        0.08, 0.09, 0.11, 0.13, 0.11, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.02, 0.03, 0.03, 0.08, 0.09, 0.24, 0.37, 0.43, 0.43, 0.55, 0.43,
        0.21, 0.32, 0.43, 0.48, 0.48, 0.48, 0.32, 0.07, 0.04, 0.15, 0.21, 0.18,
        0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.15,
    ])
    real_voiced_flag = real_voiced_prob >= 0.5
    real_f0 = np.where(real_voiced_flag, 254.0, np.nan)

    assert np.nanmean(real_voiced_prob) < 0.2  # the bug: mean rejects this frame

    frame = sine_wave(254.0, 1.0, amplitude=0.5)
    with mock.patch(
        "librosa.pyin", return_value=(real_f0, real_voiced_flag, real_voiced_prob)
    ):
        result = frame_f0(frame, SR)

    assert result["confidence"] >= 0.2
    assert is_vocal_frame(ratio=0.466, f0_result=result) is True


def test_is_vocal_frame_true_for_vibrato_with_default_thresholds():
    """Real singing has vibrato/micro-variations that lower pyin's voicing
    confidence versus a pure stable tone; the default thresholds must still
    accept it (regression test for the 0.5 -> 0.2 recalibration)."""
    frame = vibrato_tone(220.0, 1.0, amplitude=0.8)
    ratio = 0.9
    f0_result = frame_f0(frame, SR)
    assert f0_result["f0_detected"] is True
    assert is_vocal_frame(ratio, f0_result) is True


def test_is_vocal_frame_false_for_white_noise_lead_instrument():
    """High RMS ratio but no stable harmonic structure -> not a vocal frame."""
    frame = white_noise(1.0, amplitude=0.8)
    ratio = 0.9
    f0_result = frame_f0(frame, SR)
    # Must hold at the recalibrated default f0_confidence_threshold (0.2):
    # broadband leakage measures ~0.01 confidence, well under the threshold.
    assert f0_result["confidence"] < 0.2
    assert is_vocal_frame(ratio, f0_result) is False


def test_smooth_frames_removes_isolated_blip():
    flags = [False, False, True, False, False]
    smoothed = smooth_frames(flags, window=3)
    assert smoothed[2] is False


def test_smooth_frames_keeps_majority_true():
    flags = [True, True, True, False, True]
    smoothed = smooth_frames(flags, window=3)
    assert smoothed[0] is True
    assert smoothed[1] is True


def test_frames_to_segments_groups_consecutive():
    flags = [False, True, True, False, True]
    segments = frames_to_segments(flags, frame_duration=1.0)
    assert len(segments) == 2
    assert segments[0]["start"] == 1.0
    assert segments[0]["end"] == 3.0
    assert segments[1]["start"] == 4.0
    assert segments[1]["end"] == 5.0


def test_load_aligned_raises_on_duration_mismatch(wav_writer):
    mix_path = wav_writer("mix.wav", sine_wave(220.0, 3.0))
    vocals_path = wav_writer("vocals.wav", sine_wave(220.0, 1.0))

    with pytest.raises(ValueError, match="durations differ"):
        load_aligned(mix_path, vocals_path, sr=SR)


def test_analyze_acapella_coverage_near_one(wav_writer):
    signal = sine_wave(220.0, 5.0, amplitude=0.7)
    mix_path = wav_writer("mix.wav", signal)
    vocals_path = wav_writer("vocals.wav", signal)

    result = analyze(mix_path, vocals_path, frame_duration=1.0)

    assert result["vocal_coverage"] > 0.8
    assert result["frame_count"] == 5
    assert result["track"] == vocals_path


def test_analyze_silent_vocals_coverage_near_zero(wav_writer):
    mix_signal = sine_wave(220.0, 5.0, amplitude=0.7)
    vocals_signal = silence(5.0)
    mix_path = wav_writer("mix.wav", mix_signal)
    vocals_path = wav_writer("vocals.wav", vocals_signal)

    result = analyze(mix_path, vocals_path, frame_duration=1.0)

    assert result["vocal_coverage"] == pytest.approx(0.0)


def test_analyze_total_silence_excludes_frames(wav_writer):
    silent_signal = silence(3.0)
    mix_path = wav_writer("mix.wav", silent_signal)
    vocals_path = wav_writer("vocals.wav", silent_signal)

    result = analyze(mix_path, vocals_path, frame_duration=1.0)

    assert result["silent_frame_count"] == result["frame_count"]
    assert result["vocal_coverage"] == 0.0
