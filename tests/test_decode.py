from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pytest

from vocalcoverage.decode import decode_audio
from conftest import SR, sine_wave


def test_decode_wav_roundtrip(wav_writer):
    signal = sine_wave(220.0, 1.0)
    path = wav_writer("tone.wav", signal)

    decoded = decode_audio(path, sr=SR)

    assert len(decoded) == pytest.approx(len(signal), abs=SR * 0.05)
    assert np.abs(decoded).max() > 0.1


def test_decode_missing_ffmpeg_raises_clear_error(wav_writer):
    signal = sine_wave(220.0, 0.5)
    path = wav_writer("tone.wav", signal)

    with mock.patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="FFmpeg was not found"):
            decode_audio(path)


def test_decode_does_not_write_files_to_disk(wav_writer, tmp_path):
    signal = sine_wave(220.0, 0.5)
    path = wav_writer("tone.wav", signal)

    watch_dir = tmp_path / "watched"
    watch_dir.mkdir()
    before = set(os.listdir(watch_dir))

    cwd = os.getcwd()
    os.chdir(watch_dir)
    try:
        decode_audio(path, sr=SR)
    finally:
        os.chdir(cwd)

    after = set(os.listdir(watch_dir))
    assert before == after


def test_decode_wav_and_mp3_are_consistent(wav_writer, mp3_writer):
    signal = sine_wave(440.0, 1.0)
    wav_path = wav_writer("tone.wav", signal)
    mp3_path = mp3_writer("tone.mp3", signal)

    decoded_wav = decode_audio(wav_path, sr=SR)
    decoded_mp3 = decode_audio(mp3_path, sr=SR)

    min_len = min(len(decoded_wav), len(decoded_mp3))
    rms_wav = np.sqrt(np.mean(decoded_wav[:min_len] ** 2))
    rms_mp3 = np.sqrt(np.mean(decoded_mp3[:min_len] ** 2))

    assert rms_mp3 == pytest.approx(rms_wav, rel=0.2)
