from __future__ import annotations

import subprocess

import numpy as np
import pytest
import soundfile as sf

SR = 22050


def sine_wave(freq: float, duration: float, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def vibrato_tone(
    freq: float,
    duration: float,
    sr: int = SR,
    amplitude: float = 0.5,
    vibrato_rate: float = 5.5,
    vibrato_depth: float = 0.03,
) -> np.ndarray:
    """Sine with slow sinusoidal FM, simulating real singing with vibrato."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f_inst = freq * (1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    return (amplitude * np.sin(phase)).astype(np.float32)


def white_noise(duration: float, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (amplitude * rng.uniform(-1, 1, int(sr * duration))).astype(np.float32)


def silence(duration: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


@pytest.fixture
def wav_writer(tmp_path):
    def _write(name: str, signal: np.ndarray, sr: int = SR) -> str:
        path = tmp_path / name
        sf.write(str(path), signal, sr)
        return str(path)

    return _write


@pytest.fixture
def mp3_writer(tmp_path):
    def _write(name: str, signal: np.ndarray, sr: int = SR) -> str:
        wav_path = tmp_path / f"_src_{name}.wav"
        sf.write(str(wav_path), signal, sr)
        mp3_path = tmp_path / name
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(wav_path),
                str(mp3_path),
            ],
            check=True,
        )
        return str(mp3_path)

    return _write
