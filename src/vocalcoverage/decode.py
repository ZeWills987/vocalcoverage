"""In-memory audio decoding via FFmpeg.

vocalcoverage never writes converted audio files to disk: decoding
happens entirely in memory, piping PCM data through FFmpeg's stdout.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np

FFMPEG_NOT_FOUND_MESSAGE = (
    "FFmpeg was not found in the system PATH. vocalcoverage requires FFmpeg "
    "to decode audio files.\n"
    "Install it with:\n"
    "  - macOS:   brew install ffmpeg\n"
    "  - Ubuntu/Debian: sudo apt-get install ffmpeg\n"
    "  - Windows: winget install ffmpeg (or download from https://ffmpeg.org/download.html)\n"
    "Then ensure the `ffmpeg` binary is available on your PATH."
)


def decode_audio(path: str, sr: int = 22050) -> np.ndarray:
    """Decode any audio file to a mono float32 PCM buffer at the target sample rate.

    Decoding is performed entirely in memory via an FFmpeg subprocess: no
    intermediate or converted audio file is ever written to disk.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(FFMPEG_NOT_FOUND_MESSAGE)

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]

    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg failed to decode '{path}': {stderr}")

    return np.frombuffer(result.stdout, dtype=np.float32).copy()
