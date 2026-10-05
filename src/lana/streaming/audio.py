import subprocess
from collections.abc import Generator
from pathlib import Path
from tempfile import TemporaryFile

import numpy as np

from lana.streaming.frontend import PcmChunk


def iter_pcm(
    path: str | Path,
    sample_rate: int,
    *,
    block_samples: int = 800,
) -> Generator[PcmChunk]:
    """Decode a file incrementally into native-endian mono PCM16 blocks.

    Training and streaming use this same conversion. FFmpeg diagnostics are
    redirected to a temporary file, and early iterator closure terminates the
    child process before waiting for it.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Audio input not found: {source}")
    if sample_rate < 1 or block_samples < 1:
        raise ValueError("Sample rate and block size must be positive.")

    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]

    with TemporaryFile() as diagnostics:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=diagnostics,
        )
        try:
            if process.stdout is None:
                raise RuntimeError("FFmpeg did not expose an audio output pipe.")

            remainder = b""
            while block := process.stdout.read(block_samples * 2):
                data = remainder + block
                complete = len(data) - len(data) % 2
                remainder = data[complete:]
                if complete:
                    yield np.frombuffer(
                        data[:complete],
                        dtype="<i2",
                    ).astype(np.int16, copy=True)

            status = process.wait()
            if remainder or status:
                diagnostics.seek(0)
                message = diagnostics.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"FFmpeg decoding failed: {message or f'exit status {status}'}"
                )
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
