import hashlib
import json
import socket
import subprocess
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryFile

import numpy as np
from numpy.typing import NDArray

from lana.config import Config
from lana.runtime.models import Models
from lana.streaming.session import EncodedChunk, StreamDecoder, StreamEncoder
from lana.streaming.wire import (
    ACCEPT,
    CHUNK,
    END,
    REJECT,
    SETUP,
    ByteConnection,
    StreamSetup,
    pack_chunk,
    receive_message,
    send_message,
    unpack_chunk,
)

_READ_SAMPLES = 4096


def _sha256_file(path: Path) -> str:
    """Hash the exact codec artifact used by the local endpoint."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _setup(config: Config) -> StreamSetup:
    """Identify the codec and signal settings required by the receiver."""

    settings = {
        "audio": config.audio.model_dump(mode="json"),
        "streaming": config.streaming.model_dump(mode="json"),
        "vocoder": config.vocoder.model_dump(mode="json"),
    }
    serialized = json.dumps(
        settings,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return StreamSetup(
        model_sha256=_sha256_file(config.models.codec),
        config_sha256=hashlib.sha256(serialized).hexdigest(),
        sample_rate=config.audio.sample_rate,
        output_sample_rate=config.vocoder.sampling_rate,
    )


def _pcm_from_ffmpeg(path: Path, sample_rate: int) -> Iterator[NDArray[np.int16]]:
    """Decode an input file incrementally to mono PCM16 at the frontend rate."""

    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
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

    with (
        TemporaryFile() as diagnostics,
        subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=diagnostics,
        ) as process,
    ):
        if process.stdout is None:
            raise RuntimeError("Failed to open FFmpeg output pipe.")

        remainder = b""
        while block := process.stdout.read(_READ_SAMPLES * 2):
            data = remainder + block
            complete = len(data) - len(data) % 2
            remainder = data[complete:]

            if complete:
                yield np.frombuffer(data[:complete], dtype="<i2").copy()

        status = process.wait()
        if remainder or status:
            diagnostics.seek(0)
            error = (
                diagnostics.read()
                .decode(
                    "utf-8",
                    errors="replace",
                )
                .strip()
            )
            raise RuntimeError(
                f"FFmpeg decoding failed: {error or f'exit status {status}'}"
            )


def _send_chunks(
    connection: ByteConnection,
    chunks: list[EncodedChunk],
) -> None:
    """Send newly encoded chunks in their original sequence order."""

    for chunk in chunks:
        send_message(connection, CHUNK, pack_chunk(chunk))


def send_file(
    input_path: str | Path,
    *,
    host: str,
    port: int,
    config: Config,
    realtime: bool = True,
) -> None:
    """Encode an audio file and send chunks over an ordered byte connection.

    The receiver accepts the model and signal settings before file decoding
    starts. With realtime pacing, input consumption follows recording time;
    TCP backpressure can make it slower if processing cannot keep up.
    """

    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input audio not found: {source}")

    setup = _setup(config)
    encoder = StreamEncoder(config, Models(config))

    with socket.create_connection((host, port)) as connection:
        send_message(connection, SETUP, setup.to_bytes())
        kind, payload = receive_message(connection)

        if kind == REJECT:
            raise RuntimeError(payload.decode("utf-8", errors="replace"))
        if kind != ACCEPT or payload:
            raise ValueError("The receiver did not accept the stream.")

        start = time.monotonic()
        sample_count = 0

        for samples in _pcm_from_ffmpeg(source, config.audio.sample_rate):
            _send_chunks(connection, encoder.push(samples))
            sample_count += samples.size

            if realtime:
                deadline = start + sample_count / config.audio.sample_rate
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

        _send_chunks(connection, encoder.finish())
        send_message(connection, END)


def _receive_connection(
    connection: socket.socket,
    output_path: Path,
    config: Config,
    local_setup: StreamSetup,
    models: Models,
) -> None:
    """Validate one connection and progressively decode it into a WAV file."""

    kind, payload = receive_message(connection)
    if kind != SETUP:
        raise ValueError("The first stream message must be setup.")

    try:
        remote_setup = StreamSetup.from_bytes(payload)
        if remote_setup != local_setup:
            raise ValueError(
                "Codec artifact or streaming signal configuration differs."
            )
    except (ValueError, TypeError) as error:
        send_message(connection, REJECT, str(error).encode("utf-8"))

        return

    send_message(connection, ACCEPT)
    decoder = StreamDecoder(config, models)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(config.vocoder.sampling_rate)

        while True:
            kind, payload = receive_message(connection)

            if kind == END:
                if payload:
                    raise ValueError("The end message must be empty.")

                return

            if kind != CHUNK:
                raise ValueError(f"Unexpected stream message type: {kind}.")

            audio = decoder.push(unpack_chunk(payload))
            output.writeframes(audio.samples.astype("<i2", copy=False).tobytes())


def receive_file(
    output_path: str | Path,
    *,
    host: str,
    port: int,
    config: Config,
) -> None:
    """Accept one TCP connection and synthesize its chunks into a WAV file."""

    local_setup = _setup(config)
    models = Models(config)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)

        connection, _ = listener.accept()
        with connection:
            _receive_connection(
                connection,
                Path(output_path).expanduser().resolve(),
                config,
                local_setup,
                models,
            )
