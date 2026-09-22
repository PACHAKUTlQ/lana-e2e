import hashlib
import json
import socket
import struct
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
from lana.streaming.metrics import SendMetrics
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

_READ_SAMPLES = 800
_MESSAGE_HEADER_BYTES = struct.calcsize(">BI")


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


def _pcm_from_ffmpeg(
    path: Path,
    sample_rate: int,
) -> Iterator[NDArray[np.int16]]:
    """Decode an input file incrementally to mono PCM16 at the frontend rate.

    Diagnostics use a temporary file instead of a second pipe. This prevents
    FFmpeg from blocking when it writes enough diagnostic output to fill a pipe.
    """

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
    metrics: SendMetrics,
) -> None:
    """Serialize and submit chunks while measuring actual message sizes."""

    for chunk in chunks:
        build_started = time.perf_counter()
        payload = pack_chunk(chunk)
        build_seconds = time.perf_counter() - build_started

        send_started = time.perf_counter()
        send_message(connection, CHUNK, payload)
        sent_at = time.perf_counter()

        metrics.record_chunk(
            chunk,
            len(payload),
            build_seconds=build_seconds,
            send_seconds=sent_at - send_started,
            sent_at=sent_at,
        )


def _load_sender_codec(models: Models) -> float:
    """Load the sender's codec before starting the paced audio clock."""

    started = time.perf_counter()
    _ = models.codec

    return time.perf_counter() - started


def _load_receiver_models(models: Models) -> dict[str, float]:
    """Measure all receiver model and transform loads before accepting audio."""

    vocoder = models.vocoder
    operations = (
        ("codec", lambda: models.codec),
        ("pitch model", lambda: vocoder.pitch_model),
        ("prior generator", lambda: vocoder.prior_generator),
        ("waveform generator", lambda: vocoder.generator),
        ("prior Mel transform", lambda: vocoder.mel_transform),
    )
    timings: dict[str, float] = {}

    for name, operation in operations:
        started = time.perf_counter()
        operation()
        timings[name] = time.perf_counter() - started

    return timings


def send_file(
    input_path: str | Path,
    *,
    host: str,
    port: int,
    config: Config,
    realtime: bool = True,
) -> None:
    """Stream a file and measure encoder timing and exact serialized bytes.

    PCM is paced against a monotonic clock before entering the encoder. Codec
    loading and setup negotiation are excluded from that clock and reported
    separately. The final acknowledgement confirms receiver completion.
    """

    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input audio not found: {source}")

    setup_started = time.perf_counter()
    setup = _setup(config)
    setup_seconds = time.perf_counter() - setup_started

    models = Models(config)
    codec_load_seconds = _load_sender_codec(models)
    encoder = StreamEncoder(config, models)

    with socket.create_connection((host, port)) as connection:
        setup_payload = setup.to_bytes()
        send_message(connection, SETUP, setup_payload)

        kind, payload = receive_message(connection)
        if kind == REJECT:
            raise RuntimeError(payload.decode("utf-8", errors="replace"))
        if kind != ACCEPT or payload:
            raise ValueError("The receiver did not accept the stream.")

        metrics = SendMetrics(
            sample_rate=config.audio.sample_rate,
            hop_size=config.audio.hop_size,
            started_at=time.perf_counter(),
            setup_bytes=_MESSAGE_HEADER_BYTES + len(setup_payload),
            received_control_bytes=_MESSAGE_HEADER_BYTES,
        )

        for samples in _pcm_from_ffmpeg(source, config.audio.sample_rate):
            next_sample_count = metrics.input_samples + samples.size

            if realtime:
                deadline = (
                    metrics.started_at + next_sample_count / config.audio.sample_rate
                )
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

            metrics.input_samples = next_sample_count

            encode_started = time.perf_counter()
            chunks = encoder.push(samples)
            metrics.record_encoder_call(time.perf_counter() - encode_started)

            _send_chunks(connection, chunks, metrics)
            metrics.observe_send_lag(time.perf_counter())

        if metrics.input_samples == 0:
            raise ValueError("FFmpeg produced no audio samples.")

        encode_started = time.perf_counter()
        final_chunks = encoder.finish()
        metrics.record_encoder_call(time.perf_counter() - encode_started)
        _send_chunks(connection, final_chunks, metrics)

        if metrics.chunk_count == 0:
            raise ValueError("The input did not produce a complete Mel frame.")

        send_message(connection, END)
        metrics.end_sent_at = time.perf_counter()
        metrics.end_bytes = _MESSAGE_HEADER_BYTES
        metrics.observe_send_lag(metrics.end_sent_at)

        kind, payload = receive_message(connection)
        if kind != ACCEPT or payload:
            raise ValueError("The receiver did not confirm stream completion.")

        metrics.received_control_bytes += _MESSAGE_HEADER_BYTES
        metrics.completed_at = time.perf_counter()

    metrics.report(
        setup_seconds=setup_seconds,
        codec_load_seconds=codec_load_seconds,
    )


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
                break

            if kind != CHUNK:
                raise ValueError(f"Unexpected stream message type: {kind}.")

            audio = decoder.push(unpack_chunk(payload))
            output.writeframes(audio.samples.astype("<i2", copy=False).tobytes())

    send_message(connection, ACCEPT)


def receive_file(
    output_path: str | Path,
    *,
    host: str,
    port: int,
    config: Config,
) -> None:
    """Load receiver models, then accept and decode one TCP stream."""

    setup_started = time.perf_counter()
    local_setup = _setup(config)
    setup_seconds = time.perf_counter() - setup_started

    models = Models(config)
    load_timings = _load_receiver_models(models)

    print("Receiver startup")
    print(f"  Setup preparation:    {setup_seconds:.6f} s")
    for name, seconds in load_timings.items():
        print(f"  {name}: {seconds:.6f} s")
    print("  Ready to accept audio.", flush=True)

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
