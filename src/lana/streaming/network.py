import hashlib
import json
import socket
import time
import wave
from contextlib import closing
from pathlib import Path

import torch

from lana.codec.artifact import signal_signature
from lana.config import Config
from lana.runtime.models import Models
from lana.streaming.audio import iter_pcm
from lana.streaming.metrics import SendMetrics
from lana.streaming.session import EncodedChunk, StreamDecoder, StreamEncoder
from lana.streaming.wire import (
    ACCEPT,
    CHUNK,
    END,
    MESSAGE_HEADER_BYTES,
    REJECT,
    SETUP,
    ByteConnection,
    StreamSetup,
    pack_chunk,
    receive_message,
    send_message,
    unpack_chunk,
)


def _synchronize(device: torch.device) -> None:
    """Include queued accelerator work in measured processing times."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _setup(config: Config) -> StreamSetup:
    """Identify codec weights, selected bitrate, and synthesis signal settings."""

    digest = hashlib.sha256()
    with config.models.codec.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    settings = {
        "signal": signal_signature(config),
        "codec": config.codec.model_dump(mode="json"),
        "vocoder": config.vocoder.model_dump(mode="json"),
    }
    serialized = json.dumps(
        settings,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return StreamSetup(
        model_sha256=digest.hexdigest(),
        config_sha256=hashlib.sha256(serialized).hexdigest(),
        sample_rate=config.audio.sample_rate,
        output_sample_rate=config.vocoder.sampling_rate,
    )


def _send_chunks(
    connection: ByteConnection,
    chunks: list[EncodedChunk],
    metrics: SendMetrics,
) -> None:
    """Serialize chunks and measure successful transport submissions."""

    for chunk in chunks:
        started = time.perf_counter()
        payload = pack_chunk(chunk)
        build_seconds = time.perf_counter() - started
        started = time.perf_counter()
        send_message(connection, CHUNK, payload)
        sent_at = time.perf_counter()
        metrics.record_chunk(
            chunk,
            len(payload),
            build_seconds=build_seconds,
            send_seconds=sent_at - started,
            sent_at=sent_at,
        )


def send_file(
    input_path: str | Path,
    *,
    host: str,
    port: int,
    config: Config,
    realtime: bool = True,
) -> None:
    """Send a paced file stream using one configured residual quantizer prefix."""

    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Audio input not found: {source}")

    started = time.perf_counter()
    setup = _setup(config)
    setup_seconds = time.perf_counter() - started
    models = Models(config)
    started = time.perf_counter()
    _ = models.codec
    _synchronize(models.device)
    codec_load_seconds = time.perf_counter() - started
    encoder = StreamEncoder(config, models)

    with socket.create_connection(
        (host, port),
        timeout=config.streaming.io_timeout_seconds,
    ) as connection:
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
            setup_bytes=MESSAGE_HEADER_BYTES + len(setup_payload),
            received_control_bytes=MESSAGE_HEADER_BYTES,
        )

        with closing(iter_pcm(source, config.audio.sample_rate)) as blocks:
            for samples in blocks:
                next_samples = metrics.input_samples + samples.size
                if realtime:
                    deadline = (
                        metrics.started_at + next_samples / config.audio.sample_rate
                    )
                    remaining = deadline - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                metrics.input_samples = next_samples

                _synchronize(models.device)
                started = time.perf_counter()
                chunks = encoder.push(samples)
                _synchronize(models.device)
                metrics.record_encoder_call(time.perf_counter() - started)
                _send_chunks(connection, chunks, metrics)
                metrics.observe_send_lag(time.perf_counter())

        _synchronize(models.device)
        started = time.perf_counter()
        chunks = encoder.finish()
        _synchronize(models.device)
        metrics.record_encoder_call(time.perf_counter() - started)
        _send_chunks(connection, chunks, metrics)

        if metrics.chunk_count == 0:
            raise ValueError("The input produced no complete Mel frames.")

        send_message(connection, END)
        metrics.end_sent_at = time.perf_counter()
        metrics.end_bytes = MESSAGE_HEADER_BYTES
        metrics.observe_send_lag(metrics.end_sent_at)

        kind, payload = receive_message(connection)
        if kind == REJECT:
            raise RuntimeError(payload.decode("utf-8", errors="replace"))
        if kind != ACCEPT or payload:
            raise ValueError("The receiver did not confirm completion.")
        metrics.received_control_bytes += MESSAGE_HEADER_BYTES
        metrics.completed_at = time.perf_counter()

    metrics.report(
        setup_seconds=setup_seconds,
        codec_load_seconds=codec_load_seconds,
    )


def _receive_connection(
    connection: ByteConnection,
    output_path: Path,
    config: Config,
    setup: StreamSetup,
    models: Models,
) -> None:
    """Validate one session and progressively write decoded PCM to a WAV file."""

    kind, payload = receive_message(connection)
    if kind != SETUP:
        raise ValueError("The first message must be SETUP.")
    if StreamSetup.from_bytes(payload) != setup:
        raise ValueError("Codec, bitrate, or signal settings differ.")

    decoder = StreamDecoder(config, models)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(config.vocoder.sampling_rate)
        send_message(connection, ACCEPT)

        while True:
            kind, payload = receive_message(connection)
            if kind == END:
                if payload or count == 0:
                    raise ValueError("Invalid END message or empty audio stream.")
                break
            if kind != CHUNK:
                raise ValueError("Expected CHUNK or END.")

            chunk = unpack_chunk(payload)
            audio = decoder.push(chunk)
            output.writeframes(audio.samples.astype("<i2", copy=False).tobytes())
            count += 1

    send_message(connection, ACCEPT)


def receive_file(
    output_path: str | Path,
    *,
    host: str,
    port: int,
    config: Config,
) -> None:
    """Load receiver models and accept one ordered TCP audio stream."""

    setup = _setup(config)
    models = Models(config)
    _ = models.codec
    vocoder = models.vocoder
    _ = vocoder.pitch_model
    _ = vocoder.prior_generator
    _ = vocoder.generator
    _ = vocoder.mel_transform
    _synchronize(models.device)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        print(f"Listening on {host}:{port}", flush=True)

        connection, _ = listener.accept()
        with connection:
            connection.settimeout(config.streaming.io_timeout_seconds)
            try:
                _receive_connection(
                    connection,
                    Path(output_path).expanduser().resolve(),
                    config,
                    setup,
                    models,
                )
            except (OSError, ValueError, RuntimeError) as error:
                try:
                    send_message(
                        connection,
                        REJECT,
                        str(error).encode("utf-8")[:4096],
                    )
                except OSError:
                    pass
                raise
