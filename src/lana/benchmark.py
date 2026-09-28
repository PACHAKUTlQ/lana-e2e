"""In-process latency and token-rate measurements for the Lana pipeline."""

import argparse
import io
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch

from lana.artifacts import Tokens
from lana.config import Config, Feature
from lana.pipeline import Lana

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Measurement:
    """Wall-clock timings collected after the requested warm-up runs."""

    seconds: tuple[float, ...]

    @property
    def median(self) -> float:
        """Return the median elapsed time in seconds."""

        return statistics.median(self.seconds)

    @property
    def minimum(self) -> float:
        """Return the fastest elapsed time in seconds."""

        return min(self.seconds)

    @property
    def p95(self) -> float:
        """Return the 95th percentile elapsed time in seconds."""

        return float(np.percentile(self.seconds, 95))


def _synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work so wall-clock measurements include execution."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call[T](
    operation: Callable[[], T],
    device: torch.device,
) -> tuple[T, float]:
    """Execute an operation once and return its result and synchronized latency."""

    _synchronize(device)
    start = time.perf_counter()
    result = operation()
    _synchronize(device)

    return result, time.perf_counter() - start


def _measure(
    operation: Callable[[], object],
    device: torch.device,
    *,
    warmup: int,
    runs: int,
) -> Measurement:
    """Warm up an operation, then collect independent synchronized timings."""

    for _ in range(warmup):
        operation()
    _synchronize(device)

    timings: list[float] = []
    for _ in range(runs):
        _, seconds = _time_call(operation, device)
        timings.append(seconds)

    return Measurement(tuple(timings))


def _npy_size(array: np.ndarray) -> int:
    """Return the serialized NPY size, including its format header."""

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)

    return buffer.tell()


def _print_measurement(
    name: str,
    measurement: Measurement,
    duration: float,
) -> None:
    """Print latency distribution and real-time factor for one operation."""

    median = measurement.median
    print(
        f"{name:<23} "
        f"median={median:8.4f}s  "
        f"min={measurement.minimum:8.4f}s  "
        f"p95={measurement.p95:8.4f}s  "
        f"RTF={median / duration:7.3f}  "
        f"speed={duration / median:7.2f}x"
    )


def _print_rate(tokens: Tokens, duration: float) -> None:
    """Report raw array and serialized-file rates over the encoded duration."""

    content_bytes = tokens.content.nbytes
    speaker_bytes = tokens.speaker.nbytes
    content_file_bytes = _npy_size(tokens.content)
    speaker_file_bytes = _npy_size(tokens.speaker)

    print("\nTokens")
    print(
        f"  content: shape={tokens.content.shape}, "
        f"dtype={tokens.content.dtype}, payload={content_bytes} bytes"
    )
    print(
        f"  speaker: shape={tokens.speaker.shape}, "
        f"dtype={tokens.speaker.dtype}, payload={speaker_bytes} bytes"
    )
    print(f"  content payload rate: {content_bytes * 8 / duration:.2f} bps")
    print(
        "  content + one speaker payload rate: "
        f"{(content_bytes + speaker_bytes) * 8 / duration:.2f} bps"
    )
    print(
        "  two NPY files rate: "
        f"{(content_file_bytes + speaker_file_bytes) * 8 / duration:.2f} bps"
    )


@torch.inference_mode()
def benchmark(
    config_path: Path,
    input_path: Path,
    *,
    device_override: str | None,
    warmup: int,
    runs: int,
) -> None:
    """Measure an utterance without including model loading in steady-state runs.

    Duration is based on the Mel frames supplied to the codec. Model-only
    encoding reuses a fixed prompt; complete encoding retains the configured
    random-prompt behavior. Every measured path uses the public application
    API or the codec's exported method.
    """

    config = Config.from_file(config_path)
    if device_override is not None:
        config = config.model_copy(update={"device": device_override})

    application = Lana(config)
    device = application.models.device

    mel_features, frontend_first = _time_call(
        lambda: application.extract(input_path, (Feature.MEL,)),
        device,
    )
    mel = mel_features[Feature.MEL]
    duration = mel.shape[1] * config.audio.hop_size / config.audio.sample_rate
    if duration <= 0:
        raise ValueError("The input produced no Mel frames.")

    _, codec_load = _time_call(lambda: application.models.codec, device)

    def load_vocoder() -> None:
        vocoder = application.models.vocoder
        _ = vocoder.pitch_model
        _ = vocoder.prior_generator
        _ = vocoder.generator
        _ = vocoder.mel_transform

    _, vocoder_load = _time_call(load_vocoder, device)

    codec_input = (torch.from_numpy(mel).unsqueeze(0).to(device) + 1) / 2
    prompt_length = config.encoding.prompt_length
    frame_count = codec_input.shape[-1]
    if frame_count >= prompt_length:
        prompt = codec_input[:, :, :prompt_length]
    else:
        repetitions = (prompt_length + frame_count - 1) // frame_count
        prompt = codec_input.tile((1, 1, repetitions))[:, :, :prompt_length]

    tokens = application.encode(input_path)
    decoded_mel = application.decode_tokens(tokens)

    def encode_model_only() -> object:
        """Run only the codec's exported encode method on prepared tensors."""

        return application.models.codec.encode(codec_input, prompt)

    measurements = (
        (
            "Mel preprocessing",
            lambda: application.extract(input_path, (Feature.MEL,)),
        ),
        ("codec encode only", encode_model_only),
        ("encode including input", lambda: application.encode(input_path)),
        ("codec decode only", lambda: application.decode_tokens(tokens)),
        ("vocoder only", lambda: application.decode_mel(decoded_mel)),
        ("decode from tokens", lambda: application.decode(tokens)),
        ("chunked infer", lambda: application.infer(input_path, progress=False)),
    )

    print(f"Input: {input_path}")
    print(f"Device: {device}")
    if device.type == "cpu":
        print(f"PyTorch CPU threads: {torch.get_num_threads()}")
    else:
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Codec input: {mel.shape[1]} frames, {duration:.3f}s")
    print(f"First frontend call: {frontend_first:.4f}s")
    print(f"Codec model loading: {codec_load:.4f}s")
    print(f"Vocoder model/transform loading: {vocoder_load:.4f}s")
    print(f"Steady-state timings: {warmup} warm-up, {runs} measured runs\n")

    for name, operation in measurements:
        result = _measure(operation, device, warmup=warmup, runs=runs)
        _print_measurement(name, result, duration)

    _print_rate(tokens, duration)


def main(argv: list[str] | None = None) -> int:
    """Parse benchmark options and return a shell-compatible status code."""

    parser = argparse.ArgumentParser(
        prog="python -m lana.benchmark",
        description="Measure in-process SingleCodec latency and token rates.",
    )
    parser.add_argument("input", type=Path, help="Input WAV file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Application configuration file.",
    )
    parser.add_argument("--device", help="Override the configured Torch device.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up calls per measured operation.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured calls per operation.",
    )
    arguments = parser.parse_args(argv)

    if arguments.warmup < 0:
        parser.error("--warmup must be zero or greater.")
    if arguments.runs < 1:
        parser.error("--runs must be positive.")

    benchmark(
        arguments.config,
        arguments.input,
        device_override=arguments.device,
        warmup=arguments.warmup,
        runs=arguments.runs,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
