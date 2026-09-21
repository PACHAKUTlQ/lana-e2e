import argparse
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import NDArray

from lana.artifacts import Tokens
from lana.runtime.pipeline import Lana

_MAGIC = b"LANA"
_VERSION = 1
_HEADER = struct.Struct(">4sBHIBi")


@dataclass(frozen=True, slots=True)
class LowRateSettings:
    """Temporal token sampling settings for the low-rate file format."""

    stride: int

    @classmethod
    def from_file(cls, path: str | Path) -> "LowRateSettings":
        """Load and validate the temporal sampling stride from JSON."""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {"stride"}:
            raise ValueError("Settings must be a JSON object containing only stride.")

        stride = data["stride"]
        if type(stride) is not int or not 1 <= stride <= 65535:
            raise ValueError("stride must be an integer between 1 and 65535.")

        return cls(stride=stride)


def _pack_indices(indices: NDArray[np.int64], width: int) -> bytes:
    """Pack nonnegative indices consecutively, most-significant bit first."""

    output = bytearray()
    buffer = 0
    buffered_bits = 0

    for value in indices:
        buffer = (buffer << width) | int(value)
        buffered_bits += width

        while buffered_bits >= 8:
            buffered_bits -= 8
            output.append((buffer >> buffered_bits) & 0xFF)
            buffer &= (1 << buffered_bits) - 1

    if buffered_bits:
        output.append(buffer << (8 - buffered_bits))

    return bytes(output)


def _unpack_indices(
    payload: bytes,
    count: int,
    width: int,
) -> NDArray[np.int64]:
    """Recover exactly the declared number of indices and validate padding."""

    expected_bytes = (count * width + 7) // 8
    if len(payload) != expected_bytes:
        raise ValueError(
            f"Invalid payload length: expected {expected_bytes}, got {len(payload)}."
        )

    indices = np.empty(count, dtype=np.int64)
    buffer = 0
    buffered_bits = 0
    next_index = 0
    mask = (1 << width) - 1

    for byte in payload:
        buffer = (buffer << 8) | byte
        buffered_bits += 8

        while buffered_bits >= width and next_index < count:
            buffered_bits -= width
            indices[next_index] = (buffer >> buffered_bits) & mask
            next_index += 1
            buffer &= (1 << buffered_bits) - 1

    if next_index != count or buffer != 0:
        raise ValueError("The token payload has invalid trailing bits.")

    return indices


def pack_tokens(tokens: Tokens, settings: LowRateSettings) -> bytes:
    """Encode current-model token arrays into the low-rate binary file format.

    Content indices are sampled every `stride` positions. The original sequence
    length is retained so unpacking can repeat indices back to the model's
    expected shape. Speaker information is included once per file.
    """

    content = tokens.content
    speaker = tokens.speaker

    if content.dtype != np.int64 or content.ndim != 2 or content.shape[0] != 1:
        raise TypeError(
            "Expected content tokens with shape (1, frames) and int64 dtype."
        )
    if speaker.dtype != np.int32 or speaker.shape != (1, 1):
        raise TypeError("Expected speaker tokens with shape (1, 1) and int32 dtype.")
    if not 1 <= settings.stride <= 65535:
        raise ValueError("stride must be between 1 and 65535.")

    frame_count = content.shape[1]
    if not 1 <= frame_count <= 0xFFFFFFFF:
        raise ValueError("Content token count is outside the supported range.")
    if np.any(content < 0):
        raise ValueError("This format supports only nonnegative content indices.")

    selected = np.ascontiguousarray(content[0, :: settings.stride])
    width = max(1, int(np.max(selected)).bit_length())
    if width > 63:
        raise ValueError("Content indices exceed the supported signed-int64 range.")

    header = _HEADER.pack(
        _MAGIC,
        _VERSION,
        settings.stride,
        frame_count,
        width,
        int(speaker[0, 0]),
    )

    return header + _pack_indices(selected, width)


def unpack_tokens(data: bytes) -> Tokens:
    """Restore model-compatible token arrays from a low-rate binary file."""

    if len(data) < _HEADER.size:
        raise ValueError("File is shorter than the low-rate header.")

    magic, version, stride, frame_count, width, speaker = _HEADER.unpack_from(data)

    if magic != _MAGIC or version != _VERSION:
        raise ValueError("Unrecognized low-rate token format.")
    if stride == 0 or frame_count == 0 or not 1 <= width <= 63:
        raise ValueError("Invalid low-rate token header.")

    selected_count = (frame_count + stride - 1) // stride
    selected = _unpack_indices(data[_HEADER.size :], selected_count, width)
    restored = np.repeat(selected, stride)[:frame_count].reshape(1, -1)

    return Tokens(
        content=restored,
        speaker=np.array([[speaker]], dtype=np.int32),
    )


def _write_bytes(path: str | Path, data: bytes) -> Path:
    """Atomically write a packed token file to its destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)

        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return destination


def main(argv: list[str] | None = None) -> int:
    """Encode audio into a compact file or decode one into PCM16 audio."""

    parser = argparse.ArgumentParser(prog="python -m lana.lowrate")
    commands = parser.add_subparsers(dest="command", required=True)

    encode = commands.add_parser("encode")
    encode.add_argument("input", type=Path)
    encode.add_argument("output", type=Path)
    encode.add_argument("--config", type=Path, default=Path("config.json"))
    encode.add_argument("--settings", type=Path, default=Path("lowrate.json"))

    decode = commands.add_parser("decode")
    decode.add_argument("input", type=Path)
    decode.add_argument("output", type=Path)
    decode.add_argument("--config", type=Path, default=Path("config.json"))

    arguments = parser.parse_args(argv)
    application = Lana(arguments.config)

    if arguments.command == "encode":
        settings = LowRateSettings.from_file(arguments.settings)
        tokens = application.encode(arguments.input)
        packed = pack_tokens(tokens, settings)
        destination = _write_bytes(arguments.output, packed)

        print(f"Content tokens: {tokens.content.shape[1]}")
        print(f"Speaker tokens: {tokens.speaker.size} (shape={tokens.speaker.shape})")
        print(f"Speaker feature: {tokens.speaker.nbytes} bytes")
        print(f"Stride: {settings.stride}")
        print(f"Packed file: {destination} ({len(packed)} bytes)")
    else:
        packed = arguments.input.read_bytes()
        tokens = unpack_tokens(packed)
        application.decode(tokens).save(arguments.output)
        print(f"Decoded WAV: {arguments.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
