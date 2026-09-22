import json
import struct
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lana.streaming.session import EncodedChunk

MAGIC = "LANA-STREAM"
VERSION = 1
MAX_MESSAGE_BYTES = 1 << 20

_KIND_HEADER = struct.Struct(">BI")
_CHUNK_HEADER = struct.Struct(">IHHBq")

SETUP = 1
CHUNK = 2
END = 3
ACCEPT = 4
REJECT = 5


class ByteConnection(Protocol):
    """Reliable, ordered byte connection used by the framing protocol."""

    def recv(self, bufsize: int, /) -> bytes:
        """Receive up to bufsize bytes."""

        ...

    def sendall(self, data: bytes, /) -> None:
        """Send every byte or raise an exception."""

        ...


@dataclass(frozen=True, slots=True)
class StreamSetup:
    """Parameters checked before accepting an encoded audio stream."""

    model_sha256: str
    config_sha256: str
    sample_rate: int
    output_sample_rate: int

    def to_bytes(self) -> bytes:
        """Serialize the setup record as versioned UTF-8 JSON."""

        return json.dumps(
            {
                "magic": MAGIC,
                "version": VERSION,
                "model_sha256": self.model_sha256,
                "config_sha256": self.config_sha256,
                "sample_rate": self.sample_rate,
                "output_sample_rate": self.output_sample_rate,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "StreamSetup":
        """Validate and deserialize a received setup record."""

        value = json.loads(data)
        if not isinstance(value, dict):
            raise TypeError("Stream setup must be a JSON object.")

        expected = {
            "magic",
            "version",
            "model_sha256",
            "config_sha256",
            "sample_rate",
            "output_sample_rate",
        }
        if set(value) != expected:
            raise ValueError("Invalid stream setup fields.")
        if value["magic"] != MAGIC or type(value["version"]) is not int:
            raise ValueError("Unsupported stream protocol.")
        if value["version"] != VERSION:
            raise ValueError("Unsupported stream protocol.")

        for key in ("model_sha256", "config_sha256"):
            digest = value[key]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"Invalid {key}.")

        for key in ("sample_rate", "output_sample_rate"):
            if type(value[key]) is not int or value[key] <= 0:
                raise ValueError(f"Invalid {key}.")

        return cls(
            model_sha256=value["model_sha256"],
            config_sha256=value["config_sha256"],
            sample_rate=value["sample_rate"],
            output_sample_rate=value["output_sample_rate"],
        )


def _read_exact(connection: ByteConnection, length: int) -> bytes:
    """Read exactly length bytes or report an interrupted connection."""

    result = bytearray()
    while len(result) < length:
        part = connection.recv(length - len(result))
        if not part:
            raise ConnectionError("Connection closed inside a stream message.")
        result.extend(part)

    return bytes(result)


def send_message(
    connection: ByteConnection,
    kind: int,
    payload: bytes = b"",
) -> None:
    """Send one length-prefixed protocol message."""

    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Stream message exceeds the size limit.")

    connection.sendall(_KIND_HEADER.pack(kind, len(payload)) + payload)


def receive_message(connection: ByteConnection) -> tuple[int, bytes]:
    """Receive one complete length-prefixed protocol message."""

    header = _read_exact(connection, _KIND_HEADER.size)
    kind, length = _KIND_HEADER.unpack(header)
    if length > MAX_MESSAGE_BYTES:
        raise ValueError("Stream message exceeds the size limit.")

    return kind, _read_exact(connection, length)


def pack_chunk(chunk: EncodedChunk) -> bytes:
    """Pack nonnegative content indices at the width required by this chunk."""

    content = chunk.content
    if (
        content.dtype != np.int64
        or content.ndim != 1
        or not 1 <= content.size <= 65535
        or np.any(content < 0)
    ):
        raise ValueError("Expected 1–65535 nonnegative int64 content indices.")
    if not 0 <= chunk.sequence <= 0xFFFFFFFF:
        raise ValueError("Chunk sequence exceeds the protocol range.")
    if not 1 <= chunk.frames <= 65535:
        raise ValueError("Chunk frame count exceeds the protocol range.")

    width = max(1, int(np.max(content)).bit_length())
    if width > 63:
        raise ValueError("Content index exceeds the supported range.")

    payload = bytearray()
    buffer = 0
    buffered_bits = 0

    for index in content:
        buffer = (buffer << width) | int(index)
        buffered_bits += width

        while buffered_bits >= 8:
            buffered_bits -= 8
            payload.append((buffer >> buffered_bits) & 0xFF)
            buffer &= (1 << buffered_bits) - 1

    if buffered_bits:
        payload.append(buffer << (8 - buffered_bits))

    return (
        _CHUNK_HEADER.pack(
            chunk.sequence,
            chunk.frames,
            content.size,
            width,
            chunk.speaker,
        )
        + payload
    )


def unpack_chunk(data: bytes) -> EncodedChunk:
    """Restore a chunk and reject invalid lengths or nonzero padding bits."""

    if len(data) < _CHUNK_HEADER.size:
        raise ValueError("Truncated encoded chunk.")

    sequence, frames, count, width, speaker = _CHUNK_HEADER.unpack_from(data)
    if frames == 0 or count == 0 or not 1 <= width <= 63:
        raise ValueError("Invalid encoded chunk header.")

    payload = data[_CHUNK_HEADER.size :]
    if len(payload) != (count * width + 7) // 8:
        raise ValueError("Invalid encoded chunk payload length.")

    content = np.empty(count, dtype=np.int64)
    buffer = 0
    buffered_bits = 0
    position = 0
    mask = (1 << width) - 1

    for byte in payload:
        buffer = (buffer << 8) | byte
        buffered_bits += 8

        while buffered_bits >= width and position < count:
            buffered_bits -= width
            content[position] = (buffer >> buffered_bits) & mask
            position += 1
            buffer &= (1 << buffered_bits) - 1

    if position != count or buffer != 0:
        raise ValueError("Invalid encoded chunk padding.")

    return EncodedChunk(
        sequence=sequence,
        frames=frames,
        content=content,
        speaker=speaker,
    )
