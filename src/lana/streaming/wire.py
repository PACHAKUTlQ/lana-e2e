import json
import struct
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lana.streaming.session import EncodedChunk

MAGIC = "LANA-STREAM"
VERSION = 2
MAX_MESSAGE_BYTES = 1 << 20

_MESSAGE_HEADER = struct.Struct(">BI")
_CHUNK_HEADER = struct.Struct(">IHHBBq")
MESSAGE_HEADER_BYTES = _MESSAGE_HEADER.size
CHUNK_HEADER_BYTES = _CHUNK_HEADER.size

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
        """Validate setup fields before constructing a session identity."""

        values = json.loads(data)
        expected = {
            "magic",
            "version",
            "model_sha256",
            "config_sha256",
            "sample_rate",
            "output_sample_rate",
        }
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError("Invalid stream setup fields.")
        if (
            values["magic"] != MAGIC
            or type(values["version"]) is not int
            or values["version"] != VERSION
        ):
            raise ValueError("Unsupported stream protocol.")

        for name in ("model_sha256", "config_sha256"):
            digest = values[name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"Invalid {name}.")
        for name in ("sample_rate", "output_sample_rate"):
            if type(values[name]) is not int or values[name] <= 0:
                raise ValueError(f"Invalid {name}.")

        return cls(
            model_sha256=values["model_sha256"],
            config_sha256=values["config_sha256"],
            sample_rate=values["sample_rate"],
            output_sample_rate=values["output_sample_rate"],
        )


def read_exact(connection: ByteConnection, length: int) -> bytes:
    """Read a complete field or report an interrupted stream."""

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

    if kind not in {SETUP, CHUNK, END, ACCEPT, REJECT}:
        raise ValueError("Unknown stream message type.")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Stream message exceeds its size limit.")
    connection.sendall(_MESSAGE_HEADER.pack(kind, len(payload)) + payload)


def receive_message(connection: ByteConnection) -> tuple[int, bytes]:
    """Receive one complete length-prefixed protocol message."""

    kind, length = _MESSAGE_HEADER.unpack(read_exact(connection, MESSAGE_HEADER_BYTES))
    if kind not in {SETUP, CHUNK, END, ACCEPT, REJECT}:
        raise ValueError("Unknown stream message type.")
    if length > MAX_MESSAGE_BYTES:
        raise ValueError("Stream message exceeds its size limit.")

    return kind, read_exact(connection, length)


def pack_chunk(chunk: EncodedChunk) -> bytes:
    """Pack nonnegative content indices at the width required by this chunk."""

    content = chunk.content
    if (
        content.dtype != np.int64
        or content.ndim != 2
        or not 1 <= content.shape[0] <= 65535
        or not 1 <= content.shape[1] <= 16
        or np.any(content < 0)
    ):
        raise ValueError("Invalid layered int64 content codes.")
    if not 0 <= chunk.sequence <= 0xFFFFFFFF:
        raise ValueError("Sequence exceeds the protocol range.")
    if not 1 <= chunk.frames <= 65535:
        raise ValueError("Frame count exceeds the protocol range.")
    if not 0 <= chunk.speaker <= 0x7FFFFFFFFFFFFFFF:
        raise ValueError("Speaker index exceeds the protocol range.")

    width = max(1, int(content.max()).bit_length())
    payload_bytes = (content.size * width + 7) // 8
    if CHUNK_HEADER_BYTES + payload_bytes > MAX_MESSAGE_BYTES:
        raise ValueError("Encoded chunk exceeds its size limit.")

    payload = bytearray()
    buffer = 0
    bits = 0
    for value in content.flat:
        buffer = (buffer << width) | int(value)
        bits += width
        while bits >= 8:
            bits -= 8
            payload.append((buffer >> bits) & 0xFF)
            buffer &= (1 << bits) - 1
    if bits:
        payload.append(buffer << (8 - bits))

    return (
        _CHUNK_HEADER.pack(
            chunk.sequence,
            chunk.frames,
            content.shape[0],
            content.shape[1],
            width,
            chunk.speaker,
        )
        + payload
    )


def unpack_chunk(data: bytes) -> EncodedChunk:
    """Restore a chunk and reject invalid lengths or nonzero padding bits."""

    if not CHUNK_HEADER_BYTES <= len(data) <= MAX_MESSAGE_BYTES:
        raise ValueError("Invalid encoded chunk size.")
    sequence, frames, positions, layers, width, speaker = _CHUNK_HEADER.unpack_from(
        data
    )
    if (
        frames == 0
        or positions == 0
        or not 1 <= layers <= 16
        or not 1 <= width <= 63
        or speaker < 0
    ):
        raise ValueError("Invalid encoded chunk header.")

    count = positions * layers
    payload = data[CHUNK_HEADER_BYTES:]
    if len(payload) != (count * width + 7) // 8:
        raise ValueError("Encoded payload length differs from its dimensions.")

    content = np.empty(count, dtype=np.int64)
    buffer = 0
    bits = 0
    position = 0
    mask = (1 << width) - 1
    for byte in payload:
        buffer = (buffer << 8) | byte
        bits += 8
        while bits >= width and position < count:
            bits -= width
            content[position] = (buffer >> bits) & mask
            position += 1
            buffer &= (1 << bits) - 1

    if position != count or buffer != 0:
        raise ValueError("Invalid encoded chunk padding.")

    return EncodedChunk(
        sequence=sequence,
        frames=frames,
        content=content.reshape(positions, layers),
        speaker=speaker,
    )
