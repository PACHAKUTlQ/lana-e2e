import struct
from dataclasses import dataclass

from lana.streaming.session import EncodedChunk

_MESSAGE_HEADER_BYTES = struct.calcsize(">BI")
_CHUNK_HEADER_BYTES = struct.calcsize(">IHHBq")


@dataclass(slots=True)
class SendMetrics:
    """Measure encoder work, pacing, and exact serialized outbound bytes.

    Timing begins after setup negotiation. Model loading is measured separately.
    Byte counts increase only after the corresponding send operation succeeds.
    """

    sample_rate: int
    hop_size: int
    started_at: float

    input_samples: int = 0
    encoded_frames: int = 0
    chunk_count: int = 0

    encoder_seconds: float = 0.0
    longest_encoder_call_seconds: float = 0.0
    packet_build_seconds: float = 0.0
    send_seconds: float = 0.0

    content_bytes: int = 0
    chunk_bytes: int = 0
    setup_bytes: int = 0
    end_bytes: int = 0
    received_control_bytes: int = 0

    maximum_send_lag_seconds: float = 0.0
    final_send_lag_seconds: float = 0.0
    first_chunk_seconds: float | None = None
    end_sent_at: float | None = None
    completed_at: float | None = None

    def record_encoder_call(self, seconds: float) -> None:
        """Accumulate one frontend-and-codec call, including empty-frame calls."""

        self.encoder_seconds += seconds
        self.longest_encoder_call_seconds = max(
            self.longest_encoder_call_seconds,
            seconds,
        )

    def record_chunk(
        self,
        chunk: EncodedChunk,
        packed_bytes: int,
        *,
        build_seconds: float,
        send_seconds: float,
        sent_at: float,
    ) -> None:
        """Account for a successfully submitted encoded chunk and its framing."""

        self.chunk_count += 1
        self.encoded_frames += chunk.frames
        self.content_bytes += packed_bytes - _CHUNK_HEADER_BYTES
        self.chunk_bytes += _MESSAGE_HEADER_BYTES + packed_bytes
        self.packet_build_seconds += build_seconds
        self.send_seconds += send_seconds

        if self.first_chunk_seconds is None:
            self.first_chunk_seconds = sent_at - self.started_at

    def observe_send_lag(self, now: float) -> None:
        """Compare sender progress with the duration of PCM already consumed."""

        audio_seconds = self.input_samples / self.sample_rate
        lag = max(0.0, now - self.started_at - audio_seconds)
        self.maximum_send_lag_seconds = max(self.maximum_send_lag_seconds, lag)
        self.final_send_lag_seconds = lag

    def report(
        self,
        *,
        setup_seconds: float,
        codec_load_seconds: float,
    ) -> None:
        """Print measured durations, byte counts, and exact application bitrates."""

        duration = self.input_samples / self.sample_rate
        if duration <= 0 or self.end_sent_at is None or self.completed_at is None:
            raise ValueError("A complete nonempty stream is required for reporting.")

        outbound_bytes = self.setup_bytes + self.chunk_bytes + self.end_bytes
        sent_seconds = self.end_sent_at - self.started_at
        completed_seconds = self.completed_at - self.started_at

        print("\nSender measurements")
        print(f"  Setup preparation:            {setup_seconds:.6f} s")
        print(f"  Codec loading:                {codec_load_seconds:.6f} s")
        print(f"  Input PCM samples:            {self.input_samples}")
        print(f"  Input PCM duration:           {duration:.6f} s")
        print(f"  Encoded Mel frames:           {self.encoded_frames}")
        print(
            "  Encoded-frame duration:      "
            f"{self.encoded_frames * self.hop_size / self.sample_rate:.6f} s"
        )
        print(f"  Chunks sent:                  {self.chunk_count}")
        print(f"  Frontend + codec time:        {self.encoder_seconds:.6f} s")
        print(f"  Encoder time / audio time:    {self.encoder_seconds / duration:.6f}")
        print(
            f"  Longest encoder call:        {self.longest_encoder_call_seconds:.6f} s"
        )
        print(f"  Packet construction time:    {self.packet_build_seconds:.6f} s")
        print(f"  TCP sendall time:             {self.send_seconds:.6f} s")
        print(f"  First chunk submitted at:    {self.first_chunk_seconds:.6f} s")
        print(f"  Last byte submitted at:      {sent_seconds:.6f} s")
        print(f"  Receiver finished at:        {completed_seconds:.6f} s")
        print(f"  Maximum sender lag:          {self.maximum_send_lag_seconds:.6f} s")
        print(f"  Final sender lag:            {self.final_send_lag_seconds:.6f} s")

        print("\nExact application-layer bytes")
        print(f"  Packed content:              {self.content_bytes}")
        print(f"  All CHUNK messages:          {self.chunk_bytes}")
        print(f"  SETUP message:               {self.setup_bytes}")
        print(f"  END message:                 {self.end_bytes}")
        print(f"  Total outbound:              {outbound_bytes}")
        print(f"  Control received from peer: {self.received_control_bytes}")

        print("\nBitrates over actual input PCM duration")
        print(
            "  Packed content:              "
            f"{self.content_bytes * 8 / duration:.3f} bit/s"
        )
        print(
            "  All CHUNK messages:          "
            f"{self.chunk_bytes * 8 / duration:.3f} bit/s"
        )
        print(
            f"  Complete outbound stream:    {outbound_bytes * 8 / duration:.3f} bit/s"
        )
