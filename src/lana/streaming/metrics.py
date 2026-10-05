from dataclasses import dataclass

from lana.streaming.session import EncodedChunk
from lana.streaming.wire import CHUNK_HEADER_BYTES, MESSAGE_HEADER_BYTES


@dataclass(slots=True)
class SendMetrics:
    """Measure encoding work and successful application-layer transmissions."""

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
        self.content_bytes += packed_bytes - CHUNK_HEADER_BYTES
        self.chunk_bytes += MESSAGE_HEADER_BYTES + packed_bytes
        self.packet_build_seconds += build_seconds
        self.send_seconds += send_seconds

        if self.first_chunk_seconds is None:
            self.first_chunk_seconds = sent_at - self.started_at

    def observe_send_lag(self, now: float) -> None:
        """Compare sender progress with the duration of PCM already consumed."""

        duration = self.input_samples / self.sample_rate
        lag = max(0.0, now - self.started_at - duration)
        self.maximum_send_lag_seconds = max(self.maximum_send_lag_seconds, lag)
        self.final_send_lag_seconds = lag

    def report(self, *, setup_seconds: float, codec_load_seconds: float) -> None:
        """Report exact application bytes and elapsed sender-side measurements."""

        duration = self.input_samples / self.sample_rate
        if (
            duration <= 0
            or self.end_sent_at is None
            or self.completed_at is None
            or self.first_chunk_seconds is None
        ):
            raise ValueError("A complete nonempty stream is required.")

        outbound = self.setup_bytes + self.chunk_bytes + self.end_bytes
        timings = {
            "Setup preparation": setup_seconds,
            "Codec loading": codec_load_seconds,
            "Input duration": duration,
            "Frontend and codec": self.encoder_seconds,
            "Longest encoder call": self.longest_encoder_call_seconds,
            "Packet construction": self.packet_build_seconds,
            "TCP submission": self.send_seconds,
            "First chunk submitted": self.first_chunk_seconds,
            "Last byte submitted": self.end_sent_at - self.started_at,
            "Receiver completion": self.completed_at - self.started_at,
            "Maximum sender lag": self.maximum_send_lag_seconds,
            "Final sender lag": self.final_send_lag_seconds,
        }
        print("\nSender measurements")
        for name, value in timings.items():
            print(f"  {name:<26} {value:.6f} s")
        print(f"  Encoder real-time factor:  {self.encoder_seconds / duration:.6f}")
        print(f"  Input samples:             {self.input_samples}")
        print(f"  Encoded Mel frames:        {self.encoded_frames}")
        print(f"  Chunks:                    {self.chunk_count}")

        print("\nApplication-layer bytes")
        for name, value in {
            "Packed content": self.content_bytes,
            "CHUNK messages": self.chunk_bytes,
            "SETUP message": self.setup_bytes,
            "END message": self.end_bytes,
            "Complete outbound stream": outbound,
            "Received control messages": self.received_control_bytes,
        }.items():
            print(f"  {name:<26} {value}")

        print("\nBitrates over input PCM duration")
        for name, value in {
            "Packed content": self.content_bytes,
            "CHUNK messages": self.chunk_bytes,
            "Complete outbound stream": outbound,
        }.items():
            print(f"  {name:<26} {value * 8 / duration:.3f} bit/s")
