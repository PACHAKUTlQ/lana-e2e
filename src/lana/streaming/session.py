from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from lana.artifacts import Audio
from lana.config import Config
from lana.runtime.models import Models, canonical_mel
from lana.streaming.frontend import PcmChunk, StreamingMelFrontend

type ContentCodes = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class EncodedChunk:
    """One ordered segment of encoded speech and its real Mel-frame duration."""

    sequence: int
    frames: int
    content: ContentCodes
    speaker: int


class StreamEncoder:
    """Keep bounded frontend, prompt, and speaker state for one audio stream.

    Model calls follow the legacy chunked inference interface. Every packet
    carries speaker codes so decoding does not require synchronized encoder
    history. The final short Mel chunk is repeated to the minimum model length,
    but its real duration is retained in the packet.
    """

    def __init__(self, config: Config, models: Models | None = None) -> None:
        """Create per-session frontend, prompt, embedding, and sequence state."""

        config.validate_codec_frontend()
        self.config = config
        self.models = models if models is not None else Models(config)
        self.frontend = StreamingMelFrontend(config.audio)
        self._pending = np.empty((config.audio.n_mels, 0), dtype=np.float32)
        self._prompt: Tensor | None = None
        self._embedding: Tensor | None = None
        self._sequence = 0
        self._finished = False

    def push(self, samples: PcmChunk) -> list[EncodedChunk]:
        """Consume PCM samples and encode every newly completed normal chunk."""

        if self._finished:
            raise RuntimeError("The encoder has already been finished.")

        return self._consume(self.frontend.push(samples), final=False)

    def finish(self) -> list[EncodedChunk]:
        """Flush right-context Mel frames and encode the final partial chunk."""

        if self._finished:
            raise RuntimeError("The encoder has already been finished.")
        self._finished = True

        return self._consume(self.frontend.finish(), final=True)

    def _consume(
        self,
        mel: NDArray[np.float32],
        *,
        final: bool,
    ) -> list[EncodedChunk]:
        """Accumulate Mel frames and dispatch complete or final partial chunks."""

        if mel.shape[1]:
            self._pending = np.concatenate((self._pending, mel), axis=1)
        result: list[EncodedChunk] = []
        frames = self.config.streaming.chunk_frames

        while self._pending.shape[1] >= frames:
            result.append(self._encode(self._pending[:, :frames]))
            self._pending = self._pending[:, frames:].copy()

        if final and self._pending.shape[1]:
            result.append(self._encode(self._pending))
            self._pending = np.empty((self.config.audio.n_mels, 0), dtype=np.float32)

        return result

    @torch.inference_mode()
    def _encode(self, mel: NDArray[np.float32]) -> EncodedChunk:
        """Encode one chunk with bounded prompt history and smoothed embedding."""

        settings = self.config.streaming
        frames = mel.shape[1]
        if frames < settings.chunk_frames:
            mel = np.pad(
                mel,
                ((0, 0), (0, settings.chunk_frames - frames)),
                mode="edge",
            )

        tensor = torch.from_numpy(np.ascontiguousarray(mel)).to(self.models.device)
        tensor = ((tensor + 1.0) / 2.0).unsqueeze(0)
        self._prompt = (
            tensor
            if self._prompt is None
            else torch.cat((self._prompt, tensor), dim=-1)
        )
        self._prompt = self._prompt[:, :, -settings.prompt_frames :].contiguous()

        current = self.models.codec.ref_encoder(self._prompt)
        if self._embedding is None:
            self._embedding = current
        else:
            weight = settings.new_embedding_weight
            self._embedding = self._embedding * (1.0 - weight) + current * weight

        _, speaker_codes = self.models.codec.refemb_fsq(self._embedding)
        _, content_codes, _ = self.models.codec.encode_ref(tensor, self._embedding)
        content = content_codes.detach().cpu().numpy()
        speaker = speaker_codes.detach().cpu().numpy()

        if (
            content.dtype != np.int64
            or content.ndim != 3
            or content.shape[0] != 1
            or content.shape[1] == 0
            or content.shape[2] != self.config.codec.num_quantizers
        ):
            raise ValueError("The codec returned invalid layered content codes.")
        if speaker.dtype != np.int64 or speaker.shape != (1, 1):
            raise ValueError("The codec returned invalid speaker codes.")

        chunk = EncodedChunk(
            sequence=self._sequence,
            frames=frames,
            content=np.ascontiguousarray(content[0]),
            speaker=int(speaker[0, 0]),
        )
        self._sequence += 1

        return chunk


class StreamDecoder:
    """Decode ordered chunks independently of encoder prompt history."""

    def __init__(self, config: Config, models: Models | None = None) -> None:
        """Prepare a decoder with its own expected packet sequence."""

        self.config = config
        self.models = models if models is not None else Models(config)
        self._next_sequence = 0
        self._frames = 0
        self._samples = 0
        self._partial_received = False

    @torch.inference_mode()
    def push(self, chunk: EncodedChunk) -> Audio:
        """Decode one chunk and retain exactly its declared audio duration."""

        settings = self.config.streaming
        if self._partial_received:
            raise ValueError("A partial block must be the final audio block.")
        if chunk.sequence != self._next_sequence:
            raise ValueError(
                f"Expected chunk {self._next_sequence}, got {chunk.sequence}."
            )
        if not 1 <= chunk.frames <= settings.chunk_frames:
            raise ValueError("Invalid chunk frame count.")
        if (
            chunk.content.dtype != np.int64
            or chunk.content.ndim != 2
            or chunk.content.shape[0] == 0
            or chunk.content.shape[1] != self.config.codec.num_quantizers
        ):
            raise ValueError("Invalid layered content dimensions.")
        if np.any(chunk.content < 0) or chunk.speaker < 0:
            raise ValueError("Quantizer indices must be nonnegative.")

        device = self.models.device
        content = torch.from_numpy(np.ascontiguousarray(chunk.content[None])).to(device)
        speaker = torch.tensor([[chunk.speaker]], dtype=torch.int64, device=device)
        decoded = self.models.codec.decode(content, speaker)
        mel = canonical_mel(decoded * 2.0 - 1.0, self.config.audio.n_mels)

        if mel.shape[1] != settings.chunk_frames:
            raise ValueError(
                "Decoded block length differs from the trained block size."
            )

        audio = self.models.vocoder.decode(mel)
        total_frames = self._frames + chunk.frames
        sample_rate = self.config.vocoder.sampling_rate
        total_samples = (
            total_frames
            * self.config.audio.hop_size
            * sample_rate
            // self.config.audio.sample_rate
        )
        expected = total_samples - self._samples
        if audio.samples.size < expected:
            raise ValueError("Vocoder output is shorter than the declared duration.")

        self._frames = total_frames
        self._samples = total_samples
        self._next_sequence += 1
        self._partial_received = chunk.frames < settings.chunk_frames

        return Audio(audio.samples[:expected], sample_rate)
