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
        """Create independent per-stream state while sharing optionally loaded models."""

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
            raise RuntimeError("The stream encoder has already been finished.")

        return self._consume(self.frontend.push(samples), final=False)

    def finish(self) -> list[EncodedChunk]:
        """Flush right-context Mel frames and encode the final partial chunk."""

        if self._finished:
            raise RuntimeError("The stream encoder has already been finished.")

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
        chunk_frames = self.config.streaming.chunk_frames

        while self._pending.shape[1] >= chunk_frames:
            result.append(self._encode(self._pending[:, :chunk_frames]))
            self._pending = self._pending[:, chunk_frames:]

        if final and self._pending.shape[1]:
            result.append(self._encode(self._pending))
            self._pending = np.empty(
                (self.config.audio.n_mels, 0),
                dtype=np.float32,
            )

        return result

    @torch.inference_mode()
    def _encode(self, mel: NDArray[np.float32]) -> EncodedChunk:
        """Encode one chunk with bounded prompt history and smoothed embedding."""

        settings = self.config.streaming
        frames = mel.shape[1]
        minimum = settings.minimum_model_frames

        if frames < minimum:
            mel = np.pad(
                mel,
                ((0, 0), (0, minimum - frames)),
                mode="edge",
            )

        tensor = torch.from_numpy(np.ascontiguousarray(mel))
        tensor = ((tensor.to(self.models.device) + 1.0) / 2.0).unsqueeze(0)

        if self._prompt is None:
            self._prompt = tensor
        else:
            self._prompt = torch.cat((self._prompt, tensor), dim=-1)
        self._prompt = self._prompt[:, :, -settings.prompt_frames :]

        current_embedding = self.models.codec.ref_encoder(self._prompt)
        if self._embedding is None:
            self._embedding = current_embedding
        else:
            weight = settings.new_embedding_weight
            self._embedding = (
                self._embedding * (1.0 - weight) + current_embedding * weight
            )

        _, speaker_codes = self.models.codec.refemb_fsq(self._embedding)
        _, content_codes, _ = self.models.codec.encode_ref(
            tensor,
            self._embedding,
        )

        content = np.asarray(
            content_codes.detach().cpu().numpy(),
            dtype=np.int64,
        )
        speaker = speaker_codes.detach().cpu().numpy()

        if content.ndim != 2 or content.shape[0] != 1 or content.shape[1] == 0:
            raise ValueError("The codec returned invalid content-code dimensions.")
        if speaker.shape != (1, 1):
            raise ValueError("The codec returned invalid speaker-code dimensions.")

        packet = EncodedChunk(
            sequence=self._sequence,
            frames=frames,
            content=np.ascontiguousarray(content[0]),
            speaker=int(speaker[0, 0]),
        )
        self._sequence += 1

        return packet


class StreamDecoder:
    """Decode ordered chunks independently of encoder prompt history."""

    def __init__(self, config: Config, models: Models | None = None) -> None:
        """Prepare a decoder with its own expected packet sequence."""

        self.config = config
        self.models = models if models is not None else Models(config)
        self._next_sequence = 0

    @torch.inference_mode()
    def push(self, chunk: EncodedChunk) -> Audio:
        """Decode one chunk and retain exactly its declared audio duration."""

        if chunk.sequence != self._next_sequence:
            raise ValueError(
                f"Expected chunk {self._next_sequence}, got {chunk.sequence}."
            )
        if not 1 <= chunk.frames <= self.config.streaming.chunk_frames:
            raise ValueError("Chunk frame count is outside the configured range.")
        if chunk.content.ndim != 1 or chunk.content.size == 0:
            raise ValueError("Chunk contains no content codes.")

        device = self.models.device
        content = torch.from_numpy(chunk.content.reshape(1, -1)).to(device)
        speaker = torch.tensor([[chunk.speaker]], dtype=torch.int64, device=device)

        decoded = self.models.codec.decode(content, speaker)
        mel = canonical_mel(
            decoded * 2.0 - 1.0,
            self.config.audio.n_mels,
        )
        if mel.shape[1] < chunk.frames:
            raise ValueError("The codec decoded fewer frames than the packet declares.")

        audio = self.models.vocoder.decode(mel[:, : chunk.frames])
        expected_samples = round(
            chunk.frames
            * self.config.audio.hop_size
            * self.config.vocoder.sampling_rate
            / self.config.audio.sample_rate
        )
        if audio.samples.size < expected_samples:
            raise ValueError(
                "The vocoder returned less audio than the packet duration."
            )

        self._next_sequence += 1

        return Audio(
            samples=audio.samples[:expected_samples],
            sample_rate=audio.sample_rate,
        )
