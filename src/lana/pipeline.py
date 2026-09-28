import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from lana.artifacts import Audio, FeatureArray, FloatArray, Tokens
from lana.config import Config, Feature
from lana.extraction import discover_audio, extract_batch
from lana.features import extract_file, linear_to_mel
from lana.models import Models, canonical_mel


class Lana:
    """Unified API for feature extraction, codec operations, and DSPGAN synthesis.

    Models are loaded only when an operation needs them. Each instance retains
    its encoding RNG state; chunk prompt and overlap caches are local to each
    inference call.
    """

    def __init__(self, config: Config | str | Path = "config.json") -> None:
        """Accept a validated configuration or load one from a JSON file."""

        self.config = config if isinstance(config, Config) else Config.from_file(config)
        self.models = Models(self.config)
        self._random = random.Random(self.config.encoding.seed)

    def extract(
        self,
        input_path: str | Path,
        features: Iterable[Feature] = (Feature.MEL,),
    ) -> dict[Feature, FeatureArray]:
        """Extract acoustic features from one file without loading neural models."""

        return extract_file(input_path, self.config.audio, features)

    def extract_many(
        self,
        source: str | Path,
        output_directory: str | Path,
        features: Iterable[Feature] = (Feature.MEL,),
        *,
        workers: int = 20,
        incremental: bool = False,
        progress: bool = False,
    ) -> list[dict[Feature, Path]]:
        """Extract a WAV file, flat directory, or manifest into NPY artifacts."""

        paths = discover_audio(source)

        return extract_batch(
            paths,
            output_directory,
            self.config.audio,
            features,
            workers=workers,
            incremental=incremental,
            progress=progress,
        )

    def convert_linear_to_mel(self, spectrogram: FloatArray) -> FeatureArray:
        """Convert a frontend-scaled linear spectrogram to frontend-scaled Mel."""

        return linear_to_mel(spectrogram, self.config.audio)

    def _codec_input(self, input_path: str | Path) -> Tensor:
        """Extract Mel features and apply the codec's fixed [-1, 1] to [0, 1] map."""

        audio_config = self.config.audio
        if (
            not audio_config.signal_normalization
            or not audio_config.symmetric_acoustic
            or audio_config.max_abs_value != 1.0
        ):
            raise ValueError(
                "Codec input requires symmetric Mel normalization with max_abs_value=1."
            )

        mel = self.extract(input_path)[Feature.MEL]
        tensor = torch.from_numpy(mel).unsqueeze(0).to(self.models.device)

        return (tensor + 1) / 2

    @torch.inference_mode()
    def encode(self, input_path: str | Path) -> Tokens:
        """Encode a full utterance using a randomly cropped or repeated prompt."""

        mel = self._codec_input(input_path)
        prompt_length = self.config.encoding.prompt_length
        frame_count = mel.shape[-1]

        if frame_count >= prompt_length:
            start = self._random.randint(0, frame_count - prompt_length)
            prompt = mel[:, :, start : start + prompt_length]
        else:
            repetitions = (prompt_length + frame_count - 1) // frame_count
            prompt = mel.tile((1, 1, repetitions))[:, :, :prompt_length]

        _, codes, _, _, speaker_codes = self.models.codec.encode(mel, prompt)

        return Tokens(
            content=codes.detach().cpu().numpy(),
            speaker=speaker_codes.detach().cpu().numpy(),
        )

    @torch.inference_mode()
    def decode_tokens(self, tokens: Tokens) -> Tensor:
        """Decode code arrays to a device-resident Mel in the [-1, 1] domain."""

        codes = torch.from_numpy(tokens.content).to(self.models.device)
        speaker_codes = torch.from_numpy(tokens.speaker).to(self.models.device)
        mel = self.models.codec.decode(codes, speaker_codes)
        mel = canonical_mel(mel * 2 - 1, self.config.audio.n_mels)

        return mel

    @torch.inference_mode()
    def decode_mel(self, mel: Tensor | FloatArray) -> Audio:
        """Synthesize one normalized Mel spectrogram through DSPGAN."""

        tensor = (
            mel
            if isinstance(mel, Tensor)
            else torch.as_tensor(mel, dtype=torch.float32)
        )

        return self.models.vocoder.decode(tensor)

    @torch.inference_mode()
    def decode(self, tokens: Tokens) -> Audio:
        """Decode content and speaker codes directly into PCM16 audio."""

        return self.decode_mel(self.decode_tokens(tokens))

    @torch.inference_mode()
    def infer(
        self,
        input_path: str | Path,
        *,
        progress: bool = False,
    ) -> Audio:
        """Run the original chunked codec and vocoder inference sequence.

        Speaker embeddings are exponentially blended, while prompt history uses
        the overlapping input chunks. Mel caches store already weighted tails.
        """

        config = self.config.inference
        mel_input = self._codec_input(input_path)
        prompt_cache: Tensor | None = None
        speaker_embedding: Tensor | None = None
        mel_cache: Tensor | None = None
        audio_segments: list[np.ndarray[tuple[int, ...], np.dtype[np.int16]]] = []

        positions = range(0, mel_input.shape[-1], config.chunk_size)
        for start in tqdm(
            positions,
            desc="Inferring",
            disable=not progress,
        ):
            mel_chunk = mel_input[
                :,
                :,
                start : start + config.chunk_size + config.mel_cache_length,
            ]
            if mel_chunk.shape[-1] < config.minimum_chunk_frames:
                break

            if prompt_cache is None:
                prompt_cache = mel_chunk
            else:
                prompt_cache = torch.cat((prompt_cache, mel_chunk), dim=-1)
                prompt_cache = prompt_cache[:, :, -config.prompt_length :]

            current_embedding = self.models.codec.ref_encoder(prompt_cache)
            if speaker_embedding is None:
                speaker_embedding = current_embedding
            else:
                speaker_embedding = (
                    speaker_embedding * (1 - config.new_emb_weight)
                    + current_embedding * config.new_emb_weight
                )

            _, speaker_codes = self.models.codec.refemb_fsq(speaker_embedding)
            _, codes, _ = self.models.codec.encode_ref(
                mel_chunk,
                speaker_embedding,
            )
            decoded = self.models.codec.decode(codes, speaker_codes)
            decoded = canonical_mel(
                decoded * 2 - 1,
                self.config.audio.n_mels,
            )

            overlap_length = config.mel_cache_length
            if overlap_length:
                if decoded.shape[-1] <= overlap_length:
                    raise ValueError(
                        "Decoded chunk must contain more frames than mel_cache_length."
                    )

                if mel_cache is not None:
                    rising = torch.linspace(
                        0,
                        1,
                        overlap_length,
                        device=decoded.device,
                        dtype=decoded.dtype,
                    ).unsqueeze(0)
                    overlap = decoded[:, :overlap_length] * rising + mel_cache
                    decoded = torch.cat(
                        (overlap, decoded[:, overlap_length:]),
                        dim=-1,
                    )

                falling = torch.linspace(
                    1,
                    0,
                    overlap_length,
                    device=decoded.device,
                    dtype=decoded.dtype,
                ).unsqueeze(0)
                mel_cache = decoded[:, -overlap_length:] * falling
                decoded = decoded[:, :-overlap_length]

            audio_segments.append(self.decode_mel(decoded).samples)

        if not audio_segments:
            raise ValueError(
                "No audio was generated; the input may be shorter than "
                "minimum_chunk_frames."
            )

        samples = np.concatenate(audio_segments, axis=0)

        return Audio(samples, self.config.vocoder.sampling_rate)
