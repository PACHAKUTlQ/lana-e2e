from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Normalization(StrEnum):
    """Normalization modes supported by the acoustic feature extractors."""

    RAW = "raw"
    MINMAX = "minmax"
    ZSCORE = "z-score"


class Feature(StrEnum):
    """Acoustic features exposed by the public extraction interface."""

    MEL = "mel"
    LINEAR = "linear"
    LF0 = "lf0"
    UV = "uv"
    ENERGY = "energy"


class FrozenConfig(BaseModel):
    """Immutable configuration with validated defaults and explicit fields."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class ModelPaths(FrozenConfig):
    """TorchScript model locations, resolved relative to the configuration file."""

    codec: Path
    mel2f0: Path
    generator: Path
    prior_generator: Path

    def resolve(self, directory: Path) -> Self:
        """Resolve relative model paths without requiring the files to exist."""

        return self.model_copy(
            update={
                name: (directory / getattr(self, name)).resolve()
                for name in type(self).model_fields
            }
        )


class AudioConfig(FrozenConfig):
    """Waveform preprocessing and acoustic feature extraction parameters.

    The defaults reproduce the original 16 kHz frontend. Mel and linear
    spectrograms use a centered Hann-window STFT and Slaney mel filters.
    """

    sample_rate: int = Field(default=16000, gt=0)
    n_fft: int = Field(default=1024, gt=0)
    hop_size: int = Field(default=200, gt=0)
    win_size: int = Field(default=800, gt=0)
    fmin: float = Field(default=0.0, ge=0)
    fmax: float | None = Field(default=8000.0, gt=0)
    n_mels: int = Field(default=80, gt=0)
    stft_pad_mode: Literal["constant", "reflect"] = "constant"

    allow_upsample: bool = False
    resample_type: str = "kaiser_best"
    rescale: bool = False
    rescaling_max: float = Field(default=0.999, gt=0, le=1)

    trim_silence: bool = True
    trim_top_db: float = Field(default=30.0, ge=0)
    num_silent_frames: int = Field(default=8, ge=0)
    trim_fft_size: int = Field(default=512, gt=0)
    trim_hop_size: int = Field(default=128, gt=0)
    pad_sli: bool = False
    pad_sli_frame: int = Field(default=0, ge=0)

    min_level_db: float = Field(default=-115.0, lt=0)
    ref_level_db: float = 20.0
    max_abs_value: float = Field(default=1.0, gt=0)
    min_db: float = Field(default=-115.0, lt=0)
    signal_normalization: bool = True
    allow_clipping_in_normalization: bool = True
    symmetric_acoustic: bool = True

    lf0_inter: bool = False
    lf0_norm: Normalization = Normalization.RAW
    energy_norm: Normalization = Normalization.RAW

    @model_validator(mode="after")
    def validate_spectral_parameters(self) -> Self:
        """Validate relationships between FFT, window, and frequency settings."""

        if self.win_size > self.n_fft:
            raise ValueError("audio.win_size must not exceed audio.n_fft.")

        upper_frequency = self.fmax or self.sample_rate / 2
        if not self.fmin < upper_frequency <= self.sample_rate / 2:
            raise ValueError(
                "Audio frequencies must satisfy 0 <= fmin < fmax <= sample_rate / 2."
            )

        return self


class EncodingConfig(FrozenConfig):
    """Prompt length and random seed used by whole-utterance encoding."""

    prompt_length: int = Field(default=600, gt=0)
    seed: int | None = None


class InferenceConfig(FrozenConfig):
    """Chunking, prompt history, speaker smoothing, and Mel overlap parameters.

    Short final chunks and the final overlap cache are discarded, matching
    the original inference implementation.
    """

    chunk_size: int = Field(default=80, gt=0)
    prompt_length: int = Field(default=600, gt=0)
    new_emb_weight: float = Field(default=0.2, ge=0, le=1)
    mel_cache_length: int = Field(default=0, ge=0)
    minimum_chunk_frames: int = Field(default=10, gt=0)

    @model_validator(mode="after")
    def validate_chunk_parameters(self) -> Self:
        """Reject overlap and minimum-length settings incompatible with chunking."""

        if self.mel_cache_length > self.chunk_size:
            raise ValueError("mel_cache_length must not exceed chunk_size.")
        if self.minimum_chunk_frames > self.chunk_size:
            raise ValueError("minimum_chunk_frames must not exceed chunk_size.")

        return self


class VocoderConfig(FrozenConfig):
    """DSPGAN output and intermediate Mel analysis parameters."""

    sampling_rate: int = Field(default=48000, gt=0)
    max_wav_value: float = Field(default=32768.0, gt=0)
    mel2f0_scale: float = Field(default=4.0, gt=0)

    mel_sampling_rate: int = Field(default=16000, gt=0)
    mel_filter_length: int = Field(default=1024, gt=0)
    mel_hop_length: int = Field(default=200, gt=0)
    mel_win_length: int = Field(default=800, gt=0)
    n_mel_channels: int = Field(default=80, gt=0)
    mel_fmin: float = Field(default=0.0, ge=0)
    mel_fmax: float | None = Field(default=8000.0, gt=0)

    min_db: float = Field(default=-115.0, lt=0)
    max_abs_value: float = Field(default=1.0, gt=0)
    min_level_db: float = Field(default=-115.0, lt=0)
    ref_level_db: float = 20.0

    @model_validator(mode="after")
    def validate_spectral_parameters(self) -> Self:
        """Validate the centered STFT and mel-filter frequency range."""

        if self.mel_win_length > self.mel_filter_length:
            raise ValueError(
                "vocoder.mel_win_length must not exceed mel_filter_length."
            )

        upper_frequency = self.mel_fmax or self.mel_sampling_rate / 2
        if not self.mel_fmin < upper_frequency <= self.mel_sampling_rate / 2:
            raise ValueError(
                "Vocoder frequencies must satisfy "
                "0 <= mel_fmin < mel_fmax <= mel_sampling_rate / 2."
            )

        return self


class StreamingConfig(FrozenConfig):
    """Chunk and prompt settings for continuous audio transmission.

    The encoder waits for half an FFT window of future samples before emitting
    a Mel frame. A final short chunk is padded for model inference, while its
    transmitted duration remains the number of real Mel frames.
    """

    chunk_frames: int = Field(default=80, gt=0, le=65535)
    prompt_frames: int = Field(default=600, gt=0)
    minimum_model_frames: int = Field(default=10, gt=0)
    new_embedding_weight: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def validate_chunking(self) -> Self:
        """Keep the minimum model input within a normal transmitted chunk."""

        if self.minimum_model_frames > self.chunk_frames:
            raise ValueError(
                "streaming.minimum_model_frames must not exceed chunk_frames."
            )

        return self


class Config(FrozenConfig):
    """Complete application configuration with independently loadable models."""

    device: str = "cpu"
    models: ModelPaths
    audio: AudioConfig = Field(default_factory=AudioConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    vocoder: VocoderConfig = Field(default_factory=VocoderConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)

    @classmethod
    def from_file(cls, path: str | Path = "config.json") -> Self:
        """Read JSON and resolve model paths relative to its parent directory."""

        config_path = Path(path).expanduser().resolve()
        config = cls.model_validate_json(config_path.read_text(encoding="utf-8"))

        return config.model_copy(
            update={"models": config.models.resolve(config_path.parent)}
        )
