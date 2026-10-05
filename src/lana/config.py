import math
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
                name: (directory / getattr(self, name).expanduser()).resolve()
                for name in type(self).model_fields
            }
        )


class AudioConfig(FrozenConfig):
    """Waveform preprocessing and acoustic feature extraction parameters.

    Streaming and training consume mono PCM16 without utterance-level trimming
    or peak normalization. The preprocessing flags apply to offline extraction.
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

    trim_silence: bool = False
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
    def validate_spectrum(self) -> Self:
        """Validate FFT dimensions and the Mel filter frequency interval."""

        if self.win_size > self.n_fft:
            raise ValueError("audio.win_size must not exceed n_fft.")
        upper = self.sample_rate / 2 if self.fmax is None else self.fmax
        if not self.fmin < upper <= self.sample_rate / 2:
            raise ValueError("Invalid audio Mel frequency interval.")

        return self


class EncodingConfig(FrozenConfig):
    """Prompt length and random seed used by whole-utterance encoding."""

    prompt_length: int = Field(default=600, gt=0)
    seed: int | None = None


class InferenceConfig(FrozenConfig):
    """Settings for the existing offline overlapping-chunk inference interface."""

    chunk_size: int = Field(default=80, gt=0)
    prompt_length: int = Field(default=600, gt=0)
    new_emb_weight: float = Field(default=0.2, ge=0, le=1)
    mel_cache_length: int = Field(default=0, ge=0)
    minimum_chunk_frames: int = Field(default=10, gt=0)

    @model_validator(mode="after")
    def validate_chunks(self) -> Self:
        """Reject overlap and minimum lengths exceeding a normal chunk."""

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
    def validate_spectrum(self) -> Self:
        """Validate FFT dimensions and the prior Mel frequency interval."""

        if self.mel_win_length > self.mel_filter_length:
            raise ValueError("mel_win_length must not exceed mel_filter_length.")
        upper = self.mel_sampling_rate / 2 if self.mel_fmax is None else self.mel_fmax
        if not self.mel_fmin < upper <= self.mel_sampling_rate / 2:
            raise ValueError("Invalid vocoder Mel frequency interval.")

        return self


class CodecConfig(FrozenConfig):
    """Select a prefix of the trained residual quantizers for inference."""

    num_quantizers: int = Field(default=1, ge=1, le=16)


class StreamingConfig(FrozenConfig):
    """Chunk and prompt settings for continuous audio transmission.

    The encoder waits for half an FFT window of future samples before emitting
    a Mel frame. A final short chunk is padded for model inference, while its
    transmitted duration remains the number of real Mel frames.
    """

    chunk_frames: int = Field(default=80, ge=2, le=65535)
    prompt_frames: int = Field(default=600, ge=2)
    new_embedding_weight: float = Field(default=0.2, ge=0, le=1)
    io_timeout_seconds: float = Field(default=120.0, gt=0)


class CodecArchitecture(FrozenConfig):
    """Shared neural architecture and residual scalar quantizer definitions."""

    hidden_channels: int = Field(default=128, ge=8)
    latent_channels: int = Field(default=32, ge=4)
    encoder_layers: int = Field(default=4, ge=1)
    decoder_layers: int = Field(default=6, ge=1)
    speaker_layers: int = Field(default=2, ge=1)
    kernel_size: int = Field(default=5, ge=3)
    frame_stride: int = Field(default=2, ge=1, le=16)
    num_quantizers: int = Field(default=4, ge=1, le=16)
    content_levels: tuple[int, ...] = (8, 8, 8)
    speaker_levels: tuple[int, ...] = (8, 8, 8)

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        """Keep padding symmetric and quantizer indices within supported bounds."""

        if self.kernel_size % 2 == 0:
            raise ValueError("architecture.kernel_size must be odd.")
        for name in ("content_levels", "speaker_levels"):
            levels = getattr(self, name)
            if not levels or any(level < 2 or level > 256 for level in levels):
                raise ValueError(f"{name} must contain levels between 2 and 256.")
            if math.prod(levels) > 2**31:
                raise ValueError(f"{name} must define at most 2**31 entries.")

        return self

    @property
    def content_bits(self) -> int:
        """Return the fixed width of one index from one residual stage."""

        return (math.prod(self.content_levels) - 1).bit_length()


class TrainingDataConfig(FrozenConfig):
    """LJSpeech paths and fixed-length sequences of consecutive stream blocks."""

    directory: Path = Path("dataset/LJSpeech-1.1")
    cache_directory: Path = Path("outputs/cache/streaming-mels")
    validation_fraction: float = Field(default=0.02, gt=0, lt=1)
    sequence_chunks: int = Field(default=8, ge=1)
    batch_size: int = Field(default=8, ge=1)
    workers: int = Field(default=4, ge=0)


class OptimizationConfig(FrozenConfig):
    """AdamW, cosine scheduling, and multi-rate reconstruction objectives."""

    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=1000, ge=0)
    minimum_lr_ratio: float = Field(default=0.1, gt=0, le=1)
    temporal_weight: float = Field(default=0.25, ge=0)
    base_loss_weight: float = Field(default=0.25, ge=0)
    temporal_scales: tuple[int, ...] = (1, 2, 4)

    @model_validator(mode="after")
    def validate_scales(self) -> Self:
        """Require distinct positive temporal difference intervals."""

        if not self.temporal_scales or any(x < 1 for x in self.temporal_scales):
            raise ValueError("temporal_scales must contain positive integers.")
        if len(set(self.temporal_scales)) != len(self.temporal_scales):
            raise ValueError("temporal_scales must not contain duplicates.")

        return self


class TrainingRuntime(FrozenConfig):
    """Optimizer-step training limits and Lightning execution settings."""

    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: int | list[int] = 1
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"
    max_steps: int = Field(default=100000, ge=1)
    accumulate_grad_batches: int = Field(default=1, ge=1)
    gradient_clip_val: float = Field(default=1.0, gt=0)
    log_every_n_steps: int = Field(default=20, ge=1)
    validation_every_n_epochs: int = Field(default=1, ge=1)
    save_top_k: int = Field(default=2, ge=1)
    deterministic: bool = False
    matmul_precision: Literal["highest", "high", "medium"] = "high"

    @model_validator(mode="after")
    def validate_devices(self) -> Self:
        """Validate device selections and explicit CPU precision."""

        if isinstance(self.devices, int):
            if self.devices < 1:
                raise ValueError("runtime.devices must be positive.")
        elif (
            not self.devices
            or any(index < 0 for index in self.devices)
            or len(set(self.devices)) != len(self.devices)
        ):
            raise ValueError("runtime.devices must contain unique device indices.")
        if self.accelerator == "cpu" and self.precision == "16-mixed":
            raise ValueError("CPU training does not support 16-mixed.")

        return self


class TrainingConfig(FrozenConfig):
    """Training-only settings stored inside the application configuration."""

    output_directory: Path = Path("outputs/training/lana")
    seed: int = Field(default=1234, ge=0)
    data: TrainingDataConfig = Field(default_factory=TrainingDataConfig)
    architecture: CodecArchitecture = Field(default_factory=CodecArchitecture)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    runtime: TrainingRuntime = Field(default_factory=TrainingRuntime)


class Config(FrozenConfig):
    """Complete application configuration with independently loadable models."""

    device: str = "cpu"
    models: ModelPaths
    audio: AudioConfig = Field(default_factory=AudioConfig)
    codec: CodecConfig = Field(default_factory=CodecConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    vocoder: VocoderConfig = Field(default_factory=VocoderConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    training: TrainingConfig | None = None

    @classmethod
    def from_file(cls, path: str | Path = "config.json") -> Self:
        """Load JSON and resolve configured paths against its parent directory."""

        source = Path(path).expanduser().resolve()
        config = cls.model_validate_json(source.read_text(encoding="utf-8"))
        directory = source.parent
        training = config.training

        if training is not None:
            data = training.data.model_copy(
                update={
                    "directory": (
                        directory / training.data.directory.expanduser()
                    ).resolve(),
                    "cache_directory": (
                        directory / training.data.cache_directory.expanduser()
                    ).resolve(),
                }
            )
            training = training.model_copy(
                update={
                    "data": data,
                    "output_directory": (
                        directory / training.output_directory.expanduser()
                    ).resolve(),
                }
            )

        return config.model_copy(
            update={
                "models": config.models.resolve(directory),
                "training": training,
            }
        )

    def validate_codec_frontend(self) -> None:
        """Require the normalized Mel convention used by the neural codec."""

        audio = self.audio
        if (
            not audio.signal_normalization
            or not audio.symmetric_acoustic
            or audio.max_abs_value != 1.0
            or audio.stft_pad_mode != "constant"
        ):
            raise ValueError(
                "The codec requires symmetric normalized Mels, "
                "max_abs_value=1, and constant STFT padding."
            )

    def validate_training(self) -> TrainingConfig:
        """Validate the training architecture against deployed block processing."""

        self.validate_codec_frontend()
        training = self.training
        if training is None:
            raise ValueError("The configuration has no training section.")

        architecture = training.architecture
        if self.streaming.chunk_frames % architecture.frame_stride:
            raise ValueError("chunk_frames must be divisible by frame_stride.")
        if self.codec.num_quantizers > architecture.num_quantizers:
            raise ValueError("codec.num_quantizers exceeds the trained stage count.")
        if training.optimization.warmup_steps >= training.runtime.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps.")
        sequence_frames = self.streaming.chunk_frames * training.data.sequence_chunks
        if max(training.optimization.temporal_scales) >= sequence_frames:
            raise ValueError("Temporal scales must be shorter than a sequence.")

        return training
