import math
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from lana.config import Config, FrozenConfig


class TrainingDataConfig(FrozenConfig):
    """Dataset, feature cache, and fixed-size training crop settings.

    Paths are resolved relative to the training configuration file. Text columns
    in metadata.csv are not used by the audio reconstruction objective.
    """

    directory: Path
    cache_directory: Path
    validation_fraction: float = Field(default=0.02, gt=0, lt=1)
    segment_frames: int = Field(default=320, ge=2)
    prompt_frames: int = Field(default=600, gt=0)
    batch_size: int = Field(default=16, gt=0)
    workers: int = Field(default=4, ge=0)


class CodecArchitecture(FrozenConfig):
    """Architecture and finite scalar quantizer definitions for a new codec.

    Each content position produces one mixed-radix index. The product of
    content_levels is its codebook size. Changing these settings generally
    requires training a separate model.
    """

    n_mels: int = Field(default=80, gt=0)
    hidden_channels: int = Field(default=128, ge=8)
    encoder_layers: int = Field(default=4, ge=1)
    decoder_layers: int = Field(default=6, ge=1)
    speaker_layers: int = Field(default=2, ge=1)
    kernel_size: int = Field(default=5, ge=3)
    frame_stride: int = Field(default=2, ge=1, le=16)
    content_levels: tuple[int, ...] = (8, 8, 8)
    speaker_levels: tuple[int, ...] = (8, 8, 8)

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        """Keep convolution padding symmetric and indices exactly representable."""

        if self.kernel_size % 2 == 0:
            raise ValueError("architecture.kernel_size must be odd.")

        for name, levels in (
            ("content_levels", self.content_levels),
            ("speaker_levels", self.speaker_levels),
        ):
            if not levels or any(level < 2 or level > 256 for level in levels):
                raise ValueError(f"{name} must contain integers between 2 and 256.")
            if math.prod(levels) > 2**31:
                raise ValueError(f"{name} must define at most 2**31 entries.")

        return self

    @property
    def content_codebook_size(self) -> int:
        """Return the number of legal content indices."""

        return math.prod(self.content_levels)

    @property
    def speaker_codebook_size(self) -> int:
        """Return the number of legal speaker indices."""

        return math.prod(self.speaker_levels)

    @property
    def content_bits(self) -> int:
        """Return the minimum fixed width for an individual content index."""

        return (self.content_codebook_size - 1).bit_length()


class OptimizationConfig(FrozenConfig):
    """AdamW, step-based learning-rate scheduling, and reconstruction losses."""

    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=1000, ge=0)
    minimum_lr_ratio: float = Field(default=0.1, gt=0, le=1)
    reconstruction_weight: float = Field(default=1.0, gt=0)
    temporal_weight: float = Field(default=0.25, ge=0)


class TrainingRuntime(FrozenConfig):
    """Lightning execution settings shared by CPU and GPU training.

    Multiple GPU devices use Lightning's distributed strategy selection.
    Full precision is the conservative default; mixed precision is explicit.
    """

    accelerator: Literal["auto", "cpu", "gpu"] = "auto"
    devices: int | list[int] | Literal["auto"] = 1
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"
    max_epochs: int = Field(default=100, gt=0)
    max_steps: int = Field(default=-1, ge=-1)
    accumulate_grad_batches: int = Field(default=1, gt=0)
    gradient_clip_val: float = Field(default=1.0, gt=0)
    log_every_n_steps: int = Field(default=20, gt=0)
    save_top_k: int = Field(default=2, ge=1)
    deterministic: bool = False
    matmul_precision: Literal["highest", "high", "medium"] = "high"

    @model_validator(mode="after")
    def validate_runtime(self) -> Self:
        """Reject empty device selections and unsupported CPU FP16 training."""

        if isinstance(self.devices, int) and self.devices < 1:
            raise ValueError("runtime.devices must be positive.")
        if isinstance(self.devices, list):
            if not self.devices or any(device < 0 for device in self.devices):
                raise ValueError("runtime.devices must contain nonnegative indices.")
            if len(set(self.devices)) != len(self.devices):
                raise ValueError("runtime.devices must not contain duplicates.")
        if self.max_steps == 0:
            raise ValueError("runtime.max_steps must be -1 or positive.")
        if self.accelerator == "cpu" and self.precision == "16-mixed":
            raise ValueError("CPU training does not support precision='16-mixed'.")

        return self


class TrainingConfig(FrozenConfig):
    """Configuration for training and exporting the new FSQ Mel codec."""

    inference_config: Path
    output_directory: Path
    seed: int = Field(default=1234, ge=0)
    data: TrainingDataConfig
    architecture: CodecArchitecture = Field(default_factory=CodecArchitecture)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    runtime: TrainingRuntime = Field(default_factory=TrainingRuntime)

    @model_validator(mode="after")
    def validate_segments(self) -> Self:
        """Ensure that training targets cover an integral number of tokens."""

        if self.data.segment_frames % self.architecture.frame_stride:
            raise ValueError(
                "data.segment_frames must be divisible by architecture.frame_stride."
            )

        return self

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Read JSON and resolve all filesystem locations against its directory."""

        source = Path(path).expanduser().resolve()
        config = cls.model_validate_json(source.read_text(encoding="utf-8"))
        directory = source.parent

        def resolve(value: Path) -> Path:
            """Resolve a configured path, including an optional home prefix."""

            return (directory / value.expanduser()).resolve()

        data = config.data.model_copy(
            update={
                "directory": resolve(config.data.directory),
                "cache_directory": resolve(config.data.cache_directory),
            }
        )

        return config.model_copy(
            update={
                "inference_config": resolve(config.inference_config),
                "output_directory": resolve(config.output_directory),
                "data": data,
            }
        )

    def load_inference_config(self) -> Config:
        """Load and validate the frontend shared by training and inference."""

        config = Config.from_file(self.inference_config)
        audio = config.audio

        if (
            not audio.signal_normalization
            or not audio.symmetric_acoustic
            or audio.max_abs_value != 1.0
        ):
            raise ValueError(
                "Training requires symmetric Mel normalization with max_abs_value=1."
            )
        if audio.n_mels != self.architecture.n_mels:
            raise ValueError("architecture.n_mels must equal audio.n_mels.")

        vocoder = config.vocoder
        compatible = (
            audio.sample_rate == vocoder.mel_sampling_rate
            and audio.hop_size == vocoder.mel_hop_length
            and audio.n_fft == vocoder.mel_filter_length
            and audio.win_size == vocoder.mel_win_length
            and audio.n_mels == vocoder.n_mel_channels
            and audio.fmin == vocoder.mel_fmin
            and audio.fmax == vocoder.mel_fmax
            and audio.min_db == vocoder.min_db
            and audio.min_level_db == vocoder.min_level_db
            and audio.ref_level_db == vocoder.ref_level_db
            and audio.max_abs_value == vocoder.max_abs_value
        )
        if not compatible:
            raise ValueError(
                "Training frontend and vocoder Mel configurations must agree."
            )

        return config
