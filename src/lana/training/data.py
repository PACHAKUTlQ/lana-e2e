import hashlib
import json
import random
from importlib.metadata import version
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lana.artifacts import FeatureArray, load_array, save_array
from lana.config import Config, TrainingConfig, TrainingDataConfig
from lana.streaming.audio import iter_pcm
from lana.streaming.frontend import StreamingMelFrontend

type TrainingBatch = dict[str, torch.Tensor]

_CACHE_VERSION = 2


def discover_training_audio(settings: TrainingDataConfig) -> list[Path]:
    """Recursively discover audio without requiring dataset-specific metadata.

    Matching suffixes are case-insensitive. Results are sorted by paths
    relative to the configured root so discovery order is reproducible across
    machines containing the same directory layout.
    """

    directory = settings.directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Training audio directory not found: {directory}")

    extensions = set(settings.audio_extensions)
    paths = sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if len(paths) < 2:
        raise ValueError(
            f"Training requires at least two audio files in {directory}; "
            f"found {len(paths)} matching {settings.audio_extensions}."
        )

    return paths


def split_training_audio(
    paths: list[Path],
    settings: TrainingDataConfig,
    *,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Partition audio deterministically, optionally keeping directory groups intact.

    Without grouping, validation_fraction applies to files. With grouping, it
    applies to distinct values of the selected relative directory component.
    Both modes retain at least one training and one validation unit.

    Grouped output preserves discovery order. A missing directory component
    raises an error instead of silently assigning an incorrect group.
    """

    component = settings.validation_group_component
    randomizer = random.Random(seed)

    if component is None:
        shuffled = paths.copy()
        randomizer.shuffle(shuffled)
        validation_count = min(
            len(shuffled) - 1,
            max(1, round(len(shuffled) * settings.validation_fraction)),
        )

        return shuffled[validation_count:], shuffled[:validation_count]

    directory = settings.directory.expanduser().resolve()
    path_groups: list[tuple[Path, str]] = []
    for path in paths:
        relative = path.relative_to(directory)
        components = relative.parent.parts
        if component >= len(components):
            raise ValueError(
                f"Audio path {relative} has no relative directory component "
                f"{component} for validation grouping."
            )
        path_groups.append((path, components[component]))

    groups = sorted({group for _, group in path_groups})
    if len(groups) < 2:
        raise ValueError(
            "Grouped validation requires at least two distinct groups; "
            f"found {len(groups)} at directory component {component}."
        )

    randomizer.shuffle(groups)
    validation_count = min(
        len(groups) - 1,
        max(1, round(len(groups) * settings.validation_fraction)),
    )
    validation_groups = set(groups[:validation_count])
    training_paths: list[Path] = []
    validation_paths: list[Path] = []

    for path, group in path_groups:
        if group in validation_groups:
            validation_paths.append(path)
        else:
            training_paths.append(path)

    return training_paths, validation_paths


class MelSequenceDataset(Dataset[TrainingBatch]):
    """Cache streaming-compatible Mels and sample consecutive block sequences.

    Each sampled sequence starts a fresh conditioning session. Short sequences
    repeat their final frame, while the loss mask excludes padded targets.
    Validation crops are deterministic for a given seed and manifest order.
    """

    def __init__(
        self,
        paths: list[Path],
        config: Config,
        *,
        training: bool,
    ) -> None:
        """Store split membership and a versioned frontend cache identity."""

        self.paths = paths
        self.config = config
        self.settings = config.validate_training()
        self.training = training
        self.frames = config.streaming.chunk_frames * self.settings.data.sequence_chunks
        self.cache_identity = json.dumps(
            {
                "format": _CACHE_VERSION,
                "audio": config.audio.model_dump(mode="json"),
                "numpy": np.__version__,
                "scipy": version("scipy"),
                "librosa": version("librosa"),
                "input_format": "ffmpeg-mono-pcm16",
            },
            sort_keys=True,
        )

    def __len__(self) -> int:
        """Return the number of utterances in this partition."""

        return len(self.paths)

    def _load_mel(self, path: Path) -> FeatureArray:
        """Read or atomically create a cache using the live streaming frontend."""

        stat = path.stat()
        identity = f"{self.cache_identity}\n{path}\n{stat.st_size}\n{stat.st_mtime_ns}"
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        destination = self.settings.data.cache_directory / key[:2] / f"{key}.npy"

        if destination.is_file():
            mel = np.asarray(load_array(destination), dtype=np.float32)
        else:
            frontend = StreamingMelFrontend(self.config.audio)
            pieces: list[FeatureArray] = []
            for samples in iter_pcm(path, self.config.audio.sample_rate):
                values = frontend.push(samples)
                if values.shape[1]:
                    pieces.append(values)
            final = frontend.finish()
            if final.shape[1]:
                pieces.append(final)
            if not pieces:
                raise ValueError(f"Audio contains no complete Mel frames: {path}")

            mel = np.concatenate(pieces, axis=1)
            save_array(destination, mel)

        if (
            mel.ndim != 2
            or mel.shape[0] != self.config.audio.n_mels
            or mel.shape[1] == 0
            or not np.isfinite(mel).all()
        ):
            raise ValueError(f"Invalid cached Mel spectrogram: {destination}")

        return mel

    def __getitem__(self, index: int) -> TrainingBatch:
        """Return normalized consecutive blocks and a valid-frame loss mask."""

        mel = (self._load_mel(self.paths[index]) + 1.0) / 2.0
        maximum_start = max(0, mel.shape[1] - self.frames)
        if self.training:
            start = random.randint(0, maximum_start)
        else:
            start = random.Random(self.settings.seed + index).randint(
                0,
                maximum_start,
            )

        segment = mel[:, start : start + self.frames]
        valid_frames = segment.shape[1]
        if valid_frames < self.frames:
            segment = np.pad(
                segment,
                ((0, 0), (0, self.frames - valid_frames)),
                mode="edge",
            )

        mask = torch.zeros(1, self.frames, dtype=torch.float32)
        mask[:, :valid_frames] = 1.0

        return {
            "mel": torch.from_numpy(np.ascontiguousarray(segment)),
            "mask": mask,
        }


class MelDataModule(L.LightningDataModule):
    """Build reproducible utterance partitions and distributed-aware loaders."""

    def __init__(self, config: Config) -> None:
        """Store configuration without opening audio files."""

        super().__init__()
        self.config = config
        self.settings: TrainingConfig = config.validate_training()
        self.training_dataset: MelSequenceDataset | None = None
        self.validation_dataset: MelSequenceDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Discover audio and create identical file or group partitions on all ranks."""

        if self.training_dataset is not None:
            return

        paths = discover_training_audio(self.settings.data)
        training_paths, validation_paths = split_training_audio(
            paths,
            self.settings.data,
            seed=self.settings.seed,
        )
        self.validation_dataset = MelSequenceDataset(
            validation_paths,
            self.config,
            training=False,
        )
        self.training_dataset = MelSequenceDataset(
            training_paths,
            self.config,
            training=True,
        )

    def _loader(
        self,
        dataset: MelSequenceDataset,
        *,
        shuffle: bool,
    ) -> DataLoader[TrainingBatch]:
        """Construct a loader with optional persistent worker processes."""

        settings = self.settings.data

        return DataLoader(
            dataset,
            batch_size=settings.batch_size,
            shuffle=shuffle,
            num_workers=settings.workers,
            persistent_workers=settings.workers > 0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader[TrainingBatch]:
        """Return randomized training sequences."""

        if self.training_dataset is None:
            raise RuntimeError("Training dataset has not been initialized.")

        return self._loader(self.training_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[TrainingBatch]:
        """Return deterministic validation sequences."""

        if self.validation_dataset is None:
            raise RuntimeError("Validation dataset has not been initialized.")

        return self._loader(self.validation_dataset, shuffle=False)
