import csv
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
from lana.config import Config, TrainingConfig
from lana.streaming.audio import iter_pcm
from lana.streaming.frontend import StreamingMelFrontend

type TrainingBatch = dict[str, torch.Tensor]

_CACHE_VERSION = 2


def read_manifest(directory: Path) -> list[Path]:
    """Read LJSpeech identifiers and verify their corresponding audio files."""

    metadata = directory / "metadata.csv"
    paths: list[Path] = []
    seen: set[str] = set()

    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="|", quoting=csv.QUOTE_NONE)
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 3:
                raise ValueError(
                    f"{metadata}:{line_number}: expected three metadata fields."
                )
            identifier = row[0].strip()
            if (
                not identifier
                or identifier in {".", ".."}
                or "/" in identifier
                or "\\" in identifier
            ):
                raise ValueError(f"{metadata}:{line_number}: invalid audio identifier.")
            if identifier in seen:
                raise ValueError(f"Duplicate audio identifier: {identifier}")

            path = directory / "wavs" / f"{identifier}.wav"
            if not path.is_file():
                raise FileNotFoundError(f"Audio file not found: {path}")

            seen.add(identifier)
            paths.append(path)

    if len(paths) < 2:
        raise ValueError("Training requires at least two utterances.")

    return paths


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
        """Create identical partitions on all ranks from a sorted manifest."""

        if self.training_dataset is not None:
            return

        paths = read_manifest(self.settings.data.directory)
        random.Random(self.settings.seed).shuffle(paths)
        validation_count = min(
            len(paths) - 1,
            max(1, round(len(paths) * self.settings.data.validation_fraction)),
        )
        self.validation_dataset = MelSequenceDataset(
            paths[:validation_count],
            self.config,
            training=False,
        )
        self.training_dataset = MelSequenceDataset(
            paths[validation_count:],
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
