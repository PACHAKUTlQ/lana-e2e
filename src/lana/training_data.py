import csv
import hashlib
import json
import random
from importlib.metadata import version
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lana.artifacts import FeatureArray, load_array, save_array
from lana.config import AudioConfig, Feature
from lana.features import extract_file
from lana.training_config import TrainingConfig, TrainingDataConfig

type TrainingBatch = dict[str, Tensor]


def read_manifest(directory: Path) -> list[Path]:
    """Read LJSpeech identifiers and verify their corresponding WAV files.

    Metadata must contain the identifier, original text, and normalized text.
    Identifiers are treated as filenames rather than arbitrary relative paths.
    """

    metadata = directory / "metadata.csv"
    records: list[Path] = []
    seen: set[str] = set()

    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="|", quoting=csv.QUOTE_NONE)
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 3:
                raise ValueError(
                    f"{metadata}:{line_number}: expected three pipe-separated fields."
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
                raise ValueError(f"Duplicate metadata identifier: {identifier}")

            path = directory / "wavs" / f"{identifier}.wav"
            if not path.is_file():
                raise FileNotFoundError(f"Audio file not found: {path}")

            seen.add(identifier)
            records.append(path)

    if len(records) < 2:
        raise ValueError("Training requires at least two audio files.")

    return records


class MelDataset(Dataset[TrainingBatch]):
    """Load cached Mels and form segment/prompt pairs from the same utterance.

    Cache identity includes the complete frontend configuration, library
    versions, source path, file size, and modification time. Writes use the
    application's atomic NPY writer so concurrent workers cannot expose
    partially written arrays.
    """

    def __init__(
        self,
        paths: list[Path],
        audio: AudioConfig,
        config: TrainingDataConfig,
        *,
        training: bool,
        seed: int,
    ) -> None:
        """Store dataset membership and whether crop positions are randomized."""

        self.paths = paths
        self.audio = audio
        self.config = config
        self.training = training
        self.seed = seed

        cache_identity = {
            "format": 1,
            "audio": audio.model_dump(mode="json"),
            "librosa": version("librosa"),
            "numpy": np.__version__,
        }
        self.cache_identity = json.dumps(cache_identity, sort_keys=True)

    def __len__(self) -> int:
        """Return the number of utterances in this dataset partition."""

        return len(self.paths)

    def _load_mel(self, path: Path) -> FeatureArray:
        """Load a matching cache entry or extract and atomically cache its Mel."""

        stat = path.stat()
        identity = f"{self.cache_identity}\n{path}\n{stat.st_size}\n{stat.st_mtime_ns}"
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        destination = self.config.cache_directory / key[:2] / f"{key}.npy"

        if destination.is_file():
            mel = np.asarray(load_array(destination), dtype=np.float32)
        else:
            mel = extract_file(path, self.audio, (Feature.MEL,))[Feature.MEL]
            save_array(destination, mel)

        if (
            mel.ndim != 2
            or mel.shape[0] != self.audio.n_mels
            or mel.shape[1] == 0
            or not np.isfinite(mel).all()
        ):
            raise ValueError(f"Invalid Mel features for {path}")

        return mel

    def __getitem__(self, index: int) -> TrainingBatch:
        """Return a fixed-size target, independent prompt crop, and valid mask."""

        path = self.paths[index]
        mel = (self._load_mel(path) + 1.0) / 2.0
        frame_count = mel.shape[1]

        rng = (
            random.Random(random.getrandbits(64))
            if self.training
            else random.Random(self.seed + index)
        )

        segment_frames = self.config.segment_frames
        start = rng.randint(0, max(0, frame_count - segment_frames))
        segment = mel[:, start : start + segment_frames]
        valid_frames = segment.shape[1]

        if valid_frames < segment_frames:
            segment = np.pad(
                segment,
                ((0, 0), (0, segment_frames - valid_frames)),
                mode="edge",
            )

        prompt_frames = self.config.prompt_frames
        if frame_count >= prompt_frames:
            prompt_start = rng.randint(0, frame_count - prompt_frames)
            prompt = mel[:, prompt_start : prompt_start + prompt_frames]
        else:
            repetitions = (prompt_frames + frame_count - 1) // frame_count
            prompt = np.tile(mel, (1, repetitions))[:, :prompt_frames]

        mask = torch.zeros(1, segment_frames, dtype=torch.float32)
        mask[:, :valid_frames] = 1

        return {
            "mel": torch.from_numpy(np.ascontiguousarray(segment)),
            "prompt": torch.from_numpy(np.ascontiguousarray(prompt)),
            "mask": mask,
        }


class MelDataModule(L.LightningDataModule):
    """Deterministic utterance split and distributed-compatible data loaders.

    Each rank constructs the same split. Lightning installs distributed
    samplers when training on multiple devices.
    """

    def __init__(self, config: TrainingConfig, audio: AudioConfig) -> None:
        """Store configuration without extracting features or loading waveforms."""

        super().__init__()
        self.config = config
        self.audio = audio
        self.training_dataset: MelDataset | None = None
        self.validation_dataset: MelDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Build identical train/validation partitions on every participating rank."""

        if self.training_dataset is not None:
            return

        paths = read_manifest(self.config.data.directory)
        random.Random(self.config.seed).shuffle(paths)
        validation_count = max(
            1,
            round(len(paths) * self.config.data.validation_fraction),
        )
        validation_count = min(validation_count, len(paths) - 1)

        self.validation_dataset = MelDataset(
            paths[:validation_count],
            self.audio,
            self.config.data,
            training=False,
            seed=self.config.seed,
        )
        self.training_dataset = MelDataset(
            paths[validation_count:],
            self.audio,
            self.config.data,
            training=True,
            seed=self.config.seed,
        )

    def _loader(
        self,
        dataset: MelDataset,
        *,
        shuffle: bool,
    ) -> DataLoader[TrainingBatch]:
        """Construct a loader with persistent workers only when workers exist."""

        config = self.config.data

        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.workers > 0,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader[TrainingBatch]:
        """Return shuffled training crops after dataset setup."""

        if self.training_dataset is None:
            raise RuntimeError("Training dataset has not been initialized.")

        return self._loader(self.training_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[TrainingBatch]:
        """Return deterministic validation crops without shuffling."""

        if self.validation_dataset is None:
            raise RuntimeError("Validation dataset has not been initialized.")

        return self._loader(self.validation_dataset, shuffle=False)
