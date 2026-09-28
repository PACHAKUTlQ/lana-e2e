import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

type FloatArray = NDArray[np.float32 | np.float64]
type FeatureArray = NDArray[np.float32]
type TokenArray = NDArray[np.generic]


def load_array(path: str | Path) -> TokenArray:
    """Load a single NumPy array without enabling pickle deserialization."""

    array = np.load(Path(path), allow_pickle=False)
    if not isinstance(array, np.ndarray):
        array.close()
        raise TypeError(f"Expected an NPY array, not an NPZ archive: {path}")

    return array


def save_array(path: str | Path, array: TokenArray) -> Path:
    """Atomically save an NPY array, appending the suffix like numpy.save."""

    destination = Path(path)
    if destination.suffix != ".npy":
        destination = destination.with_name(f"{destination.name}.npy")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.save(stream, array, allow_pickle=False)

        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return destination


@dataclass(frozen=True, slots=True)
class Tokens:
    """Content and speaker codes with their model-produced shapes and dtypes."""

    content: TokenArray
    speaker: TokenArray

    @classmethod
    def load(cls, content_path: str | Path, speaker_path: str | Path) -> "Tokens":
        """Load content and speaker codes from separate NPY files."""

        return cls(load_array(content_path), load_array(speaker_path))

    def save(
        self,
        content_path: str | Path,
        speaker_path: str | Path,
    ) -> tuple[Path, Path]:
        """Save both code arrays without reshaping or converting their dtypes."""

        content = save_array(content_path, self.content)
        speaker = save_array(speaker_path, self.speaker)

        return content, speaker


@dataclass(frozen=True, slots=True)
class Audio:
    """Mono PCM16 audio and its output sampling rate."""

    samples: NDArray[np.int16]
    sample_rate: int

    def save(self, path: str | Path) -> Path:
        """Write PCM16 samples to a WAV file without additional normalization."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(destination, self.sample_rate, self.samples)

        return destination
