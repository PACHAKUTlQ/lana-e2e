import multiprocessing
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from tqdm.auto import tqdm

from lana.artifacts import save_array
from lana.config import AudioConfig, Feature
from lana.features import extract_file

_FEATURE_DIRECTORIES: dict[Feature, str] = {
    Feature.MEL: "mels",
    Feature.LINEAR: "linear_spec",
    Feature.LF0: "lf0",
    Feature.UV: "uv",
    Feature.ENERGY: "energys",
}


def discover_audio(source: str | Path) -> list[Path]:
    """Discover WAV files from a file, a flat directory, or a text manifest.

    Relative manifest entries are resolved against the manifest directory.
    Duplicate paths are removed while preserving their original order.
    """

    source_path = Path(source).expanduser().resolve()
    if source_path.is_dir():
        paths = sorted(
            path
            for path in source_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".wav"
        )
    elif source_path.is_file() and source_path.suffix.lower() == ".wav":
        paths = [source_path]
    elif source_path.is_file():
        paths = []
        for line in source_path.read_text(encoding="utf-8-sig").splitlines():
            entry = line.strip()
            if entry:
                path = Path(entry).expanduser()
                paths.append((source_path.parent / path).resolve())
    else:
        raise FileNotFoundError(f"Audio source not found: {source_path}")

    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError(f"No audio files found: {source_path}")

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

    return paths


def _output_paths(
    source: Path,
    directory: Path,
    features: tuple[Feature, ...],
) -> dict[Feature, Path]:
    """Preserve standalone-Mel and grouped legacy feature directory layouts."""

    if features == (Feature.MEL,):
        return {Feature.MEL: directory / f"{source.stem}.npy"}

    return {
        feature: directory / _FEATURE_DIRECTORIES[feature] / f"{source.stem}.npy"
        for feature in features
    }


def _extract_and_save(
    source: Path,
    *,
    directory: Path,
    config: AudioConfig,
    features: tuple[Feature, ...],
    incremental: bool,
) -> dict[Feature, Path]:
    """Extract one file and atomically write only the requested missing outputs."""

    outputs = _output_paths(source, directory, features)
    pending = tuple(
        feature
        for feature, path in outputs.items()
        if not incremental or not path.is_file()
    )

    if pending:
        try:
            extracted = extract_file(source, config, pending)
            for feature, values in extracted.items():
                save_array(outputs[feature], values)
        except Exception as error:
            raise RuntimeError(f"Feature extraction failed for {source}") from error

    return outputs


def extract_batch(
    paths: Iterable[str | Path],
    directory: str | Path,
    config: AudioConfig,
    features: Iterable[Feature] = (Feature.MEL,),
    *,
    workers: int = 20,
    incremental: bool = False,
    progress: bool = False,
) -> list[dict[Feature, Path]]:
    """Extract files serially or with a bounded process pool.

    Duplicate stems are rejected before writing. Worker failures propagate with
    their source path, and the executor is closed when processing finishes.
    """

    if workers < 1:
        raise ValueError("workers must be positive.")

    selected = tuple(dict.fromkeys(features))
    if not selected:
        raise ValueError("At least one feature must be selected.")

    sources = list(dict.fromkeys(Path(path).expanduser().resolve() for path in paths))
    if not sources:
        raise ValueError("No audio files were provided.")

    stems: dict[str, Path] = {}
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Audio file not found: {source}")
        if source.stem in stems:
            raise ValueError(
                f"Output filename collision: {stems[source.stem]} and {source}"
            )
        stems[source.stem] = source

    output_directory = Path(directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    worker = partial(
        _extract_and_save,
        directory=output_directory,
        config=config,
        features=selected,
        incremental=incremental,
    )

    if workers == 1:
        results = [
            worker(source)
            for source in tqdm(
                sources,
                desc="Extracting",
                disable=not progress,
            )
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            results = list(
                tqdm(
                    executor.map(worker, sources),
                    total=len(sources),
                    desc="Extracting",
                    disable=not progress,
                )
            )

    return results
