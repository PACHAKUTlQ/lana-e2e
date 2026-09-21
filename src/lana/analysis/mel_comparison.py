from pathlib import Path

import librosa
import librosa.display
import matplotlib
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import signal

from lana.acoustics.features import process_wav
from lana.config import AudioConfig, Config

matplotlib.use("Agg")

type Spectrogram = NDArray[np.float64]


def _load_generated_audio(path: Path, config: AudioConfig) -> NDArray[np.float32]:
    """Load generated audio as mono float samples at the frontend sample rate.

    Generated audio is not silence-trimmed: its leading samples are needed to
    estimate the offset relative to the frontend-processed original audio.
    """

    waveform, sample_rate = librosa.load(
        path,
        sr=None,
        mono=True,
        dtype=np.float32,
    )
    if sample_rate != config.sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=config.sample_rate,
            res_type=config.resample_type,
        )

    if waveform.size < config.n_fft or not np.isfinite(waveform).all():
        raise ValueError(f"Generated audio is empty, too short, or non-finite: {path}")

    return waveform


def _mel_power(waveform: NDArray[np.float32], config: AudioConfig) -> Spectrogram:
    """Calculate physical Mel-band power without codec input normalization.

    Both recordings use identical STFT and Slaney filter settings. Squared
    magnitude is used here so the colorbar describes power rather than the
    frontend's normalized Mel values.
    """

    power = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_size,
        win_length=config.win_size,
        window="hann",
        center=True,
        pad_mode=config.stft_pad_mode,
        power=2.0,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        htk=False,
        norm="slaney",
    )

    return np.asarray(power, dtype=np.float64)


def _alignment(
    original_db: Spectrogram,
    generated_db: Spectrogram,
    max_shift_frames: int,
    minimum_frames: int,
) -> tuple[int, int, int, int]:
    """Find an energy-envelope offset with sufficient overlapping Mel frames.

    A positive lag means that a matching event occurs later in the generated
    recording. The returned starts index the original and generated arrays.
    """

    original_envelope = np.mean(original_db, axis=0)
    generated_envelope = np.mean(generated_db, axis=0)
    original_envelope -= np.mean(original_envelope)
    generated_envelope -= np.mean(generated_envelope)

    correlation = signal.correlate(
        generated_envelope,
        original_envelope,
        mode="full",
        method="fft",
    )
    lags = signal.correlation_lags(
        generated_envelope.size,
        original_envelope.size,
        mode="full",
    )

    eligible = (np.abs(lags) <= max_shift_frames) & (
        np.minimum(
            original_envelope.size - np.maximum(-lags, 0),
            generated_envelope.size - np.maximum(lags, 0),
        )
        >= minimum_frames
    )
    if not np.any(eligible):
        raise ValueError("The recordings have no sufficiently long overlap.")

    candidate_indices = np.flatnonzero(eligible)
    lag = int(lags[candidate_indices[np.argmax(correlation[eligible])]])
    original_start = max(-lag, 0)
    generated_start = max(lag, 0)
    frame_count = min(
        original_db.shape[1] - original_start,
        generated_db.shape[1] - generated_start,
    )

    return original_start, generated_start, frame_count, lag


def plot_mel_comparison(
    original_path: str | Path,
    generated_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = "config.json",
    max_shift_seconds: float = 2.0,
) -> Path:
    """Save aligned original and generated Mel-power spectrograms as a PNG.

    The plots share time, frequency, and color limits. A common dB reference
    permits direct color comparison; the log-Mel correlation describes only
    the aligned plotted frames, not perceptual quality.
    """

    if max_shift_seconds < 0:
        raise ValueError("max_shift_seconds must be nonnegative.")

    config = Config.from_file(config_path).audio
    original = np.asarray(process_wav(original_path, config), dtype=np.float32)
    generated = _load_generated_audio(Path(generated_path), config)

    original_power = _mel_power(original, config)
    generated_power = _mel_power(generated, config)
    original_db = np.asarray(
        librosa.power_to_db(original_power, ref=1.0, top_db=None),
        dtype=np.float64,
    )
    generated_db = np.asarray(
        librosa.power_to_db(generated_power, ref=1.0, top_db=None),
        dtype=np.float64,
    )

    frames_per_second = config.sample_rate / config.hop_size
    minimum_frames = max(2, int(frames_per_second))
    original_start, generated_start, frame_count, lag = _alignment(
        original_db,
        generated_db,
        max_shift_frames=round(max_shift_seconds * frames_per_second),
        minimum_frames=minimum_frames,
    )
    original_aligned = original_db[:, original_start : original_start + frame_count]
    generated_aligned = generated_db[:, generated_start : generated_start + frame_count]

    combined = np.concatenate((original_aligned.ravel(), generated_aligned.ravel()))
    lower_limit, upper_limit = np.percentile(combined, (1, 99))
    if lower_limit == upper_limit:
        upper_limit = lower_limit + 1.0

    correlation = float(
        np.corrcoef(original_aligned.ravel(), generated_aligned.ravel())[0, 1]
    )
    correlation_text = f"{correlation:.3f}" if np.isfinite(correlation) else "undefined"

    figure = Figure(figsize=(14, 8), layout="constrained")
    axes = figure.subplots(2, 1, sharex=True, sharey=True)
    image = None

    for axis, values, title in zip(
        axes,
        (original_aligned, generated_aligned),
        ("Original audio", "Generated audio"),
        strict=True,
    ):
        image = librosa.display.specshow(
            values,
            sr=config.sample_rate,
            hop_length=config.hop_size,
            x_axis="time",
            y_axis="mel",
            fmin=config.fmin,
            fmax=config.fmax,
            cmap="magma",
            vmin=float(lower_limit),
            vmax=float(upper_limit),
            ax=axis,
        )
        axis.set_title(title)
        axis.set_ylabel("Frequency (Hz)")

    if image is None:
        raise RuntimeError("No spectrogram image was created.")

    axes[-1].set_xlabel("Time in aligned overlap (s)")
    figure.colorbar(
        image,
        ax=list(axes),
        location="right",
        label="Power (dB, ref = 1)",
    )
    figure.suptitle(
        "Mel-power comparison"
        f"  |  log-Mel correlation: {correlation_text}"
        f"  |  estimated offset: {lag / frames_per_second:+.3f} s"
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)

    return destination
