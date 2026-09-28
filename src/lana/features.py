from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
from numpy.typing import NDArray

from lana.artifacts import FeatureArray, FloatArray
from lana.config import AudioConfig, Feature, Normalization


@lru_cache(maxsize=32)
def _mel_basis(config: AudioConfig) -> FeatureArray:
    """Cache Slaney mel filters using the complete immutable frontend config."""

    basis = librosa.filters.mel(
        sr=config.sample_rate,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        htk=False,
        norm="slaney",
        dtype=np.float32,
    )
    basis.setflags(write=False)

    return basis


def process_wav(path: str | Path, config: AudioConfig) -> FloatArray:
    """Load, resample, truncate, rescale, and optionally trim a mono waveform.

    Hop alignment occurs before silence trimming, as in the original frontend.
    Trimming can therefore leave a waveform whose length is not hop-aligned.
    """

    wav, original_rate = librosa.load(
        Path(path),
        sr=None,
        mono=True,
        dtype=np.float32,
    )
    if original_rate < config.sample_rate and not config.allow_upsample:
        raise ValueError(
            f"Upsampling from {original_rate} Hz to {config.sample_rate} Hz "
            f"is disabled: {path}"
        )

    wav = librosa.resample(
        wav,
        orig_sr=original_rate,
        target_sr=config.sample_rate,
        res_type=config.resample_type,
    )
    target_length = wav.size // config.hop_size * config.hop_size
    processed: FloatArray = wav[:target_length]

    if processed.size == 0:
        raise ValueError(f"Audio contains less than one hop after resampling: {path}")
    if not np.isfinite(processed).all():
        raise ValueError(f"Audio contains non-finite samples: {path}")

    if config.rescale:
        peak = np.max(np.abs(processed))
        if peak > 0:
            processed = processed / peak * config.rescaling_max

    if config.trim_silence:
        _, interval = librosa.effects.trim(
            processed,
            top_db=config.trim_top_db,
            frame_length=config.trim_fft_size,
            hop_length=config.trim_hop_size,
        )
        silence_samples = config.num_silent_frames * config.hop_size
        start = max(int(interval[0]) - silence_samples // 2, 0)
        stop = min(int(interval[1]) + silence_samples * 2, len(processed))
        processed = processed[start:stop]

        if config.pad_sli:
            padding = np.zeros(
                config.pad_sli_frame * config.hop_size,
                dtype=np.float64,
            )
            processed = np.concatenate((padding, processed, padding))

    if processed.size < config.hop_size:
        raise ValueError(
            f"Audio contains less than one hop after preprocessing: {path}"
        )

    return processed


def amplitude_to_db(values: FloatArray, config: AudioConfig) -> FloatArray:
    """Convert amplitudes to decibels with the configured amplitude floor."""

    minimum = np.exp(config.min_level_db / 20 * np.log(10))

    return 20 * np.log10(np.maximum(minimum, values))


def normalize_spectrogram(values: FloatArray, config: AudioConfig) -> FloatArray:
    """Map relative decibels to the configured symmetric or asymmetric range."""

    if not config.allow_clipping_in_normalization and (
        np.max(values) > 0 or np.min(values) < config.min_db
    ):
        raise ValueError("Spectrogram lies outside [min_db, 0] with clipping disabled.")

    normalized = (values - config.min_db) / -config.min_db
    if config.symmetric_acoustic:
        normalized = 2 * config.max_abs_value * normalized - config.max_abs_value
        lower_bound = -config.max_abs_value
    else:
        normalized = config.max_abs_value * normalized
        lower_bound = 0.0

    if config.allow_clipping_in_normalization:
        normalized = np.clip(normalized, lower_bound, config.max_abs_value)

    return normalized


def denormalize_spectrogram(values: FloatArray, config: AudioConfig) -> FloatArray:
    """Invert frontend normalization, including its configured clipping policy."""

    restored = values
    if config.allow_clipping_in_normalization:
        lower_bound = -config.max_abs_value if config.symmetric_acoustic else 0.0
        restored = np.clip(restored, lower_bound, config.max_abs_value)

    if config.symmetric_acoustic:
        restored = (
            (restored + config.max_abs_value)
            * -config.min_db
            / (2 * config.max_abs_value)
        ) + config.min_db
    else:
        restored = restored * -config.min_db / config.max_abs_value + config.min_db

    return restored


def _scale_spectrogram(
    amplitudes: FloatArray,
    config: AudioConfig,
) -> FeatureArray:
    """Apply amplitude-to-decibel conversion and optional frontend normalization."""

    values = amplitude_to_db(amplitudes, config) - config.ref_level_db
    if config.signal_normalization:
        values = normalize_spectrogram(values, config)

    return np.asarray(values, dtype=np.float32)


def _magnitude_spectrogram(wav: FloatArray, config: AudioConfig) -> FloatArray:
    """Compute the centered STFT magnitude used by both spectral extractors."""

    spectrum = librosa.stft(
        y=wav,
        n_fft=config.n_fft,
        hop_length=config.hop_size,
        win_length=config.win_size,
        window="hann",
        center=True,
        pad_mode=config.stft_pad_mode,
    )

    return np.abs(spectrum)


def _align_spectrogram(
    spectrum: FeatureArray,
    sample_count: int,
    hop_size: int,
) -> FeatureArray:
    """Drop frames beyond the original waveform's complete-hop duration."""

    frame_count = min(sample_count // hop_size, spectrum.shape[1])

    return spectrum[:, :frame_count]


def linear_to_mel(
    linear_spectrogram: FloatArray,
    config: AudioConfig,
) -> FeatureArray:
    """Convert a frontend-scaled linear spectrogram into the same Mel scale."""

    values = linear_spectrogram
    if config.signal_normalization:
        values = denormalize_spectrogram(values, config)
    amplitudes = np.power(10.0, (values + config.ref_level_db) * 0.05)
    mel_amplitudes = np.dot(_mel_basis(config), np.abs(amplitudes))

    return _scale_spectrogram(mel_amplitudes, config)


def interpolate_lf0(
    data: FeatureArray,
) -> tuple[FeatureArray, FeatureArray]:
    """Apply the original in-place LF0 interpolation and return its voiced mask.

    The interpolation intentionally retains the original gap denominator and
    terminal-frame behavior. The returned values share storage with the input.
    """

    values = data.reshape(-1, 1)
    voiced = (values > 0).astype(np.float32)
    frame_count = values.shape[0]
    last_value = np.float32(0.0)

    for index in range(frame_count):
        if values[index, 0] <= 0:
            next_index = index + 1
            for next_index in range(index + 1, frame_count):
                if values[next_index, 0] > 0:
                    break

            if next_index < frame_count - 1:
                if last_value > 0:
                    step = (values[next_index, 0] - values[index - 1, 0]) / np.float32(
                        next_index - index
                    )
                    for gap_index in range(index, next_index):
                        values[gap_index, 0] = values[index - 1, 0] + step * (
                            gap_index - index + 1
                        )
                else:
                    values[index:next_index, 0] = values[next_index, 0]
            else:
                values[index:, 0] = last_value
        else:
            last_value = values[index, 0]

    return values, voiced


def extract_lf0(
    wav: FloatArray,
    config: AudioConfig,
) -> tuple[FeatureArray, FeatureArray]:
    """Extract pYIN log-F0 and a binary voiced mask.

    Raw LF0 has shape (frames,); normalized LF0 and UV have shape (frames, 1).
    The interpolation flag controls mutation of raw LF0, preserving the original
    interaction between interpolation and normalization.
    """

    f0, _, _ = librosa.pyin(
        np.asarray(wav, dtype=np.float64),
        sr=config.sample_rate,
        frame_length=config.n_fft,
        hop_length=config.hop_size,
        fmin=71.0,
        fmax=800.0,
        center=True,
        fill_na=0.0,
    )

    log_f0 = np.log(f0 + 1e-8)
    log_f0[log_f0 < 1e-3] = 0
    raw_lf0 = np.asarray(log_f0, dtype=np.float32).reshape(-1)

    interpolation_input = raw_lf0 if config.lf0_inter else raw_lf0.copy()
    interpolated, voiced = interpolate_lf0(interpolation_input)

    match config.lf0_norm:
        case Normalization.RAW:
            result = raw_lf0
        case Normalization.MINMAX:
            minimum = np.min(interpolated)
            result = (interpolated - minimum) / (np.max(interpolated) - minimum + 1e-6)
        case Normalization.ZSCORE:
            voiced_values = interpolated[voiced > 0]
            result = np.full_like(interpolated, -4.0)
            if voiced_values.size:
                mean = np.mean(voiced_values)
                standard_deviation = np.std(voiced_values)
                if standard_deviation == 0:
                    standard_deviation = np.float32(1.0)
                normalized = (interpolated - mean) / standard_deviation
                result[voiced > 0] = normalized[voiced > 0]

    return np.asarray(result, dtype=np.float32), voiced


def extract_energy(wav: FloatArray, config: AudioConfig) -> FeatureArray:
    """Extract log10 mean-absolute amplitude from unpadded, forward windows.

    The exclusive endpoint matches the original range expression. This feature
    is neither RMS energy nor frame-aligned with the centered Mel spectrogram.
    """

    values = []
    for start in range(0, len(wav) - config.win_size, config.hop_size):
        amplitude = (
            np.sum(np.abs(wav[start : start + config.win_size])) / config.win_size
        )
        values.append([np.log10(np.clip(amplitude, 1e-9, 1e9))])

    if not values:
        raise ValueError("Energy extraction requires more samples than audio.win_size.")

    energy = np.asarray(values)
    match config.energy_norm:
        case Normalization.RAW:
            pass
        case Normalization.MINMAX:
            energy = (energy - (-4.5)) / (np.max(energy) - (-4.5) + 1e-6)
        case Normalization.ZSCORE:
            mean = np.mean(energy)
            standard_deviation = np.std(energy)
            if standard_deviation == 0:
                energy = np.zeros_like(energy)
            else:
                energy = (energy - mean) / standard_deviation

    return np.asarray(energy, dtype=np.float32)


def extract_waveform(
    wav: FloatArray,
    config: AudioConfig,
    features: Iterable[Feature] = (Feature.MEL,),
) -> dict[Feature, FeatureArray]:
    """Extract selected features while sharing spectral and WORLD computations."""

    selected = tuple(dict.fromkeys(features))
    if not selected:
        raise ValueError("At least one feature must be selected.")
    if wav.ndim != 1 or wav.size < config.hop_size:
        raise ValueError("Expected a mono waveform containing at least one hop.")
    if not np.isfinite(wav).all():
        raise ValueError("Waveform contains non-finite samples.")

    result: dict[Feature, FeatureArray] = {}

    if Feature.MEL in selected or Feature.LINEAR in selected:
        magnitude = _magnitude_spectrogram(wav, config)

        if Feature.MEL in selected:
            mel = _scale_spectrogram(np.dot(_mel_basis(config), magnitude), config)
            result[Feature.MEL] = _align_spectrogram(mel, len(wav), config.hop_size)

        if Feature.LINEAR in selected:
            linear = _scale_spectrogram(magnitude, config)
            result[Feature.LINEAR] = _align_spectrogram(
                linear, len(wav), config.hop_size
            )

    if Feature.LF0 in selected or Feature.UV in selected:
        lf0, voiced = extract_lf0(wav, config)
        if Feature.LF0 in selected:
            result[Feature.LF0] = lf0
        if Feature.UV in selected:
            result[Feature.UV] = voiced

    if Feature.ENERGY in selected:
        result[Feature.ENERGY] = extract_energy(wav, config)

    return {feature: result[feature] for feature in selected}


def extract_file(
    path: str | Path,
    config: AudioConfig,
    features: Iterable[Feature] = (Feature.MEL,),
) -> dict[Feature, FeatureArray]:
    """Preprocess an audio file and extract the requested acoustic features."""

    wav = process_wav(path, config)

    return extract_waveform(wav, config, features)


def interpolate_pitch(pitch: FloatArray) -> NDArray[np.float64]:
    """Interpolate positive Hz values only when at least two anchors exist.

    Endpoint values are extended outside the anchor interval. Zero or one
    positive anchor leaves the input unchanged, matching vocoder preprocessing.
    """

    values = np.asarray(pitch, dtype=np.float64).reshape(-1)
    nonzero_indices = np.flatnonzero(values > 0)
    if nonzero_indices.size > 1:
        values = np.interp(
            np.arange(values.size),
            nonzero_indices,
            values[nonzero_indices],
        )

    return values.reshape(-1, 1)
