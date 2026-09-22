from collections.abc import Iterator

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.fft import rfft
from scipy.signal.windows import hann

from lana.acoustics.features import amplitude_to_db, normalize_spectrogram
from lana.config import AudioConfig

type PcmChunk = NDArray[np.int16]
type MelChunk = NDArray[np.float32]


class StreamingMelFrontend:
    """Extract centered Mel frames from successive mono PCM16 sample blocks.

    A frame is emitted only after its complete right-hand FFT context arrives.
    On finish, missing right-hand samples are zero-padded. Frames are limited
    to complete-hop duration, matching the offline frontend's frame count.
    The buffer retains only samples needed for the next analysis frame.
    """

    def __init__(self, config: AudioConfig) -> None:
        """Prepare a centered Hann window and the configured Slaney Mel filters."""

        if (
            not config.signal_normalization
            or not config.symmetric_acoustic
            or config.max_abs_value != 1.0
            or config.stft_pad_mode != "constant"
        ):
            raise ValueError(
                "Streaming requires symmetric normalized Mel features with "
                "max_abs_value=1 and constant STFT padding."
            )

        self.config = config
        self._basis = librosa.filters.mel(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax,
            htk=False,
            norm="slaney",
            dtype=np.float32,
        )

        window = hann(config.win_size, sym=False).astype(np.float32)
        self._window = np.pad(
            window,
            (
                (config.n_fft - config.win_size) // 2,
                config.n_fft - config.win_size - (config.n_fft - config.win_size) // 2,
            ),
        )

        self._samples = np.empty(0, dtype=np.float32)
        self._sample_offset = 0
        self._total_samples = 0
        self._next_frame = 0
        self._finished = False

    def push(self, samples: PcmChunk) -> MelChunk:
        """Append PCM16 samples and return newly available complete-context frames."""

        if self._finished:
            raise RuntimeError("The Mel frontend has already been finished.")
        if samples.ndim != 1 or samples.dtype != np.int16:
            raise ValueError("Expected one-dimensional PCM16 samples.")

        if samples.size:
            self._samples = np.concatenate(
                (self._samples, samples.astype(np.float32) / 32768.0)
            )
            self._total_samples += samples.size

        return self._emit(final=False)

    def finish(self) -> MelChunk:
        """Emit remaining complete-hop frames using zero right-hand padding."""

        if self._finished:
            raise RuntimeError("The Mel frontend has already been finished.")

        self._finished = True

        return self._emit(final=True)

    def _frames(self, *, final: bool) -> Iterator[NDArray[np.float32]]:
        """Yield centered FFT windows whose required samples have arrived."""

        config = self.config
        half_fft = config.n_fft // 2
        maximum_frames = self._total_samples // config.hop_size

        while self._next_frame < maximum_frames:
            start = self._next_frame * config.hop_size - half_fft
            stop = start + config.n_fft
            if not final and stop > self._total_samples:
                break

            available_start = max(start, 0)
            available_stop = min(stop, self._total_samples)
            segment = self._samples[
                available_start - self._sample_offset : available_stop
                - self._sample_offset
            ]
            frame = np.zeros(config.n_fft, dtype=np.float32)
            frame[available_start - start : available_start - start + segment.size] = (
                segment
            )

            self._next_frame += 1
            yield frame

        earliest_needed = max(
            0,
            self._next_frame * config.hop_size - half_fft,
        )
        removable = min(
            earliest_needed - self._sample_offset,
            self._samples.size,
        )
        if removable > 0:
            self._samples = self._samples[removable:]
            self._sample_offset += removable

    def _emit(self, *, final: bool) -> MelChunk:
        """Transform available FFT windows into frontend-normalized Mel frames."""

        frames = list(self._frames(final=final))
        if not frames:
            return np.empty((self.config.n_mels, 0), dtype=np.float32)

        windows = np.stack(frames, axis=1) * self._window[:, None]
        magnitude = np.abs(rfft(windows, axis=0))
        mel = self._basis @ magnitude
        decibels = amplitude_to_db(mel, self.config) - self.config.ref_level_db
        normalized = normalize_spectrogram(decibels, self.config)

        return np.asarray(normalized, dtype=np.float32)
