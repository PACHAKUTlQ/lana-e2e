from functools import cached_property
from pathlib import Path
from typing import Protocol, cast

import librosa
import numpy as np
import torch
from torch import Tensor

from lana.artifacts import Audio
from lana.config import Config, ModelPaths, VocoderConfig
from lana.features import interpolate_pitch


class CodecModel(Protocol):
    """TorchScript methods used by whole-utterance and chunked codec inference."""

    def encode(
        self,
        mel: Tensor,
        prompt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return content and speaker quantizer outputs and code tensors."""

        ...

    def decode(self, codes: Tensor, speaker_codes: Tensor) -> Tensor:
        """Decode content and speaker codes into a batch of normalized Mels."""

        ...

    def ref_encoder(self, prompt: Tensor) -> Tensor:
        """Encode the current prompt history into a speaker embedding."""

        ...

    def refemb_fsq(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize a speaker embedding and return its discrete codes."""

        ...

    def encode_ref(
        self,
        mel: Tensor,
        embedding: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a Mel chunk using an externally supplied speaker embedding."""

        ...


class PitchModel(Protocol):
    """Mel-to-pitch TorchScript callable."""

    def __call__(self, mel: Tensor) -> Tensor:
        """Return predicted log-F0 and voicing values."""

        ...


class PriorGeneratorModel(Protocol):
    """DSPGAN prior generator with its exported infer method."""

    def infer(
        self,
        mel: Tensor,
        pitch: Tensor,
        voiced: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return prior audio, harmonic noise, and upsampled voicing."""

        ...


class GeneratorModel(Protocol):
    """DSPGAN waveform generator conditioned on prior Mels and voicing."""

    def __call__(self, mel: Tensor, voiced: Tensor) -> Tensor:
        """Generate the final waveform batch."""

        ...


def load_torchscript(path: Path, device: torch.device) -> torch.jit.ScriptModule:
    """Load a trusted TorchScript artifact directly onto the target device."""

    if not path.is_file():
        raise FileNotFoundError(f"TorchScript model not found: {path}")

    model = torch.jit.load(str(path), map_location=device)
    model.eval()

    return model


def canonical_mel(mel: Tensor, channels: int) -> Tensor:
    """Convert a single Mel spectrogram to (channels, frames) without flattening."""

    if mel.ndim == 3:
        if mel.shape[0] != 1:
            raise ValueError("Only a single Mel spectrogram is supported.")
        mel = mel.squeeze(0)

    if mel.ndim != 2:
        raise ValueError(f"Expected a two-dimensional Mel spectrogram: {mel.shape}")

    if mel.shape[0] != channels:
        if mel.shape[1] != channels:
            raise ValueError(f"Expected {channels} Mel channels: {mel.shape}")
        mel = mel.transpose(0, 1)

    if mel.shape[1] == 0:
        raise ValueError("Mel spectrogram has no frames.")

    return mel


class PriorMelTransform:
    """Centered Torch STFT and Slaney filtering for DSPGAN prior audio.

    The magnitude computation deliberately omits the epsilon used by other
    legacy Torch spectral helpers.
    """

    def __init__(self, config: VocoderConfig, device: torch.device) -> None:
        """Create float32 analysis tensors local to this vocoder instance."""

        self.config = config
        basis = librosa.filters.mel(
            sr=config.mel_sampling_rate,
            n_fft=config.mel_filter_length,
            n_mels=config.n_mel_channels,
            fmin=config.mel_fmin,
            fmax=config.mel_fmax,
            htk=False,
            norm="slaney",
            dtype=np.float32,
        )
        self.basis = torch.from_numpy(basis).to(device)
        self.window = torch.hann_window(config.mel_win_length).to(device)

    def __call__(
        self,
        original_mel: Tensor,
        waveform: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return normalized prior Mels and unnormalized residual amplitudes."""

        config = self.config
        spectrum = torch.stft(
            waveform,
            n_fft=config.mel_filter_length,
            hop_length=config.mel_hop_length,
            win_length=config.mel_win_length,
            window=self.window.to(dtype=waveform.dtype),
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(dim=-1))
        prior_mel = torch.matmul(self.basis.to(dtype=waveform.dtype), magnitude)

        frame_count = original_mel.shape[-1]
        if prior_mel.shape[-1] < frame_count:
            raise ValueError(
                "Prior audio produces fewer Mel frames than the codec output: "
                f"{prior_mel.shape[-1]} < {frame_count}."
            )
        prior_mel = prior_mel[:, :, :frame_count]

        original_db = (
            (
                torch.clamp(
                    original_mel,
                    -config.max_abs_value,
                    config.max_abs_value,
                )
                + config.max_abs_value
            )
            * -config.min_db
            / (2 * config.max_abs_value)
        ) + config.min_db
        original_amplitude = torch.pow(10.0, (original_db + config.ref_level_db) * 0.05)
        residual = original_amplitude - prior_mel

        minimum = float(np.exp(config.min_level_db / 20 * np.log(10)))
        prior_db = 20 * torch.log10(torch.clamp(prior_mel, min=minimum))
        prior_db = prior_db - config.ref_level_db
        normalized = torch.clamp(
            (2 * config.max_abs_value) * ((prior_db - config.min_db) / -config.min_db)
            - config.max_abs_value,
            -config.max_abs_value,
            config.max_abs_value,
        )

        return normalized, residual


class Vocoder:
    """Lazy-loaded DSPGAN synthesis pipeline for decoded Mel spectrograms."""

    def __init__(
        self,
        paths: ModelPaths,
        config: VocoderConfig,
        device: torch.device,
    ) -> None:
        """Store model paths and analysis settings without loading artifacts."""

        self.paths = paths
        self.config = config
        self.device = device

    @cached_property
    def pitch_model(self) -> PitchModel:
        """Load the exported Mel-to-pitch model on first use."""

        return cast(
            PitchModel,
            load_torchscript(self.paths.mel2f0, self.device),
        )

    @cached_property
    def prior_generator(self) -> PriorGeneratorModel:
        """Load the exported harmonic prior generator on first use."""

        return cast(
            PriorGeneratorModel,
            load_torchscript(self.paths.prior_generator, self.device),
        )

    @cached_property
    def generator(self) -> GeneratorModel:
        """Load the exported final waveform generator on first use."""

        return cast(
            GeneratorModel,
            load_torchscript(self.paths.generator, self.device),
        )

    @cached_property
    def mel_transform(self) -> PriorMelTransform:
        """Build the prior-audio spectral transform on first use."""

        return PriorMelTransform(self.config, self.device)

    @torch.inference_mode()
    def decode(self, mel: Tensor) -> Audio:
        """Synthesize PCM16 audio using the original pitch and DSPGAN sequence."""

        mel = canonical_mel(mel, self.config.n_mel_channels)
        mel = mel.to(device=self.device, dtype=torch.float32)
        conditions = (mel * self.config.mel2f0_scale).unsqueeze(0)

        prediction = self.pitch_model(conditions)
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction.squeeze(0)
        if prediction.ndim != 2 or prediction.shape[1] != 2:
            raise ValueError(
                "The pitch model must return shape (1, frames, 2) "
                f"or (frames, 2), got {tuple(prediction.shape)}."
            )

        pitch_data = prediction.detach().cpu().numpy()
        log_f0 = pitch_data[:, 0].copy()
        log_f0[pitch_data[:, 1] < 0.5] = 0

        voiced = np.zeros(log_f0.shape, dtype=np.float32)
        voiced[log_f0 > 0] = 1
        pitch_hz = np.where(log_f0 > 0, np.exp(log_f0), 0.0)
        pitch_hz = interpolate_pitch(pitch_hz)

        pitch = torch.from_numpy(pitch_hz).to(
            device=self.device,
            dtype=torch.float32,
        )
        pitch = pitch.reshape(1, 1, -1)
        voiced_tensor = torch.from_numpy(voiced).to(self.device).reshape(1, 1, -1)
        batched_mel = mel.unsqueeze(0)

        mel_frames = batched_mel.shape[-1]
        pitch_frames = pitch.shape[-1]
        if pitch_frames == 0:
            raise ValueError("The pitch model returned no frames.")
        if pitch_frames > mel_frames:
            if pitch_frames - mel_frames >= 10:
                raise ValueError(
                    "Pitch exceeds Mel length by at least ten frames: "
                    f"{pitch_frames} versus {mel_frames}."
                )
            pitch = pitch[:, :, :mel_frames]
            voiced_tensor = voiced_tensor[:, :, :mel_frames]
        elif pitch_frames < mel_frames:
            batched_mel = batched_mel[:, :, :pitch_frames]

        prior_audio, _, voiced_upsampled = self.prior_generator.infer(
            batched_mel,
            pitch,
            voiced_tensor,
        )
        prior_mel, _ = self.mel_transform(
            batched_mel,
            prior_audio.squeeze(1),
        )
        waveform = self.generator(prior_mel, voiced_upsampled)
        samples = waveform.detach().cpu().float().numpy().reshape(-1)

        if not np.isfinite(samples).all():
            raise ValueError("The vocoder generated non-finite samples.")

        # Direct PCM conversion preserves the legacy behavior at full scale.
        pcm = (samples * self.config.max_wav_value).astype(np.int16)

        return Audio(pcm, self.config.sampling_rate)


class Models:
    """Lazy model registry shared by the application's codec and vocoder paths."""

    def __init__(self, config: Config) -> None:
        """Store configuration and resolve the requested Torch device."""

        self.config = config
        self.device = torch.device(config.device)

    @cached_property
    def codec(self) -> CodecModel:
        """Load the exported codec without loading any vocoder components."""

        return cast(
            CodecModel,
            load_torchscript(self.config.models.codec, self.device),
        )

    @cached_property
    def vocoder(self) -> Vocoder:
        """Create the independently lazy-loaded DSPGAN pipeline."""

        return Vocoder(
            self.config.models,
            self.config.vocoder,
            self.device,
        )
