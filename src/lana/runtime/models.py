from functools import cached_property
from pathlib import Path
from typing import Protocol, cast

import librosa
import numpy as np
import torch
from torch import Tensor

from lana.acoustics.features import interpolate_pitch
from lana.artifacts import Audio
from lana.codec.artifact import load_codec
from lana.config import Config, ModelPaths, VocoderConfig


class CodecModel(Protocol):
    """Codec methods shared by offline and streaming application interfaces."""

    def encode(
        self,
        mel: Tensor,
        prompt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return quantized content, layered codes, and speaker information."""

        ...

    def decode(self, codes: Tensor, speaker_codes: Tensor) -> Tensor:
        """Decode content codes with shape (batch, positions, quantizers)."""

        ...

    def ref_encoder(self, prompt: Tensor) -> Tensor:
        """Encode a bounded prompt history."""

        ...

    def refemb_fsq(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize speaker conditioning."""

        ...

    def encode_ref(
        self,
        mel: Tensor,
        embedding: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode content using external speaker conditioning."""

        ...


class PitchModel(Protocol):
    """Exported Mel-to-pitch predictor."""

    def __call__(self, mel: Tensor) -> Tensor:
        """Predict log-F0 and voicing."""

        ...


class PriorGeneratorModel(Protocol):
    """DSPGAN harmonic prior generator."""

    def infer(
        self,
        mel: Tensor,
        pitch: Tensor,
        voiced: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate prior audio, noise, and upsampled voicing."""

        ...


class GeneratorModel(Protocol):
    """DSPGAN final waveform generator."""

    def __call__(self, mel: Tensor, voiced: Tensor) -> Tensor:
        """Generate a waveform batch."""

        ...


def load_torchscript(path: Path, device: torch.device) -> torch.jit.ScriptModule:
    """Load a locally trusted TorchScript model on the requested device."""

    if not path.is_file():
        raise FileNotFoundError(f"TorchScript model not found: {path}")
    model = torch.jit.load(str(path), map_location=device)
    model.eval()

    return model


def canonical_mel(mel: Tensor, channels: int) -> Tensor:
    """Convert a single spectrogram to (channels, frames) without flattening."""

    if mel.ndim == 3:
        if mel.shape[0] != 1:
            raise ValueError("Only one spectrogram is supported.")
        mel = mel.squeeze(0)
    if mel.ndim != 2:
        raise ValueError(f"Invalid Mel dimensions: {tuple(mel.shape)}")
    if mel.shape[0] != channels:
        if mel.shape[1] != channels:
            raise ValueError(f"Expected {channels} Mel channels.")
        mel = mel.transpose(0, 1)
    if mel.shape[1] == 0:
        raise ValueError("Mel spectrogram has no frames.")

    return mel


class PriorMelTransform:
    """Centered STFT and Slaney filtering for DSPGAN prior audio."""

    def __init__(self, config: VocoderConfig, device: torch.device) -> None:
        """Prepare device-local filter and window tensors."""

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
        self.window = torch.hann_window(config.mel_win_length, device=device)

    def __call__(
        self,
        original_mel: Tensor,
        waveform: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return normalized prior Mels and residual amplitude features."""

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
        frames = original_mel.shape[-1]
        if prior_mel.shape[-1] < frames:
            raise ValueError("Prior audio contains fewer frames than codec output.")
        prior_mel = prior_mel[:, :, :frames]

        original_db = (
            original_mel.clamp(-config.max_abs_value, config.max_abs_value)
            + config.max_abs_value
        ) * -config.min_db / (2 * config.max_abs_value) + config.min_db
        original_amplitude = torch.pow(
            10.0,
            (original_db + config.ref_level_db) * 0.05,
        )
        residual = original_amplitude - prior_mel
        minimum = float(np.exp(config.min_level_db / 20 * np.log(10)))
        prior_db = 20 * torch.log10(prior_mel.clamp_min(minimum)) - config.ref_level_db
        normalized = (
            2 * config.max_abs_value * ((prior_db - config.min_db) / -config.min_db)
            - config.max_abs_value
        ).clamp(-config.max_abs_value, config.max_abs_value)

        return normalized, residual


class Vocoder:
    """Lazily loaded DSPGAN synthesis pipeline."""

    def __init__(
        self,
        paths: ModelPaths,
        config: VocoderConfig,
        device: torch.device,
    ) -> None:
        """Store independently loadable synthesis components."""

        self.paths = paths
        self.config = config
        self.device = device

    @cached_property
    def pitch_model(self) -> PitchModel:
        """Load the pitch predictor on first use."""

        return cast(PitchModel, load_torchscript(self.paths.mel2f0, self.device))

    @cached_property
    def prior_generator(self) -> PriorGeneratorModel:
        """Load the harmonic prior generator on first use."""

        return cast(
            PriorGeneratorModel,
            load_torchscript(self.paths.prior_generator, self.device),
        )

    @cached_property
    def generator(self) -> GeneratorModel:
        """Load the final waveform generator on first use."""

        return cast(
            GeneratorModel,
            load_torchscript(self.paths.generator, self.device),
        )

    @cached_property
    def mel_transform(self) -> PriorMelTransform:
        """Build device-local prior Mel analysis tensors."""

        return PriorMelTransform(self.config, self.device)

    @torch.inference_mode()
    def decode(self, mel: Tensor) -> Audio:
        """Synthesize a single Mel block and safely saturate its PCM16 output."""

        mel = canonical_mel(mel, self.config.n_mel_channels)
        mel = mel.to(device=self.device, dtype=torch.float32)
        prediction = self.pitch_model((mel * self.config.mel2f0_scale).unsqueeze(0))
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction.squeeze(0)
        if prediction.ndim != 2 or prediction.shape[1] != 2:
            raise ValueError("Pitch output must have shape (frames, 2).")

        values = prediction.detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Pitch prediction contains non-finite values.")

        log_f0 = values[:, 0].copy()
        log_f0[values[:, 1] < 0.5] = 0
        voiced = (log_f0 > 0).astype(np.float32)
        pitch_hz = np.zeros_like(log_f0)
        active = log_f0 > 0
        pitch_hz[active] = np.exp(log_f0[active])
        if not np.isfinite(pitch_hz).all():
            raise ValueError("Pitch prediction exceeds the finite Hz range.")
        pitch_hz = interpolate_pitch(pitch_hz)

        pitch = (
            torch.from_numpy(pitch_hz)
            .to(
                device=self.device,
                dtype=torch.float32,
            )
            .reshape(1, 1, -1)
        )
        voiced_tensor = torch.from_numpy(voiced).to(self.device).reshape(1, 1, -1)
        batched_mel = mel.unsqueeze(0)

        mel_frames = batched_mel.shape[-1]
        pitch_frames = pitch.shape[-1]
        if pitch_frames == 0:
            raise ValueError("Pitch prediction has no frames.")
        if pitch_frames > mel_frames:
            if pitch_frames - mel_frames >= 10:
                raise ValueError("Pitch exceeds Mel length by at least ten frames.")
            pitch = pitch[:, :, :mel_frames]
            voiced_tensor = voiced_tensor[:, :, :mel_frames]
        elif pitch_frames < mel_frames:
            batched_mel = batched_mel[:, :, :pitch_frames]

        prior_audio, _, voiced_upsampled = self.prior_generator.infer(
            batched_mel,
            pitch,
            voiced_tensor,
        )
        prior_mel, _ = self.mel_transform(batched_mel, prior_audio.squeeze(1))
        waveform = self.generator(prior_mel, voiced_upsampled)
        samples = waveform.detach().cpu().float().numpy().reshape(-1)
        if not np.isfinite(samples).all():
            raise ValueError("The vocoder generated non-finite samples.")

        pcm = np.clip(
            samples * self.config.max_wav_value,
            -32768,
            32767,
        ).astype(np.int16)

        return Audio(pcm, self.config.sampling_rate)


class Models:
    """Shared lazy model registry for one application configuration."""

    def __init__(self, config: Config) -> None:
        """Store settings and resolve the Torch device."""

        self.config = config
        self.device = torch.device(config.device)

    @cached_property
    def codec(self) -> CodecModel:
        """Load the scalable codec and select its configured quantizer prefix."""

        return cast(CodecModel, load_codec(self.config, self.device))

    @cached_property
    def vocoder(self) -> Vocoder:
        """Create the independently lazy DSPGAN pipeline."""

        return Vocoder(self.config.models, self.config.vocoder, self.device)
