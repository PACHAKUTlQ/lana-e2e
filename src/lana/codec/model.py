import math
from collections.abc import Iterator
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lana.config import CodecArchitecture


class FiniteScalarQuantizer(nn.Module):
    """Quantize independent scalars and encode them as mixed-radix indices.

    Bounding and rounding run in float32. The straight-through estimator
    supplies gradients through rounding without a learned codebook loss.
    """

    levels: Tensor
    basis: Tensor

    def __init__(self, levels: tuple[int, ...]) -> None:
        """Register quantization levels and integer radix multipliers."""

        super().__init__()
        self.codebook_size = math.prod(levels)
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.long))
        self.register_buffer(
            "basis",
            torch.tensor(
                [math.prod(levels[:index]) for index in range(len(levels))],
                dtype=torch.long,
            ),
        )

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Return normalized quantized scalars and one index per vector."""

        values = values.float()
        levels = self.levels.float()
        half_width = torch.floor(levels / 2)
        bound = (levels - 1) * 1.001 / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / bound)

        bounded = torch.tanh(values + shift) * bound - offset
        rounded = bounded.round()
        rounded = torch.maximum(rounded, -half_width)
        rounded = torch.minimum(rounded, levels - 1 - half_width)
        quantized = (bounded + (rounded - bounded).detach()) / half_width
        digits = rounded.long() + self.levels // 2
        indices = (digits * self.basis).sum(dim=-1)

        return quantized, indices

    @torch.jit.export
    def decode_indices(self, indices: Tensor) -> Tensor:
        """Recover normalized scalar vectors from legal integer indices."""

        if indices.dtype != torch.int64:
            raise ValueError("Quantizer indices must use int64.")
        if bool(torch.any(indices < 0)) or bool(
            torch.any(indices >= self.codebook_size)
        ):
            raise ValueError("Quantizer index is outside its codebook.")

        digits = (
            torch.div(
                indices.unsqueeze(-1),
                self.basis,
                rounding_mode="floor",
            )
            % self.levels
        )
        half_width = self.levels // 2

        return (digits - half_width).float() / half_width.float()


class ResidualScalarStage(nn.Module):
    """Project a latent residual into FSQ dimensions and reconstruct it."""

    def __init__(self, channels: int, levels: tuple[int, ...]) -> None:
        """Create stage-specific analysis and synthesis projections."""

        super().__init__()
        dimensions = len(levels)
        self.analysis = nn.Linear(channels, dimensions)
        self.quantizer = FiniteScalarQuantizer(levels)
        self.synthesis = nn.Linear(dimensions, channels, bias=False)

    def forward(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        """Return this stage's latent contribution and transmitted indices."""

        quantized, indices = self.quantizer(self.analysis(residual))
        contribution = self.synthesis(quantized)

        return contribution.float(), indices

    def decode(self, indices: Tensor) -> Tensor:
        """Reconstruct the same latent contribution from transmitted indices."""

        return self.synthesis(self.quantizer.decode_indices(indices)).float()


class ResidualStageList(nn.ModuleList):
    """Register residual stages with a precisely typed Python iterator.

    TorchScript expands this collection as a ModuleList. The iterator
    annotation supplies the element type to Python static type checkers.
    """

    def __init__(self, stages: list[ResidualScalarStage]) -> None:
        """Register the ordered residual stages under numeric module names."""

        super().__init__(stages)

    def __iter__(self) -> Iterator[ResidualScalarStage]:
        """Iterate over the registered residual stages."""

        return cast(Iterator[ResidualScalarStage], super().__iter__())


class ResidualScalarQuantizer(nn.Module):
    """Represent a latent vector with a truncatable prefix of residual stages."""

    def __init__(
        self,
        channels: int,
        levels: tuple[int, ...],
        num_quantizers: int,
    ) -> None:
        """Create independent residual stages sharing one latent coordinate space."""

        super().__init__()
        self.channels = channels
        self.num_quantizers = num_quantizers
        self.stages = ResidualStageList(
            [ResidualScalarStage(channels, levels) for _ in range(num_quantizers)]
        )

    def forward(self, values: Tensor, layers: int) -> tuple[Tensor, Tensor]:
        """Quantize only the requested prefix and return time-major indices.

        Residual targets stop gradients through earlier stage contributions.
        Reconstruction gradients still reach every transmitted contribution.
        """

        if layers < 1 or layers > self.num_quantizers:
            raise ValueError("Invalid residual quantizer count.")

        residual = values.float()
        reconstruction = torch.zeros_like(residual)
        indices = torch.jit.annotate(list[Tensor], [])

        for index, stage in enumerate(self.stages):
            if index < layers:
                contribution, codes = stage(residual)
                reconstruction = reconstruction + contribution
                residual = residual - contribution.detach()
                indices.append(codes)

        return reconstruction, torch.stack(indices, dim=-1)

    def decode(self, codes: Tensor) -> Tensor:
        """Sum a transmitted prefix with shape (batch, positions, layers)."""

        if codes.dim() != 3:
            raise ValueError("Content codes must have three dimensions.")
        layers = codes.size(2)
        if layers < 1 or layers > self.num_quantizers or codes.size(1) == 0:
            raise ValueError("Invalid content code dimensions.")

        result = torch.zeros(
            (codes.size(0), codes.size(1), self.channels),
            dtype=torch.float32,
            device=codes.device,
        )
        for index, stage in enumerate(self.stages):
            if index < layers:
                result = result + stage.decode(codes[:, :, index])

        return result


class ResidualBlock(nn.Module):
    """Normalize channels locally and apply finite-context temporal filtering."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        """Construct a depthwise filter and pointwise channel mixing branch."""

        super().__init__()
        self.normalization = nn.LayerNorm(channels)
        self.temporal = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            groups=channels,
        )
        self.expand = nn.Conv1d(channels, channels * 2, 1)
        self.contract = nn.Conv1d(channels * 2, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1), 0.1))

    def forward(self, values: Tensor) -> Tensor:
        """Apply the residual branch without normalization across time."""

        hidden = self.normalization(values.transpose(1, 2)).transpose(1, 2)
        hidden = self.temporal(hidden)
        hidden = self.contract(F.gelu(self.expand(hidden)))

        return values + self.scale * hidden


def residual_stack(channels: int, kernel_size: int, layers: int) -> nn.Sequential:
    """Create a repeating finite dilation schedule."""

    return nn.Sequential(
        *[
            ResidualBlock(channels, kernel_size, 2 ** (index % 3))
            for index in range(layers)
        ]
    )


class ScalableCodec(nn.Module):
    """Block Mel codec with residual FSQ content and separate speaker conditioning.

    Content codes have shape (batch, positions, quantizers). Speaker codes have
    shape (batch, 1). Inputs and reconstructions use the [0, 1] Mel convention.
    """

    def __init__(self, n_mels: int, config: CodecArchitecture) -> None:
        """Build shared analysis and synthesis networks for all bitrate prefixes."""

        super().__init__()
        channels = config.hidden_channels
        speaker_dimensions = len(config.speaker_levels)
        self.n_mels = n_mels
        self.frame_stride = config.frame_stride
        self.num_quantizers = config.num_quantizers
        self.active_quantizers = config.num_quantizers

        self.content_frontend = nn.Sequential(
            nn.Conv1d(
                n_mels,
                channels,
                2 * config.frame_stride + 1,
                stride=config.frame_stride,
                padding=config.frame_stride,
            ),
            residual_stack(channels, config.kernel_size, config.encoder_layers),
        )
        self.content_condition = nn.Linear(speaker_dimensions, channels)
        self.content_projection = nn.Conv1d(channels, config.latent_channels, 1)
        self.content_quantizer = ResidualScalarQuantizer(
            config.latent_channels,
            config.content_levels,
            config.num_quantizers,
        )

        self.speaker_frontend = nn.Sequential(
            nn.Conv1d(n_mels, channels, 5, stride=2, padding=2),
            residual_stack(channels, config.kernel_size, config.speaker_layers),
        )
        self.speaker_projection = nn.Linear(channels, speaker_dimensions)
        self.speaker_quantizer = FiniteScalarQuantizer(config.speaker_levels)

        self.decoder_content = nn.Conv1d(config.latent_channels, channels, 1)
        self.decoder_speaker = nn.Linear(speaker_dimensions, channels)
        self.decoder = nn.Sequential(
            residual_stack(channels, config.kernel_size, config.decoder_layers),
            nn.Conv1d(channels, n_mels, 5, padding=2),
            nn.Sigmoid(),
        )

    @torch.jit.export
    def set_num_quantizers(self, layers: int) -> None:
        """Select the inference prefix without changing model weights."""

        if layers < 1 or layers > self.num_quantizers:
            raise ValueError("Requested quantizer count exceeds the model.")
        self.active_quantizers = layers

    @torch.jit.export
    def ref_encoder(self, prompt: Tensor) -> Tensor:
        """Pool a bounded prompt history into continuous conditioning."""

        features = self.speaker_frontend(prompt)

        return self.speaker_projection(features.mean(dim=-1))

    @torch.jit.export
    def refemb_fsq(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize conditioning into one separately transmitted speaker index."""

        quantized, codes = self.speaker_quantizer(embedding)

        return quantized, codes.unsqueeze(-1)

    def _encode_content(
        self,
        mel: Tensor,
        embedding: Tensor,
        layers: int,
    ) -> tuple[Tensor, Tensor]:
        """Analyze a block and quantize its latent representation."""

        features = self.content_frontend(mel)
        features = features + self.content_condition(embedding).unsqueeze(-1)
        latent = self.content_projection(features).transpose(1, 2)
        quantized, codes = self.content_quantizer(latent, layers)

        return quantized.transpose(1, 2), codes

    @torch.jit.export
    def encode_ref(
        self,
        mel: Tensor,
        embedding: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a block with external conditioning and the selected prefix."""

        quantized, codes = self._encode_content(
            mel,
            embedding,
            self.active_quantizers,
        )

        return quantized, codes, mel.new_zeros(())

    @torch.jit.export
    def encode(
        self,
        mel: Tensor,
        prompt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Encode content and speaker information through the public codec API."""

        embedding = self.ref_encoder(prompt)
        speaker, speaker_codes = self.refemb_fsq(embedding)
        content, content_codes, auxiliary = self.encode_ref(mel, embedding)

        return content, content_codes, auxiliary, speaker, speaker_codes

    def _decode_quantized(self, content: Tensor, speaker: Tensor) -> Tensor:
        """Reconstruct Mel frames from differentiable quantized representations."""

        hidden = self.decoder_content(content)
        hidden = hidden + self.decoder_speaker(speaker).unsqueeze(-1)
        hidden = torch.repeat_interleave(hidden, self.frame_stride, dim=-1)

        return self.decoder(hidden)

    @torch.jit.export
    def decode(self, codes: Tensor, speaker_codes: Tensor) -> Tensor:
        """Decode any supported prefix identified by the content tensor shape."""

        if speaker_codes.dim() != 2 or speaker_codes.size(1) != 1:
            raise ValueError("Speaker codes must have shape (batch, 1).")
        if codes.size(0) != speaker_codes.size(0):
            raise ValueError("Content and speaker batch sizes differ.")

        content = self.content_quantizer.decode(codes).transpose(1, 2)
        speaker = self.speaker_quantizer.decode_indices(speaker_codes[:, 0])

        return self._decode_quantized(content, speaker)

    def reconstruct(
        self,
        mel: Tensor,
        embedding: Tensor,
        layers: int,
    ) -> Tensor:
        """Reconstruct a training block through a selected quantizer prefix."""

        content, _ = self._encode_content(mel, embedding, layers)
        speaker, _ = self.refemb_fsq(embedding)
        reconstruction = self._decode_quantized(content, speaker)

        return reconstruction[:, :, : mel.size(-1)]

    def forward(self, mel: Tensor, prompt: Tensor, layers: int) -> Tensor:
        """Reconstruct one block with differentiable speaker and content paths."""

        return self.reconstruct(mel, self.ref_encoder(prompt), layers)
