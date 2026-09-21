import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lana.training.config import CodecArchitecture


class FiniteScalarQuantizer(nn.Module):
    """Bound and round independent scalar channels into mixed-radix indices.

    Training uses a straight-through estimator for rounding. Index conversion
    remains discrete. Quantization arithmetic runs in float32 even when the
    surrounding convolutions use mixed precision.
    """

    def __init__(self, levels: tuple[int, ...]) -> None:
        """Register scalar levels and mixed-radix multipliers as model buffers."""

        super().__init__()
        self.codebook_size = math.prod(levels)

        basis = [math.prod(levels[:index]) for index in range(len(levels))]
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.long))
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.long))

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize the last dimension and return normalized values and indices."""

        values = values.float()
        level_values = cast(Tensor, self.levels)
        basis = cast(Tensor, self.basis)
        levels = level_values.float()
        half_width = torch.floor(levels / 2)
        bound = (levels - 1) * 1.001 / 2
        offset = torch.where(level_values % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / bound)

        bounded = torch.tanh(values + shift) * bound - offset
        rounded = bounded.round()
        rounded = torch.maximum(rounded, -half_width)
        rounded = torch.minimum(rounded, levels - 1 - half_width)

        straight_through = bounded + (rounded - bounded).detach()
        quantized = straight_through / half_width
        digits = rounded.to(torch.long) + level_values // 2
        indices = (digits * basis).sum(dim=-1)

        return quantized, indices

    @torch.jit.export
    def decode_indices(self, indices: Tensor) -> Tensor:
        """Convert legal mixed-radix indices into normalized scalar channels."""

        indices = indices.to(torch.long)
        if bool(torch.any(indices < 0)) or bool(
            torch.any(indices >= self.codebook_size)
        ):
            raise ValueError("Quantizer index lies outside the codebook.")

        basis = cast(Tensor, self.basis)
        level_values = cast(Tensor, self.levels)
        digits = (
            torch.div(
                indices.unsqueeze(-1),
                basis,
                rounding_mode="floor",
            )
            % level_values
        )
        half_width = level_values // 2
        values = (digits - half_width).float() / half_width.float()

        return values


class ResidualBlock(nn.Module):
    """Channel-normalized residual convolution without temporal normalization.

    Layer normalization acts independently at each time position. Symmetric
    convolution padding supplies finite lookahead rather than causal streaming.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        """Construct a depthwise temporal filter and pointwise channel mixer."""

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
        """Apply local normalization, temporal filtering, and residual mixing."""

        hidden = self.normalization(values.transpose(1, 2)).transpose(1, 2)
        hidden = self.temporal(hidden)
        hidden = self.contract(F.gelu(self.expand(hidden)))

        return values + self.scale * hidden


def _residual_stack(
    channels: int,
    kernel_size: int,
    layers: int,
) -> nn.Sequential:
    """Create a bounded repeating dilation schedule for temporal context."""

    return nn.Sequential(
        *[
            ResidualBlock(channels, kernel_size, 2 ** (index % 3))
            for index in range(layers)
        ]
    )


class TrainableCodec(nn.Module):
    """FSQ Mel codec implementing the inference application's exported interface.

    Inputs and outputs use the codec's [0, 1] Mel domain. Content indices have
    shape (batch, tokens); speaker indices have shape (batch, 1). Decoding emits
    frame_stride frames per token, including any final partial input group.
    """

    def __init__(self, config: CodecArchitecture) -> None:
        """Build content, speaker, and reconstruction networks from configuration."""

        super().__init__()
        channels = config.hidden_channels
        content_dimensions = len(config.content_levels)
        speaker_dimensions = len(config.speaker_levels)

        self.n_mels = config.n_mels
        self.frame_stride = config.frame_stride

        self.content_frontend = nn.Sequential(
            nn.Conv1d(
                config.n_mels,
                channels,
                kernel_size=2 * config.frame_stride + 1,
                stride=config.frame_stride,
                padding=config.frame_stride,
            ),
            _residual_stack(
                channels,
                config.kernel_size,
                config.encoder_layers,
            ),
        )
        self.content_condition = nn.Linear(speaker_dimensions, channels)
        self.content_projection = nn.Conv1d(channels, content_dimensions, 1)
        self.content_quantizer = FiniteScalarQuantizer(config.content_levels)

        self.speaker_frontend = nn.Sequential(
            nn.Conv1d(config.n_mels, channels, 5, stride=2, padding=2),
            _residual_stack(
                channels,
                config.kernel_size,
                config.speaker_layers,
            ),
        )
        self.speaker_projection = nn.Linear(channels, speaker_dimensions)
        self.speaker_quantizer = FiniteScalarQuantizer(config.speaker_levels)

        self.decoder_content = nn.Conv1d(content_dimensions, channels, 1)
        self.decoder_speaker = nn.Linear(speaker_dimensions, channels)
        self.decoder = nn.Sequential(
            _residual_stack(
                channels,
                config.kernel_size,
                config.decoder_layers,
            ),
            nn.Conv1d(channels, config.n_mels, 5, padding=2),
            nn.Sigmoid(),
        )

    @torch.jit.export
    def ref_encoder(self, prompt: Tensor) -> Tensor:
        """Pool a prompt into an unquantized speaker-conditioning vector."""

        features = self.speaker_frontend(prompt)
        embedding = self.speaker_projection(features.mean(dim=-1))

        return embedding

    @torch.jit.export
    def refemb_fsq(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize speaker conditioning and retain a singleton token dimension."""

        quantized, indices = self.speaker_quantizer(embedding)

        return quantized, indices.unsqueeze(-1)

    @torch.jit.export
    def encode_ref(
        self,
        mel: Tensor,
        embedding: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode content with external speaker conditioning and FSQ rounding.

        FSQ has no learned codebook or commitment objective. The third return
        value is a scalar zero retained for the existing exported interface.
        """

        features = self.content_frontend(mel)
        features = features + self.content_condition(embedding).unsqueeze(-1)
        values = self.content_projection(features).transpose(1, 2)
        quantized, indices = self.content_quantizer(values)

        return quantized.transpose(1, 2), indices, mel.new_zeros(())

    @torch.jit.export
    def encode(
        self,
        mel: Tensor,
        prompt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return quantized content, content codes, zero loss, and speaker codes."""

        embedding = self.ref_encoder(prompt)
        speaker, speaker_codes = self.refemb_fsq(embedding)
        content, content_codes, commitment = self.encode_ref(mel, embedding)

        return content, content_codes, commitment, speaker, speaker_codes

    def _decode_quantized(self, content: Tensor, speaker: Tensor) -> Tensor:
        """Reconstruct Mels from quantized values with differentiable gradients."""

        hidden = self.decoder_content(content)
        hidden = hidden + self.decoder_speaker(speaker).unsqueeze(-1)
        hidden = torch.repeat_interleave(hidden, self.frame_stride, dim=-1)

        return self.decoder(hidden)

    @torch.jit.export
    def decode(self, codes: Tensor, speaker_codes: Tensor) -> Tensor:
        """Decode integer content and speaker indices using the same decoder."""

        if codes.dim() != 2:
            raise ValueError("Content codes must have shape (batch, tokens).")
        if speaker_codes.dim() != 2 or speaker_codes.size(1) != 1:
            raise ValueError("Speaker codes must have shape (batch, 1).")
        if codes.size(0) != speaker_codes.size(0):
            raise ValueError("Content and speaker batch sizes must agree.")

        content = self.content_quantizer.decode_indices(codes).transpose(1, 2)
        speaker = self.speaker_quantizer.decode_indices(speaker_codes[:, 0])

        return self._decode_quantized(content, speaker)

    def forward(self, mel: Tensor, prompt: Tensor) -> Tensor:
        """Train through quantized values rather than nondifferentiable indices."""

        content, _, _, speaker, _ = self.encode(mel, prompt)
        reconstruction = self._decode_quantized(content, speaker)

        return reconstruction[:, :, : mel.size(-1)]
