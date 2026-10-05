import json
from pathlib import Path
from typing import Protocol, cast

import torch

from lana.config import CodecArchitecture, Config

FORMAT_VERSION = 2
MODEL_TYPE = "lana-residual-fsq"


class ConfigurableCodec(Protocol):
    """Exported configuration method required by scalable codec artifacts."""

    def set_num_quantizers(self, layers: int) -> None:
        """Select a trained residual prefix."""

        ...


def signal_signature(config: Config) -> dict[str, object]:
    """Describe frontend values and block conditioning affecting codec inputs."""

    audio = config.audio
    fields = (
        "sample_rate",
        "n_fft",
        "hop_size",
        "win_size",
        "fmin",
        "fmax",
        "n_mels",
        "stft_pad_mode",
        "min_level_db",
        "ref_level_db",
        "max_abs_value",
        "min_db",
        "signal_normalization",
        "allow_clipping_in_normalization",
        "symmetric_acoustic",
    )
    audio_values = audio.model_dump(mode="json")

    return {
        "audio": {name: audio_values[name] for name in fields},
        "chunk_frames": config.streaming.chunk_frames,
        "prompt_frames": config.streaming.prompt_frames,
        "new_embedding_weight": config.streaming.new_embedding_weight,
    }


def load_codec(config: Config, device: torch.device) -> torch.jit.ScriptModule:
    """Load a trusted artifact and validate its training and inference contract."""

    config.validate_codec_frontend()
    path: Path = config.models.codec
    if not path.is_file():
        raise FileNotFoundError(f"Codec model not found: {path}")

    extra_files: dict[str, str | bytes] = {"lana.json": ""}
    model = torch.jit.load(
        str(path),
        map_location=device,
        _extra_files=extra_files,
    )
    metadata_json = cast(str | bytes, extra_files["lana.json"])
    if not metadata_json:
        raise ValueError("The codec has no Lana metadata.")

    metadata = json.loads(metadata_json)
    if not isinstance(metadata, dict):
        raise TypeError("Invalid codec metadata.")
    if (
        metadata.get("format_version") != FORMAT_VERSION
        or metadata.get("model_type") != MODEL_TYPE
    ):
        raise ValueError("Unsupported codec artifact format.")
    if metadata.get("signal") != signal_signature(config):
        raise ValueError("Codec training and inference signal settings differ.")

    architecture = CodecArchitecture.model_validate(metadata["architecture"])
    if config.codec.num_quantizers > architecture.num_quantizers:
        raise ValueError("Requested quantizer count exceeds the trained model.")
    if config.streaming.chunk_frames % architecture.frame_stride:
        raise ValueError("Streaming chunks must align with the codec frame stride.")

    cast(ConfigurableCodec, model).set_num_quantizers(config.codec.num_quantizers)
    model.eval()

    return model
