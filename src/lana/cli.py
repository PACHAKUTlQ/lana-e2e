import argparse
import sys
from pathlib import Path

import numpy as np

from lana.artifacts import Tokens, load_array, save_array
from lana.config import Config, Feature
from lana.pipeline import Lana


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add configuration and device options shared by inference subcommands."""

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Application configuration file.",
    )
    parser.add_argument(
        "--device",
        help="Override the configured Torch device.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build commands for training, comparison, and inference operations."""

    parser = argparse.ArgumentParser(
        prog="lana",
        description="Mel codec training, SingleCodec inference, and DSPGAN synthesis.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser(
        "train",
        help="Train and export a new FSQ Mel codec.",
    )
    train.add_argument(
        "--config",
        type=Path,
        default=Path("training.json"),
        help="Training configuration file.",
    )
    train.add_argument(
        "--resume",
        type=Path,
        help="Resume a trusted Lightning training checkpoint.",
    )

    comparison = commands.add_parser(
        "compare-mel",
        help="Plot aligned Mel-power spectrograms of two audio files.",
    )
    comparison.add_argument("original", type=Path)
    comparison.add_argument("generated", type=Path)
    comparison.add_argument("output", type=Path)
    comparison.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Configuration supplying the audio frontend.",
    )
    comparison.add_argument(
        "--max-shift-seconds",
        type=float,
        default=2.0,
        help="Maximum alignment offset in seconds.",
    )

    encode = commands.add_parser(
        "encode", help="Encode audio into content and speaker codes."
    )
    _add_common_arguments(encode)
    encode.add_argument("input", type=Path)
    encode.add_argument("content", type=Path)
    encode.add_argument("speaker", type=Path)
    encode.add_argument("--seed", type=int)
    encode.add_argument("--prompt-length", type=int)

    decode = commands.add_parser(
        "decode", help="Decode content and speaker codes into audio."
    )
    _add_common_arguments(decode)
    decode.add_argument("content", type=Path)
    decode.add_argument("speaker", type=Path)
    decode.add_argument("output", type=Path)

    decode_mel = commands.add_parser(
        "decode-mel", help="Synthesize a normalized Mel NPY file."
    )
    _add_common_arguments(decode_mel)
    decode_mel.add_argument("input", type=Path)
    decode_mel.add_argument("output", type=Path)

    infer = commands.add_parser("infer", help="Run chunked audio reconstruction.")
    _add_common_arguments(infer)
    infer.add_argument("input", type=Path)
    infer.add_argument("output", type=Path)
    infer.add_argument("--chunk-size", type=int)
    infer.add_argument("--prompt-length", type=int)
    infer.add_argument("--new-emb-weight", type=float)
    infer.add_argument("--mel-cache-length", type=int)
    infer.add_argument("--minimum-chunk-frames", type=int)
    infer.add_argument("--quiet", action="store_true")

    extract = commands.add_parser(
        "extract", help="Extract acoustic features into NPY files."
    )
    _add_common_arguments(extract)
    extract.add_argument(
        "input",
        type=Path,
        help="A WAV file, a directory of WAV files, or a text manifest.",
    )
    extract.add_argument("output", type=Path)
    extract.add_argument(
        "--features",
        nargs="+",
        type=Feature,
        choices=list(Feature),
        default=[Feature.MEL],
    )
    extract.add_argument("--workers", type=int, default=20)
    extract.add_argument("--incremental", action="store_true")
    extract.add_argument("--quiet", action="store_true")

    convert = commands.add_parser(
        "linear-to-mel",
        help="Convert a frontend-scaled linear NPY file into Mel features.",
    )
    _add_common_arguments(convert)
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)

    return parser


def _load_config(arguments: argparse.Namespace) -> Config:
    """Load JSON and revalidate explicitly supplied inference overrides."""

    config = Config.from_file(arguments.config)
    values = config.model_dump(mode="python")

    if arguments.device is not None:
        values["device"] = arguments.device

    if arguments.command == "encode":
        encoding = config.encoding.model_dump(mode="python")
        for name in ("seed", "prompt_length"):
            value = getattr(arguments, name)
            if value is not None:
                encoding[name] = value
        values["encoding"] = encoding

    if arguments.command == "infer":
        inference = config.inference.model_dump(mode="python")
        for name in (
            "chunk_size",
            "prompt_length",
            "new_emb_weight",
            "mel_cache_length",
            "minimum_chunk_frames",
        ):
            value = getattr(arguments, name)
            if value is not None:
                inference[name] = value
        values["inference"] = inference

    return Config.model_validate(values)


def _execute(arguments: argparse.Namespace) -> None:
    """Dispatch commands without loading training or plotting code unnecessarily."""

    if arguments.command == "train":
        from lana.training import train

        train(arguments.config, resume=arguments.resume)

        return

    if arguments.command == "compare-mel":
        from lana.mel_comparison import plot_mel_comparison

        destination = plot_mel_comparison(
            arguments.original,
            arguments.generated,
            arguments.output,
            config_path=arguments.config,
            max_shift_seconds=arguments.max_shift_seconds,
        )
        print(destination)

        return

    application = Lana(_load_config(arguments))

    match arguments.command:
        case "encode":
            tokens = application.encode(arguments.input)
            tokens.save(arguments.content, arguments.speaker)
        case "decode":
            tokens = Tokens.load(arguments.content, arguments.speaker)
            application.decode(tokens).save(arguments.output)
        case "decode-mel":
            mel = np.asarray(load_array(arguments.input), dtype=np.float32)
            application.decode_mel(mel).save(arguments.output)
        case "infer":
            application.infer(
                arguments.input,
                progress=not arguments.quiet,
            ).save(arguments.output)
        case "extract":
            application.extract_many(
                arguments.input,
                arguments.output,
                arguments.features,
                workers=arguments.workers,
                incremental=arguments.incremental,
                progress=not arguments.quiet,
            )
        case "linear-to-mel":
            linear = np.asarray(load_array(arguments.input), dtype=np.float32)
            mel = application.convert_linear_to_mel(linear)
            save_array(arguments.output, mel)
        case _:
            raise ValueError(f"Unknown command: {arguments.command}")


def _format_error(error: BaseException) -> str:
    """Format explicit exception causes without discarding the original failure."""

    messages = [str(error)]
    cause = error.__cause__
    while cause is not None:
        messages.append(str(cause))
        cause = cause.__cause__

    return "\nCaused by: ".join(messages)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        _execute(arguments)
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print(f"lana: {_format_error(error)}", file=sys.stderr)

        return 1
    except KeyboardInterrupt:
        print("lana: interrupted", file=sys.stderr)

        return 130

    return 0
