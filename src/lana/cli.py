import argparse
import sys
from pathlib import Path

import numpy as np

from lana.artifacts import Tokens, load_array, save_array
from lana.config import Config, Feature
from lana.runtime.pipeline import Lana


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add application configuration and optional device selection."""

    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--device", help="Override the configured Torch device.")


def build_parser() -> argparse.ArgumentParser:
    """Build the unified training, inference, extraction, and streaming CLI."""

    parser = argparse.ArgumentParser(
        prog="lana",
        description="Scalable Mel codec training and streaming synthesis.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train and export a scalable codec.")
    train.add_argument("--config", type=Path, default=Path("config.json"))
    train.add_argument("--resume", type=Path)

    encode = commands.add_parser("encode", help="Encode an utterance into NPY codes.")
    add_common_arguments(encode)
    encode.add_argument("input", type=Path)
    encode.add_argument("content", type=Path)
    encode.add_argument("speaker", type=Path)
    encode.add_argument("--seed", type=int)
    encode.add_argument("--prompt-length", type=int)

    decode = commands.add_parser("decode", help="Decode NPY codes into audio.")
    add_common_arguments(decode)
    decode.add_argument("content", type=Path)
    decode.add_argument("speaker", type=Path)
    decode.add_argument("output", type=Path)

    decode_mel = commands.add_parser("decode-mel", help="Synthesize a Mel NPY file.")
    add_common_arguments(decode_mel)
    decode_mel.add_argument("input", type=Path)
    decode_mel.add_argument("output", type=Path)

    infer = commands.add_parser("infer", help="Run offline chunked reconstruction.")
    add_common_arguments(infer)
    infer.add_argument("input", type=Path)
    infer.add_argument("output", type=Path)
    infer.add_argument("--quiet", action="store_true")

    extract = commands.add_parser("extract", help="Extract acoustic feature files.")
    add_common_arguments(extract)
    extract.add_argument("input", type=Path)
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

    convert = commands.add_parser("linear-to-mel", help="Convert linear features.")
    add_common_arguments(convert)
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)

    send = commands.add_parser("stream-send", help="Stream an encoded audio file.")
    add_common_arguments(send)
    send.add_argument("input", type=Path)
    send.add_argument("--host", default="127.0.0.1")
    send.add_argument("--port", type=int, default=8765)
    send.add_argument("--no-realtime", action="store_true")

    receive = commands.add_parser(
        "stream-receive",
        help="Receive one stream into a WAV file.",
    )
    add_common_arguments(receive)
    receive.add_argument("output", type=Path)
    receive.add_argument("--host", default="127.0.0.1")
    receive.add_argument("--port", type=int, default=8765)

    return parser


def load_config(arguments: argparse.Namespace) -> Config:
    """Apply explicit CLI overrides through configuration validation."""

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

    return Config.model_validate(values)


def execute(arguments: argparse.Namespace) -> None:
    """Dispatch commands while keeping training imports outside inference paths."""

    if arguments.command == "train":
        from lana.training.runner import train

        train(arguments.config, resume=arguments.resume)

        return

    config = load_config(arguments)
    if arguments.command in {"stream-send", "stream-receive"}:
        from lana.streaming.network import receive_file, send_file

        if not 1 <= arguments.port <= 65535:
            raise ValueError("Port must be between 1 and 65535.")
        if arguments.command == "stream-send":
            send_file(
                arguments.input,
                host=arguments.host,
                port=arguments.port,
                config=config,
                realtime=not arguments.no_realtime,
            )
        else:
            receive_file(
                arguments.output,
                host=arguments.host,
                port=arguments.port,
                config=config,
            )

        return

    application = Lana(config)
    match arguments.command:
        case "encode":
            application.encode(arguments.input).save(
                arguments.content,
                arguments.speaker,
            )
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
            save_array(arguments.output, application.convert_linear_to_mel(linear))
        case _:
            raise ValueError(f"Unknown command: {arguments.command}")


def format_error(error: BaseException) -> str:
    """Include explicit exception causes without losing the originating failure."""

    messages = [str(error)]
    seen = {id(error)}
    cause = error.__cause__
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        messages.append(str(cause))
        cause = cause.__cause__

    return "\nCaused by: ".join(messages)


def main(argv: list[str] | None = None) -> int:
    """Run the application command and return its process exit status."""

    arguments = build_parser().parse_args(argv)
    try:
        execute(arguments)
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print(f"lana: {format_error(error)}", file=sys.stderr)

        return 1
    except KeyboardInterrupt:
        print("lana: interrupted", file=sys.stderr)

        return 130

    return 0
